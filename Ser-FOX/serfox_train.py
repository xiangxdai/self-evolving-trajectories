"""
Ser-FOX training: self-evolving serialized-trajectory learning.

The model emits a response as a serialized sequence of (index, value) pairs appended
to the quiz/prompt:

    [quiz]  I_{p1} y_{p1}  I_{p2} y_{p2}  ...  I_{pK} y_{pK}

Each index token I_p is an ABSOLUTE position in [index_token_start, index_token_start +
response_size); y_p is the value placed at that position. The model therefore chooses
BOTH the order in which it fills positions AND the value at each one, so it can learn an
easy-to-hard solving order instead of a fixed left-to-right one.

Training uses round boundaries (default num_rounds=10; round_interval = max_iters // num_rounds)
for checkpointing, evaluation, and optional trajectory regeneration. Round 1 trains on
canonical / L2R trajectories; later rounds regenerate trajectories from the model's own
confidence order (self-evolving), mixing three sources (main / prev-pool / canonical, see
--mix_ratios). The index target is either a hard CE (one-hot on the chosen position) or a
SOFT distribution over remaining positions (--index_loss_mode soft; see serfox_soft_index.py
for score modes such as logit_margin/uniform).

Single GPU, example:
$ python serfox_train.py --dataset sudoku --test_file data/sudoku_test5k.jsonl \
      --n_layer 3 --n_head 12 --n_embd 384 --max_iters 100000

DDP on 4 GPUs (1 node), example:
$ torchrun --standalone --nproc_per_node=4 serfox_train.py --dataset sudoku \
      --test_file data/sudoku_test5k.jsonl --n_layer 3 --n_head 12 --n_embd 384 --max_iters 100000

Round-2 soft-index example:
$ python serfox_train.py --dataset sudoku --test_file data/sudoku_test5k.jsonl \
      --index_loss_mode soft --soft_index_score_mode logit_margin --mix_ratios 0.7,0.2,0.1 ...

Run `python serfox_train.py --help` for the full flag list. (If your cluster has no
Infiniband interconnect, prepend NCCL_IB_DISABLE=1.)
"""

import csv
import os
import sys
import time
import math
import pickle
import json
from contextlib import nullcontext
import argparse
from pathlib import Path
from datetime import datetime, timedelta


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from serfox_model import GPTConfig, GPT, sample_positions_from_scores
from serfox_soft_index import (
    mixed_soft_index_ar_loss,
    soft_index_distribution_from_value_logits,
)
from logger import get_logger
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.serfox_task_configs import (
    REGIME_MIX_RATIOS,
    get_task_config,
    task_names,
)


# -----------------------------------------------------------------------------
# the input parameters

def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def explicit_cli_options(argv):
    """Return argparse destination names explicitly supplied on the CLI."""
    aliases = {
        "config": "task",
        "no_compile": "compile",
    }
    options = set()
    for token in argv[1:]:
        if not token.startswith("--"):
            continue
        option = token[2:].split("=", 1)[0].replace("-", "_")
        options.add(aliases.get(option, option))
    return options


parser = argparse.ArgumentParser(description='Train Ser-FOX (configurable loss mode and order shuffling).')

parser.add_argument('--task', '--config', dest='task', type=str, default=None, choices=task_names(), help='Apply a saved Ser-FOX task/config profile from configs/serfox_task_configs.py')
parser.add_argument('--rounds', type=int, default=None, help='Training rounds; with --task, max_iters defaults to rounds * task round_interval')
parser.add_argument('--regime', type=str, default=None, choices=tuple(REGIME_MIX_RATIOS), help='Optional saved mix recipe: warm, warmmix, warmbest, or warmmixbest')
parser.add_argument('--backbone', type=str, default='small', choices=('small', 'general'), help="With --task: 'small'=task's own original size, 'general'=shared 3-12-384 backbone (sudoku is 3-12-384 either way)")
parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=210, help='Embedding dimension')
parser.add_argument('--max_iters', type=int, default=500000, help='Number of Iterations (default: 500000)')
parser.add_argument('--train_batch_size', type=int, default=256, help='Training micro-batch size')
parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size for train/val/test loss evaluation')
parser.add_argument('--eval_decode_batch_size', type=int, default=256, help='Batch size for AR/PI accuracy decode eval. Decode is per-sample independent, so batching does not change results, only speed (the old per-sample loop is ~B times slower). Set 1 for the legacy per-sample path.')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Global gradient accumulation steps')
parser.add_argument('--round_interval', type=int, default=None, help='Round length for LR schedule and trajectory regeneration; default is max_iters // 10')
parser.add_argument('--eval_interval', type=int, default=None, help='Estimate train/val/test losses every N iterations; default is 500')
parser.add_argument('--checkpoint_interval', type=int, default=5000, help='Save checkpoints and run test accuracy every N iterations; default is 5000')
parser.add_argument('--eval_iters', type=int, default=None, help='Number of batches used to estimate train/val/test loss')
parser.add_argument('--compile', type=parse_bool, nargs='?', const=True, default=True, help='Use torch.compile for the training forward path')
parser.add_argument('--no_compile', action='store_false', dest='compile', help='Disable torch.compile')
parser.add_argument('--regen_blocks_per_step', type=int, default=1000, help='Number of base samples processed per trajectory regeneration chunk')
parser.add_argument('--regen_max_blocks', type=int, default=0, help='If >0, regenerate only this many randomly chosen base samples per round boundary instead of the full set (deterministic per-round subset seeded by iter). Big speedup on large datasets (e.g. 900k-block path/sudoku); training resamples rows from the regenerated pool anyway, so a 200k subset still gives each row ~256 visits per 50k-iter round.')
parser.add_argument('--regen_position_temperature', type=float, default=0.0, help='FOX regeneration temperature over trajectory positions only; <=0 keeps deterministic argmax ordering')
parser.add_argument('--dataset', type=str, default=None, help='Dataset path relative to data/ (e.g. cd/cd3/k1); default keeps legacy root meta.pkl/base.bin')
parser.add_argument('--test_file', type=str, default=None, help='Optional test file for eval_interval test loss and per-round exact-match eval (.bin, .jsonl, or .json)')
parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'], help='Start from scratch or resume from a checkpoint')
parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint path/name to resume from; defaults to latest *_ckpt.pt in out_dir')
parser.add_argument('--out_dir', type=str, default=None, help='Explicit output/resume directory; relative paths are resolved from the repo root')
parser.add_argument('--run_name', type=str, default=None, help='Optional suffix appended to the default output directory name')
parser.add_argument('--no_timestamp', action='store_true', help='Keep the legacy deterministic output directory for scratch runs')
parser.add_argument('--no_regen_kv_cache', action='store_true', help='Disable KV-cache acceleration during trajectory regeneration')
parser.add_argument('--first_round_l2r', type=parse_bool, nargs='?', const=True, default=True, help='DEPRECATED / no-op. train_0.bin is now ALWAYS built in L2R base order; use --shuffle_order ALONE to control whether round-1 trains shuffled (true) or as-is L2R (false). Accepted but ignored so existing launch scripts do not break.')
parser.add_argument('--loss_mode', type=str, default='all', choices=['all', 'value_only'], help='Loss on all serialized tokens (V1) or only value tokens (V2); default: all')
parser.add_argument('--skip_loss_eval', type=parse_bool, nargs='?', const=True, default=False, help='Skip estimate_loss (train/val/test loss); keeps only AR/PI accuracy eval. Big speedup when soft-index val eval triggers slow online build_soft_index_targets.')
parser.add_argument('--index_loss_mode', type=str, default='hard', choices=['hard', 'soft'], help="OPTIONAL soft-index toggle. 'hard' (DEFAULT) = standard next-token cross-entropy on index tokens; soft-index is OFF. 'soft' = distribution distillation: from round 2 onward, each regenerated trajectory also yields a soft target over the remaining reveal positions, trained via mixed_soft_index_ar_loss (see serfox_soft_index.py). Round 1 ALWAYS uses hard regardless of this flag. Enable with: --index_loss_mode soft (then tune --soft_index_score_mode / --soft_index_temperature).")
parser.add_argument('--soft_index_score_mode', type=str, default='p1-p2', choices=['argmax', 'entropy', 'p1-p2', 'logit_margin', 'gt_prob', 'gt_logprob', 'uniform'], help='[only used when --index_loss_mode soft] Score for soft index targets from parallel value distributions; p1-p2 = prob margin (saturates to 1.0 in confident region -> soft target collapses toward uniform), logit_margin = raw-logit top1-top2 (unbounded, preserves full clue>easy>hard order, pair with soft_index_temperature ~5), uniform = 1/k order-invariance prior')
parser.add_argument('--soft_index_temperature', type=float, default=1.0, help='[only used when --index_loss_mode soft] Temperature for soft index target distribution; default 1.0')
parser.add_argument('--shuffle_order', type=parse_bool, nargs='?', const=True, default=False, help='Randomly permute (index, value) pair order per train sample. FOX round design: ROUND-1 (iter<round_interval) shuffles ALL rows -> model learns ORDER-INVARIANCE -> HIGH PI, and this round-1 model is what generates the round-2 trajectories. ROUND-2+ shuffles ONLY the canonical/L2R mix rows (regen rows are kept in their confident-first easy-to-hard order, NOT shuffled, which is what boosts AR via self-distillation). default: False')
parser.add_argument('--first_round_only', type=parse_bool, nargs='?', const=True, default=False, help='Stop after round 1 (skip trajectory regeneration); default: False')
parser.add_argument('--warm_from_best_round', type=parse_bool, nargs='?', const=True, default=False, help='At each round boundary, reload the highest-AR checkpoint of the just-finished round (overfit guard: stop scanning after 2 consecutive AR drops) before regen+continue, instead of warm-starting from the last ckpt; default: False')
parser.add_argument('--mix_ratios', type=str, default='1.0,0.0,0.0', help='Sampling ratios for [Main, Prev, Canonical] in round 2+; 1.0,0.0,0.0 = pure regen (default), 0.7,0.2,0.1 = warm+mix')
parser.add_argument('--regen_ckpt', type=str, default=None, help='Use this checkpoint model for trajectory regeneration instead of the current training model')
parser.add_argument('--online_regen_sample', type=parse_bool, nargs='?', const=True, default=False, help='In round 2+, resample trajectories on every train batch instead of reusing train_<round>.bin')
parser.add_argument('--online_regen_method', type=str, default='softmax', choices=['softmax', 'gaussian'], help='Online trajectory sampling rule: softmax over p_gt or argmax(p_gt + Gaussian noise)')
parser.add_argument('--online_regen_temperature', type=float, default=0.1, help='Temperature tau for online softmax(p_gt / tau); tau is also the Gumbel noise scale')
parser.add_argument('--online_regen_noise_std', type=float, default=0.1, help='Gaussian noise std for online argmax(p_gt + N(0, std))')
parser.add_argument('--learning_rate', type=float, default=None, help='Optional explicit peak learning rate; bypasses auto-scaling from global_batch/base_batch_size')

explicit_options = explicit_cli_options(sys.argv)
args = parser.parse_args()

active_task_config = None
active_task_metadata = None
if args.rounds is not None and args.rounds < 1:
    raise ValueError(f"--rounds must be >= 1, got {args.rounds}")
if args.task is not None:
    active_task_config = get_task_config(args.task)
    task_defaults = active_task_config.as_arg_defaults(
        rounds=args.rounds,
        regime=args.regime,
        backbone=args.backbone,
    )
    for key, value in task_defaults.items():
        if key not in explicit_options:
            setattr(args, key, value)
    if "max_iters" not in explicit_options:
        selected_rounds = args.rounds if args.rounds is not None else active_task_config.rounds
        args.max_iters = args.round_interval * selected_rounds
    if (
        args.regime is not None
        and args.regime.endswith("best")
        and "warm_from_best_round" not in explicit_options
    ):
        args.warm_from_best_round = True
    active_task_metadata = active_task_config.as_metadata(
        rounds=args.rounds,
        regime=args.regime,
        backbone=args.backbone,
    )
    active_task_metadata["selected_round_interval"] = args.round_interval
    active_task_metadata["selected_max_iters"] = args.max_iters
elif args.rounds is not None:
    inferred_round_interval = args.round_interval if args.round_interval is not None else max(1, args.max_iters // 10)
    if args.round_interval is None:
        args.round_interval = inferred_round_interval
    if "max_iters" not in explicit_options:
        args.max_iters = args.round_interval * args.rounds

n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
regen_blocks_per_step = args.regen_blocks_per_step
regen_max_blocks = args.regen_max_blocks
regen_position_temperature = args.regen_position_temperature
dataset = args.dataset
init_from = args.init_from
run_name = args.run_name
no_timestamp = args.no_timestamp
explicit_out_dir = args.out_dir
use_regen_kv_cache = not args.no_regen_kv_cache
loss_mode = args.loss_mode
skip_loss_eval = args.skip_loss_eval
index_loss_mode = args.index_loss_mode
soft_index_score_mode = args.soft_index_score_mode
soft_index_temperature = args.soft_index_temperature
shuffle_order = args.shuffle_order
first_round_only = args.first_round_only
warm_from_best_round = args.warm_from_best_round
online_regen_sample = args.online_regen_sample
online_regen_method = args.online_regen_method
online_regen_temperature = args.online_regen_temperature
online_regen_noise_std = args.online_regen_noise_std
if soft_index_temperature <= 0:
    raise ValueError(f'--soft_index_temperature must be > 0, got {soft_index_temperature}')
if online_regen_temperature <= 0:
    raise ValueError(f'--online_regen_temperature must be > 0, got {online_regen_temperature}')
if online_regen_noise_std < 0:
    raise ValueError(f'--online_regen_noise_std must be >= 0, got {online_regen_noise_std}')
mix_ratios = [float(x) for x in args.mix_ratios.split(',')]
if len(mix_ratios) != 3:
    raise ValueError(f"--mix_ratios must have exactly 3 values [Main, Prev, Canonical], got {len(mix_ratios)}")
if abs(sum(mix_ratios) - 1.0) > 1e-5:
    raise ValueError(f"--mix_ratios must sum to 1.0, got {sum(mix_ratios)}")
if active_task_config is not None and int(os.environ.get("RANK", "-1")) in {-1, 0}:
    selected_rounds = active_task_metadata["selected_rounds"]
    print(
        f"Using Ser-FOX task config '{args.task}': "
        f"dataset={args.dataset}, backbone={args.n_layer}-{args.n_head}-{args.n_embd}, "
        f"rounds={selected_rounds}, round_interval={args.round_interval}, "
        f"max_iters={args.max_iters}, lr={args.learning_rate}, "
        f"test_file={args.test_file}"
    )

num_rounds = 10

seed = 1337

OUT_ROOT = REPO_ROOT / "out" / "serfox_train"

data_dir = (REPO_ROOT / "data" / dataset).resolve() if dataset is not None else REPO_ROOT
meta_path = data_dir / "meta.pkl"
base_path = data_dir / "base.bin"
if not base_path.exists():
    base_path = data_dir / "train.bin"
if not meta_path.exists():
    raise FileNotFoundError(f"meta.pkl not found for dataset={dataset!r}: {meta_path}")
if not base_path.exists():
    raise FileNotFoundError(
        f"Neither base.bin nor train.bin found for dataset={dataset!r} in {data_dir}"
    )

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

if 'quiz_size' in meta and 'response_size' in meta:
    quiz_size = meta['quiz_size']
    response_size = meta['response_size']
else:
    # Backward-compatible fallback for the current equal-split layout.
    quiz_size = meta['block_size'] // 2
    response_size = meta['block_size'] - quiz_size

base_seq_len = quiz_size + response_size
train_seq_len = quiz_size + 2 * response_size
value_vocab_size = meta['vocab_size']
_stoi = meta.get("stoi", {})
_pad_token_name = meta.get("pad_token", "<PAD>")
_eos_token_name = meta.get("eos_token", "<EOS>")
pad_id = _stoi.get(_pad_token_name, None)
eos_id = _stoi.get(_eos_token_name, None)
vocab_size = value_vocab_size + response_size
block_size = quiz_size + 3 * response_size


def _resolve_token_id(token_spec, *, arg_name):
    if token_spec is None:
        return None
    if token_spec in _stoi:
        return _stoi[token_spec]
    try:
        token_id = int(token_spec)
    except ValueError as exc:
        raise ValueError(
            f"{arg_name}={token_spec!r} is neither a token in meta.pkl nor an integer token id"
        ) from exc
    if not (0 <= token_id < value_vocab_size):
        raise ValueError(f"{arg_name}={token_id} is outside value vocab range [0, {value_vocab_size})")
    return token_id


loss_tag = "vo" if loss_mode == "value_only" else "all"
if shuffle_order:
    shuf_tag = "shuf"
else:
    shuf_tag = "fix"
idx_loss_tag = "" if index_loss_mode == "hard" else f"_softidx_{soft_index_score_mode.replace('-', '')}"
if dataset is None:
    config_name = f'{n_layer}_{n_head}_{n_embd}'
else:
    safe_dataset = dataset.replace("/", "_").replace("\\", "_")
    config_name = f'{safe_dataset}_{n_layer}_{n_head}_{n_embd}'

base_run_name = f'{config_name}_{loss_tag}_{shuf_tag}{idx_loss_tag}_{seed}'
if explicit_out_dir is not None:
    out_dir_path = Path(explicit_out_dir).expanduser()
    if not out_dir_path.is_absolute():
        out_dir_path = REPO_ROOT / out_dir_path
else:
    run_suffix = run_name
    if run_suffix is None and init_from == 'scratch' and not no_timestamp:
        run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_suffix:
        run_suffix = run_suffix.replace("/", "_").replace("\\", "_").replace(":", "-")
        run_name = run_suffix
        out_dir_path = OUT_ROOT / f'{base_run_name}_{run_suffix}'
    else:
        out_dir_path = OUT_ROOT / base_run_name

if explicit_out_dir is None and init_from == 'resume' and args.resume_ckpt is not None:
    resume_arg = Path(args.resume_ckpt).expanduser()
    for candidate in (resume_arg, REPO_ROOT / resume_arg):
        if candidate.exists() and candidate.is_file():
            out_dir_path = candidate.resolve().parent
            break

out_dir = str(out_dir_path)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
round_interval = args.round_interval if args.round_interval is not None else max(1, max_iters // num_rounds)
eval_interval = args.eval_interval if args.eval_interval is not None else 500
checkpoint_interval = args.checkpoint_interval
log_interval = max(1, round_interval // 100)
eval_iters = args.eval_iters if args.eval_iters is not None else min(200, round_interval)
if round_interval < 1:
    raise ValueError(f"round_interval must be >= 1, got {round_interval}")
if eval_interval < 1:
    raise ValueError(f"eval_interval must be >= 1, got {eval_interval}")
if checkpoint_interval < 1:
    raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")
if eval_iters < 1:
    raise ValueError(f"eval_iters must be >= 1, got {eval_iters}")

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = args.init_from # 'scratch' or 'resume'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = f'serfox4-{n_layer}_{n_head}_{n_embd}_{loss_tag}_{shuf_tag}_{seed}'
# data
#dataset = 'reasoning'
gradient_accumulation_steps = args.gradient_accumulation_steps # used to simulate larger batch sizes
train_batch_size = args.train_batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size
val_batch_size = args.eval_batch_size
eval_decode_batch_size = args.eval_decode_batch_size
batch_size = train_batch_size
#block_size = 64
# model
#n_layer = 1 #12
#n_head = 1 #12
#n_embd = 384 #768


dropout = 0.1
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
base_batch_size = 512
initial_base_lr = 3e-4
learning_rate = initial_base_lr # scaled after DDP setup using the effective batch size
#max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # align FOX with AR/DOG: warmup + cosine decay within each round
warmup_iters = max(1, round_interval // 20) # 5% warmup inside each round
lr_decay_iters = round_interval
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = args.compile # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
if active_task_metadata is not None:
    config["task"] = args.task
    config["task_config"] = active_task_metadata
if args.regime is not None:
    config["regime"] = args.regime
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # Effectively no NCCL timeout (default 600s is too short for long
    # checkpoint-time test eval). Override with DDP_TIMEOUT_HOURS env var
    # if you do want a finite timeout.
    _ddp_timeout_hours = float(os.environ.get('DDP_TIMEOUT_HOURS', str(24 * 365)))
    init_process_group(backend=backend, timeout=timedelta(hours=_ddp_timeout_hours))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * train_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

global_batch_size = train_batch_size * ddp_world_size * gradient_accumulation_steps
lr_scaling_factor = global_batch_size / base_batch_size
if args.learning_rate is not None:
    learning_rate = args.learning_rate
    lr_source = "explicit override"
else:
    learning_rate = initial_base_lr * lr_scaling_factor
    lr_source = "auto-scaled from base"
min_lr = learning_rate / 10
config.update({
    "base_batch_size": base_batch_size,
    "initial_base_lr": initial_base_lr,
    "global_batch_size": global_batch_size,
    "lr_scaling_factor": lr_scaling_factor,
    "learning_rate": learning_rate,
    "min_lr": min_lr,
    "warmup_iters": warmup_iters,
    "lr_decay_iters": lr_decay_iters,
    "decay_lr": decay_lr,
    "gradient_accumulation_steps": gradient_accumulation_steps,
})
print(
    f"FOX learning rate ({lr_source}): peak {learning_rate:.6g}, min {min_lr:.6g}, "
    f"global batch {global_batch_size}, warmup {warmup_iters}, round length {lr_decay_iters}, "
    f"round interval {round_interval}, eval interval {eval_interval}, "
    f"checkpoint interval {checkpoint_interval}, "
    f"loss_mode={loss_mode}, index_loss_mode={index_loss_mode}, "
    f"soft_index_score_mode={soft_index_score_mode}, soft_index_temperature={soft_index_temperature}, "
    f"shuffle_order={shuffle_order}, mix_ratios={mix_ratios}, "
    f"regen_position_temperature={regen_position_temperature}, "
    f"online_regen_sample={online_regen_sample}, online_regen_method={online_regen_method}, "
    f"online_regen_temperature={online_regen_temperature}, online_regen_noise_std={online_regen_noise_std}"
)

if ddp:
    if master_process:
        out_dir_bytes = out_dir.encode("utf-8")
        length_tensor = torch.tensor([len(out_dir_bytes)], dtype=torch.long, device=device)
    else:
        length_tensor = torch.tensor([0], dtype=torch.long, device=device)
    torch.distributed.broadcast(length_tensor, src=0)
    buf = torch.zeros(int(length_tensor.item()), dtype=torch.uint8, device=device)
    if master_process:
        buf[:] = torch.tensor(list(out_dir_bytes), dtype=torch.uint8, device=device)
    torch.distributed.broadcast(buf, src=0)
    out_dir = bytes(buf.cpu().tolist()).decode("utf-8")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"output directory: {out_dir}")
if ddp:
    torch.distributed.barrier()
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


base_data = np.memmap(base_path, dtype=np.uint16, mode='r')
if len(base_data) % base_seq_len != 0:
    raise ValueError(
        f"Training data length {len(base_data)} is not divisible by base_seq_len "
        f"{base_seq_len}: {base_path}"
    )
actual_train_rows = len(base_data) // base_seq_len
config.update({
    "dataset_base_path": str(base_path),
    "dataset_meta_path": str(meta_path),
    "actual_train_rows": actual_train_rows,
})
if master_process:
    print(f"Loaded train data: {base_path} ({actual_train_rows:,} rows)")
batch_offsets = np.arange(train_seq_len, dtype=np.int64)
#print(len(base_data))
#print(xxx)

# poor man's data loader




def _sample_from_source(src, batch_size, data_size, return_row_ids=False):
    row_ids = torch.randint(len(src) // data_size, (batch_size,))
    ix = row_ids * data_size
    offsets = ix.cpu().numpy()[:, None] + batch_offsets[None, :]
    z = torch.from_numpy(src[offsets].astype(np.int64))
    if return_row_ids:
        return z, row_ids.cpu().numpy()
    return z


def _sample_soft_index_targets(soft_src, row_ids):
    if soft_src is None:
        return None
    return torch.from_numpy(np.asarray(soft_src[row_ids], dtype=np.float32))


def _sample_uniform_from_sources(sources_list, count, data_size, soft_sources_list=None):
    """Truly uniform sample across the union of all (file, row) pairs.

    File selection is weighted by per-file row count so every row in every
    file has the same probability of being drawn. In FOX all train_<r>.bin
    files are the same size (one regen pass over base.bin), so the weights
    are uniform in practice; this code path stays correct even if a future
    run produces unequal-size files.

    Each batch element is independently drawn (with replacement).
    """
    if not sources_list or count == 0:
        return None
    n_files = len(sources_list)
    sizes = np.array([len(src) // data_size for src in sources_list], dtype=np.int64)
    if (sizes <= 0).any():
        raise ValueError(f"prev source has zero rows; sizes={sizes.tolist()}")
    file_probs = sizes / sizes.sum()
    file_ids = np.random.choice(n_files, size=count, p=file_probs)
    out = torch.empty(count, data_size, dtype=torch.long)
    soft_out = None
    if soft_sources_list is not None:
        soft_out = torch.empty(count, response_size, response_size, dtype=torch.float32)
    for f in range(n_files):
        mask = file_ids == f
        n_in_file = int(mask.sum())
        if n_in_file == 0:
            continue
        z_f, rows_f = _sample_from_source(
            sources_list[f], n_in_file, data_size, return_row_ids=True
        )
        batch_rows = torch.from_numpy(np.where(mask)[0])
        out[batch_rows] = z_f
        if soft_out is not None:
            soft_f = (
                _sample_soft_index_targets(soft_sources_list[f], rows_f)
                if f < len(soft_sources_list)
                else None
            )
            if soft_f is not None:
                soft_out[batch_rows] = soft_f
            else:
                soft_out = None
    if soft_sources_list is not None:
        return out, soft_out
    return out


online_regen_model = None
online_base_data = np.memmap(base_path, dtype=np.uint16, mode='r')


def _refresh_online_regen_model(reason):
    """Freeze a snapshot used for per-batch stochastic trajectory sampling."""
    global online_regen_model
    source = get_scoring_model(model)
    cfg = source.config
    if online_regen_model is None:
        online_regen_model = GPT(cfg).to(device)
        for p in online_regen_model.parameters():
            p.requires_grad_(False)
    online_regen_model.load_state_dict(source.state_dict())
    online_regen_model.eval()
    if master_process:
        msg = f"Online regen snapshot refreshed ({reason}); method={online_regen_method}, tau={online_regen_temperature}, noise_std={online_regen_noise_std}"
        print(msg)
        logger.info(msg)


@torch.inference_mode()
def _sample_online_regen_batch(count):
    """Build a fresh serialized FOX trajectory batch from base.bin/train.bin."""
    if online_regen_model is None:
        raise RuntimeError("online_regen_model is not initialized")
    scoring_model = online_regen_model
    cfg = scoring_model.config
    num_rows = len(online_base_data) // cfg.base_seq_len
    ix = np.random.randint(0, num_rows, size=count)
    z = torch.stack([
        torch.from_numpy(online_base_data[i * cfg.base_seq_len:(i + 1) * cfg.base_seq_len].astype(np.int64))
        for i in ix
    ]).to(device)

    z_basic = z[:, :cfg.quiz_size]
    z_result = z[:, cfg.quiz_size:cfg.quiz_size + cfg.response_size]
    bsz = z_basic.shape[0]
    index_tokens = torch.arange(0, cfg.num_index_tokens, dtype=torch.long, device=device).unsqueeze(0)
    index_tokens = (index_tokens + cfg.index_token_start).expand(bsz, -1)

    kv_cache = None
    if use_regen_kv_cache:
        with ctx:
            kv_cache = scoring_model.build_prefix_kv_cache(z_basic)

    for _ in range(cfg.response_size):
        with ctx:
            if kv_cache is None:
                z_basic_app = torch.cat([z_basic, index_tokens], dim=1)
                index_logits = scoring_model.score_parallel_indices(z_basic_app, cfg.num_index_tokens)
            else:
                index_logits = scoring_model.score_parallel_indices_cached(index_tokens, kv_cache)
        probs = F.softmax(index_logits.float(), dim=-1)
        decode = scoring_model.unused_index_mask(z_basic)
        p = probs.gather(dim=2, index=z_result.unsqueeze(-1)).squeeze(-1)

        special_value = torch.zeros_like(decode, dtype=torch.bool)
        regular_decode = decode & ~special_value
        sample_decode = torch.where(regular_decode.any(dim=1, keepdim=True), regular_decode, decode)
        scores = p.float().masked_fill(~sample_decode, -float('inf'))
        if online_regen_method == 'gaussian':
            noisy_scores = scores + torch.randn_like(scores) * online_regen_noise_std
            max_idx = noisy_scores.argmax(dim=1)
        else:
            sample_probs = F.softmax(scores / online_regen_temperature, dim=1)
            max_idx = torch.multinomial(sample_probs, num_samples=1).squeeze(1)

        max_token = z_result.gather(dim=1, index=max_idx.unsqueeze(-1))
        append_tokens = torch.cat([max_idx.unsqueeze(-1) + cfg.index_token_start, max_token], dim=1)
        z_basic = torch.cat([z_basic, append_tokens], dim=1)
        if kv_cache is not None:
            with ctx:
                kv_cache = scoring_model.append_to_kv_cache(kv_cache, append_tokens)

    return z_basic.detach().cpu().long()


def _atomic_write_float32_file(path, array):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        np.asarray(array, dtype=np.float32).tofile(str(tmp_path))
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _soft_index_mode_tag():
    temp = f"{soft_index_temperature:g}".replace("-", "m").replace(".", "p")
    mode = soft_index_score_mode.replace("-", "")
    return f"{mode}_t{temp}"


def _soft_index_file_for_train_path(train_path):
    train_path = Path(train_path)
    return train_path.with_name(f"{train_path.stem}.softidx_{_soft_index_mode_tag()}.f32")


def _soft_index_expected_bytes(cfg):
    raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')
    num_blocks = len(raw_base) // cfg.base_seq_len
    del raw_base
    return num_blocks * cfg.response_size * cfg.response_size * np.dtype(np.float32).itemsize


def _soft_index_file_ready(path, cfg):
    path = Path(path)
    try:
        if not path.exists():
            return False
        size = path.stat().st_size
        if regen_max_blocks > 0:
            row_bytes = cfg.response_size * cfg.response_size * np.dtype(np.float32).itemsize
            return size > 0 and size % row_bytes == 0
        return size == _soft_index_expected_bytes(cfg)
    except OSError:
        return False


def _map_soft_index_file(path, cfg):
    if index_loss_mode != 'soft':
        return None
    path = Path(path)
    if not _soft_index_file_ready(path, cfg):
        return None
    if regen_max_blocks > 0:
        # Subsampled regen writes only regen_max_blocks rows of soft-index targets;
        # derive num_blocks from the actual soft-index file size, NOT the full base
        # set, or the mmap shape overshoots the file ("mmap length > file size").
        row_bytes = cfg.response_size * cfg.response_size * np.dtype(np.float32).itemsize
        num_blocks = path.stat().st_size // row_bytes
    else:
        raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')
        num_blocks = len(raw_base) // cfg.base_seq_len
        del raw_base
    return np.memmap(
        str(path),
        dtype=np.float32,
        mode='r',
        shape=(num_blocks, cfg.response_size, cfg.response_size),
    )


def get_batch(split):
    if split == 'train':
        batch_size = train_batch_size
    elif split == 'val':
        data = val_data
        batch_size = val_batch_size
    elif split == 'test':
        data = test_loss_data
        batch_size = val_batch_size
    else:
        raise ValueError(f"Unknown split: {split}")

    prompt_len = quiz_size
    target_len = response_size
    data_size = prompt_len + 2 * target_len

    use_online_regen = (split == 'train' and online_regen_sample and iter_num >= round_interval and online_regen_model is not None)
    use_mix = (split == 'train' and iter_num >= round_interval and not use_online_regen
               and (mix_ratios[1] > 0 or mix_ratios[2] > 0))
    sources = None
    soft_index_targets = None
    if use_online_regen:
        z = _sample_online_regen_batch(batch_size)
    elif use_mix:
        has_prev = bool(data_prev_list)
        if has_prev:
            effective_ratios = mix_ratios
        else:
            effective_ratios = [mix_ratios[0] + mix_ratios[1], 0.0, mix_ratios[2]]
        sources = np.random.choice(3, size=batch_size, p=effective_ratios)
        z = torch.empty(batch_size, data_size, dtype=torch.long)
        if index_loss_mode == 'soft' and not (loss_mode == 'value_only' and iter_num < round_interval):
            soft_index_targets = torch.empty(batch_size, response_size, response_size, dtype=torch.float32)
        # source 0: main (current round regen, model's confident-first order)
        mask0 = sources == 0
        n0 = int(mask0.sum())
        if n0 > 0:
            batch_rows = torch.from_numpy(np.where(mask0)[0])
            if soft_index_targets is not None:
                z0, rows0 = _sample_from_source(data_main, n0, data_size, return_row_ids=True)
                z[batch_rows] = z0
                soft0 = _sample_soft_index_targets(data_main_soft, rows0)
                if soft0 is None:
                    soft_index_targets[batch_rows] = 0.0
                else:
                    soft_index_targets[batch_rows] = soft0
            else:
                z[batch_rows] = _sample_from_source(data_main, n0, data_size)
        # source 1: prev — UNIFORM across all prior-round regen files
        mask1 = sources == 1
        n1 = int(mask1.sum())
        if n1 > 0 and has_prev:
            batch_rows = torch.from_numpy(np.where(mask1)[0])
            if soft_index_targets is not None:
                z1, soft1 = _sample_uniform_from_sources(
                    data_prev_list, n1, data_size, soft_sources_list=data_prev_soft_list
                )
                z[batch_rows] = z1
                if soft1 is None:
                    soft_index_targets[batch_rows] = 0.0
                else:
                    soft_index_targets[batch_rows] = soft1
            else:
                z[batch_rows] = _sample_uniform_from_sources(data_prev_list, n1, data_size)
        # source 2: canonical (L2R on disk, optionally reordered below)
        mask2 = sources == 2
        n2 = int(mask2.sum())
        if n2 > 0:
            canonical_batch_rows = torch.from_numpy(np.where(mask2)[0])
            z[canonical_batch_rows] = _sample_from_source(
                data_canonical_traj, n2, data_size
            )
            if soft_index_targets is not None:
                soft_index_targets[canonical_batch_rows] = 0.0
    else:
        if split == 'train':
            data = data_main
        if split == 'train' and index_loss_mode == 'soft' and data is data_main:
            z, rows = _sample_from_source(data, batch_size, data_size, return_row_ids=True)
            soft_index_targets = _sample_soft_index_targets(data_main_soft, rows)
        else:
            z = _sample_from_source(data, batch_size, data_size)

    if split == 'train' and shuffle_order:
        # FOX multi-round design:
        #   Round 1 (iter < round_interval): shuffle ALL rows -> model learns
        #     order-invariance -> HIGH PI -> used to GENERATE the round-2 regen data.
        #   Round 2+: the main/prev regen rows are kept in their confident-first
        #     (easy-to-hard) order (NOT shuffled) -> this self-distillation BOOSTS AR.
        #     Only the canonical (L2R, source==2) rows from the MIX are shuffled, which
        #     SLIGHTLY maintains PI. So with mix_ratios=0.7,0.2,0.1 round-2 trades a bit
        #     of PI for a large AR gain (intended). With mix 1.0,0,0 (pure regen) there
        #     is nothing to shuffle -> PI fully collapses, so MIX is required to keep PI.
        if sources is not None:
            shuffle_idx = np.where(sources == 2)[0]
        elif iter_num < round_interval:
            shuffle_idx = np.arange(batch_size)
        else:
            shuffle_idx = np.array([], dtype=np.intp)
        n = len(shuffle_idx)
        if n > 0:
            rows = torch.from_numpy(shuffle_idx)
            qi = z[rows, :quiz_size]
            pairs = z[rows, quiz_size:].reshape(n, response_size, 2)
            # pure random (PAD-last is a decode/regen concept, never a train-shuffle bias)
            perm = torch.rand(n, response_size, device=z.device).argsort(dim=1)
            pairs = pairs[torch.arange(n).unsqueeze(1), perm]
            z[rows] = torch.cat([qi, pairs.reshape(n, -1)], dim=1)
            if soft_index_targets is not None:
                perm_expanded = perm.unsqueeze(-1).expand(-1, -1, response_size)
                soft_index_targets[rows] = torch.gather(
                    soft_index_targets[rows], 1, perm_expanded
                )

    x = z[:,:-1].clone()
    y = z[:,1:].clone()
    y[:, :prompt_len - 1] = -100

    # When index_loss_mode='soft', do NOT mask index targets in round 1: soft loss
    # is only applied at iter >= round_interval, so round 1 needs the index targets
    # for hard CE supervision. Without them, AR generation never learns to predict
    # indices and outputs malformed sequences.
    if (
        loss_mode == 'value_only'
        and iter_num < round_interval
        and index_loss_mode != 'soft'
    ):
        idx_target_pos = torch.arange(target_len) * 2 + (prompt_len - 1)
        y[:, idx_target_pos] = -100

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        if soft_index_targets is not None:
            soft_index_targets = soft_index_targets.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        if soft_index_targets is not None:
            soft_index_targets = soft_index_targets.to(device)
    return x, y, soft_index_targets


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# logger (only master writes to the log file)
import logging as _logging
if master_process:
    logger = get_logger(os.path.join(out_dir, "train.log"))
else:
    logger = _logging.getLogger(f"serfox_null_{os.getpid()}")
    logger.handlers.clear()
    logger.addHandler(_logging.NullHandler())






# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    quiz_size=quiz_size,
    response_size=response_size,
    value_vocab_size=value_vocab_size,
) # start with model_args from command line

checkpoint = None


def checkpoint_iter_from_path(path):
    try:
        return int(Path(path).name.split("_ckpt.pt")[0])
    except ValueError:
        return -1


def resolve_resume_checkpoint_candidates(resume_ckpt):
    if resume_ckpt is not None:
        candidates = [
            Path(resume_ckpt),
            REPO_ROOT / resume_ckpt,
            Path(out_dir) / resume_ckpt,
        ]
        existing = []
        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate.exists():
                existing.append(candidate.resolve())
        if existing:
            return existing, True
        raise FileNotFoundError(f"Could not resolve resume checkpoint from candidates: {candidates}")

    ckpts = sorted(
        [path for path in Path(out_dir).glob("*_ckpt.pt") if checkpoint_iter_from_path(path) >= 0],
        key=checkpoint_iter_from_path,
        reverse=True,
    )
    if not ckpts:
        raise FileNotFoundError(f"No *_ckpt.pt checkpoints found in {out_dir}; cannot resume.")
    return [path.resolve() for path in ckpts], False



if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

if init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_candidates, explicit_resume_ckpt = resolve_resume_checkpoint_candidates(args.resume_ckpt)
    last_load_error = None
    for ckpt_path in ckpt_candidates:
        try:
            print(f"Loading checkpoint {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            break
        except Exception as exc:
            last_load_error = exc
            if explicit_resume_ckpt:
                raise
            if master_process:
                print(f"WARNING: Failed to load checkpoint {ckpt_path}: {exc}; trying an older checkpoint.")
    else:
        raise RuntimeError(f"Could not load any checkpoint from {out_dir}") from last_load_error
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        'n_layer', 'n_head', 'n_embd',
        'block_size', 'bias', 'vocab_size',
        'quiz_size', 'response_size', 'value_vocab_size',
    ]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = int(checkpoint.get('iter_num', checkpoint_iter_from_path(ckpt_path)))
    best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
_skip_resume_eval = (init_from == 'resume' and iter_num > 0)
_resume_needs_boundary_regen = False
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and checkpoint is not None and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)


def run_forward_ar(model, idx, targets=None):
    if isinstance(model, DDP) or compile:
        return model(idx, targets)
    return model.forward_ar(idx, targets)


def get_scoring_model(model):
    base_model = model.module if isinstance(model, DDP) else model
    return getattr(base_model, "_orig_mod", base_model)


def _soft_index_score_mode_internal():
    if soft_index_score_mode == "argmax":
        return "top1_prob"
    if soft_index_score_mode == "entropy":
        return "neg_entropy"
    if soft_index_score_mode == "p1-p2":
        return "margin"
    return soft_index_score_mode


def _index_target_positions(device):
    return torch.arange(response_size, device=device, dtype=torch.long) * 2 + (quiz_size - 1)


def _gt_values_by_response_position(idx, targets, cfg):
    last_token = targets[:, -1:].clone()
    if (last_token == -100).any():
        raise ValueError("Cannot reconstruct serialized sequence: final target token is ignored")
    z = torch.cat([idx, last_token], dim=1)
    pairs = z[:, cfg.quiz_size:].reshape(idx.size(0), cfg.response_size, 2)
    positions = pairs[:, :, 0] - cfg.index_token_start
    values = pairs[:, :, 1]
    valid = (positions >= 0) & (positions < cfg.response_size)
    if not valid.all():
        raise ValueError("Serialized batch contains out-of-range index tokens")
    gt_values = torch.empty_like(values)
    gt_values.scatter_(1, positions.long(), values)
    return pairs, gt_values


@torch.no_grad()
def build_soft_index_targets(model, idx, targets):
    scoring_model = get_scoring_model(model)
    cfg = scoring_model.config
    index_positions = _index_target_positions(idx.device)
    valid_index_targets = targets[:, index_positions] != -100
    if not valid_index_targets.any():
        return torch.zeros(
            idx.size(0),
            cfg.response_size,
            cfg.response_size,
            dtype=torch.float32,
            device=idx.device,
        )

    pairs, gt_values = _gt_values_by_response_position(idx, targets, cfg)
    candidate_index_ids = (
        torch.arange(cfg.response_size, device=idx.device, dtype=torch.long)
        + cfg.index_token_start
    )
    candidate_suffix = candidate_index_ids.unsqueeze(0).expand(idx.size(0), -1)

    special_value = torch.zeros_like(gt_values, dtype=torch.bool)
    if pad_id is not None:
        special_value = special_value | (gt_values == pad_id)
    if eos_id is not None:
        special_value = special_value | (gt_values == eos_id)
    regular_value = ~special_value

    soft_targets = []
    used = torch.zeros(idx.size(0), cfg.response_size, dtype=torch.bool, device=idx.device)
    internal_score_mode = _soft_index_score_mode_internal()

    was_training = scoring_model.training
    scoring_model.eval()
    try:
        for step, index_pos in enumerate(index_positions.tolist()):
            unresolved = ~used
            regular_unresolved = unresolved & regular_value
            has_regular = regular_unresolved.any(dim=1, keepdim=True)
            eligible = torch.where(has_regular, regular_unresolved, unresolved)

            if internal_score_mode == "uniform":
                # uniform target = 1/k over eligible (remaining) positions; it does
                # NOT use value_logits, so skip the per-step score_parallel_indices
                # forward (huge speedup: e.g. sudoku avoids 81 model forwards/batch).
                dist = eligible.float()
                dist = dist / dist.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                prefix = idx[:, : index_pos + 1]
                scoring_input = torch.cat([prefix, candidate_suffix], dim=1)
                with ctx:
                    value_logits = scoring_model.score_parallel_indices(scoring_input, cfg.response_size)
                dist = soft_index_distribution_from_value_logits(
                    value_logits,
                    eligible,
                    score_mode=internal_score_mode,
                    temperature=soft_index_temperature,
                    gt_values=gt_values,
                )
            soft_targets.append(dist.detach())

            current_positions = pairs[:, step, 0] - cfg.index_token_start
            used.scatter_(1, current_positions.long().unsqueeze(1), True)
    finally:
        if was_training:
            scoring_model.train()

    return torch.stack(soft_targets, dim=1)


def run_forward_train(model, idx, targets, soft_index_targets=None):
    # round-1 (iter < round_interval) ALWAYS uses hard CE for the index: soft targets are
    # only generated at round boundaries (round-2+), matching the note in get_batch. Without
    # the `iter_num < round_interval` guard, soft + a non-uniform score (e.g. logit_margin)
    # falls into per-step build_soft_index_targets (response_size forwards/step) and the
    # training loop crawls to a near-hang. (uniform has a fast path so it never hit this.)
    if index_loss_mode != "soft" or targets is None or iter_num < round_interval:
        return run_forward_ar(model, idx, targets)

    logits, _ = run_forward_ar(model, idx, None)
    if soft_index_targets is None:
        soft_index_targets = build_soft_index_targets(model, idx, targets)
    index_positions = _index_target_positions(targets.device)
    candidate_index_ids = (
        torch.arange(response_size, device=targets.device, dtype=torch.long)
        + value_vocab_size
    )
    loss = mixed_soft_index_ar_loss(
        logits,
        targets,
        index_positions,
        candidate_index_ids,
        soft_index_targets,
    )
    return logits, loss


def resolve_optional_test_path(test_file):
    if test_file is None:
        return None

    candidates = [
        Path(test_file),
        REPO_ROOT / test_file,
        REPO_ROOT / "data" / test_file,
        data_dir / test_file,
    ]
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve test_file={test_file!r} from candidates: {candidates}")


def encode_json_eval_file(json_path, meta, cfg):
    stoi = meta["stoi"]
    input_key = meta.get("input_key", "input")
    output_key = meta.get("output_key", "output")
    pad_id = stoi[meta.get("pad_token", "<PAD>")]
    sep_id = stoi[meta.get("sep_token", "<SEP>")]
    eos_id = stoi[meta.get("eos_token", "<EOS>")]
    max_quiz_len = cfg.quiz_size - 1
    max_response_len = cfg.response_size - 1
    rows = []

    def encode_text(text, sample_idx, field_name):
        token_ids = []
        for ch in str(text):
            if ch not in stoi:
                raise ValueError(
                    f"Unknown character {ch!r} in {json_path} sample {sample_idx} field {field_name!r}; "
                    "the eval file must use the same vocabulary as meta.pkl."
                )
            token_ids.append(stoi[ch])
        return token_ids

    with open(json_path, "r", encoding="utf-8") as f:
        if json_path.suffix == ".json":
            loaded = json.load(f)
            samples = loaded if isinstance(loaded, list) else loaded.get("data", [loaded])
        else:
            samples = [json.loads(line) for line in f if line.strip()]

    for sample_idx, sample in enumerate(samples):
        quiz_tokens = encode_text(sample.get(input_key, ""), sample_idx, input_key)
        response_tokens = encode_text(sample.get(output_key, ""), sample_idx, output_key)

        if len(quiz_tokens) > max_quiz_len:
            raise ValueError(f"Quiz too long in {json_path} sample {sample_idx}: {len(quiz_tokens)} > {max_quiz_len}")
        if len(response_tokens) > max_response_len:
            raise ValueError(
                f"Response too long in {json_path} sample {sample_idx}: {len(response_tokens)} > {max_response_len}"
            )

        quiz_padded = quiz_tokens + [pad_id] * (max_quiz_len - len(quiz_tokens)) + [sep_id]
        response_padded = response_tokens + [pad_id] * (max_response_len - len(response_tokens)) + [eos_id]
        rows.append(quiz_padded + response_padded)

    if not rows:
        raise ValueError(f"No samples found in {json_path}")

    return np.array(rows, dtype=np.uint16).reshape(-1)


def load_test_array(test_path, meta, cfg):
    if test_path.suffix in {".jsonl", ".json"}:
        test_arr = encode_json_eval_file(test_path, meta, cfg)
    else:
        test_arr = np.memmap(test_path, dtype=np.uint16, mode="r")

    if len(test_arr) % cfg.base_seq_len != 0:
        raise ValueError(
            f"Test file length {len(test_arr)} is not divisible by base_seq_len {cfg.base_seq_len}: {test_path}"
        )
    return test_arr


def serialize_base_array_for_loss(base_arr, cfg):
    rows = np.asarray(base_arr, dtype=np.uint16).reshape(-1, cfg.base_seq_len)
    prompts = rows[:, :cfg.quiz_size]
    responses = rows[:, cfg.quiz_size : cfg.quiz_size + cfg.response_size]

    serialized = np.empty((rows.shape[0], cfg.train_seq_len), dtype=np.uint16)
    serialized[:, :cfg.quiz_size] = prompts

    index_tokens = (
        np.arange(cfg.num_index_tokens, dtype=np.uint16)
        + np.uint16(cfg.index_token_start)
    )
    serialized[:, cfg.quiz_size::2] = index_tokens[None, :]
    serialized[:, cfg.quiz_size + 1::2] = responses
    return serialized.reshape(-1)


def deserialize_indexed_response(serialized_tokens, cfg):
    expected_len = cfg.quiz_size + 2 * cfg.response_size
    if len(serialized_tokens) < expected_len:
        raise ValueError(f"Generated sequence too short: {len(serialized_tokens)} < {expected_len}")

    response = [-1] * cfg.response_size
    seen = set()
    for i in range(cfg.response_size):
        idx_tok = int(serialized_tokens[cfg.quiz_size + 2 * i])
        val_tok = int(serialized_tokens[cfg.quiz_size + 2 * i + 1])
        pos = idx_tok - cfg.index_token_start

        if not (0 <= pos < cfg.response_size):
            raise ValueError(f"Invalid index token {idx_tok} at pair {i}")
        if pos in seen:
            raise ValueError(f"Duplicate index token {idx_tok} at pair {i}")
        if not (0 <= val_tok < cfg.index_token_start):
            raise ValueError(f"Invalid value token {val_tok} at pair {i}")

        seen.add(pos)
        response[pos] = val_tok

    if any(v == -1 for v in response):
        raise ValueError("Missing index positions in serialized output")
    return response


@torch.inference_mode()
def evaluate_test_split(test_arr, mode="serialized_ar"):
    was_training = model.training
    model.eval()
    eval_model = get_scoring_model(model)
    cfg = eval_model.config

    correct = 0
    total = 0
    malformed = 0

    bsz = max(1, eval_decode_batch_size)
    try:
        num_samples = len(test_arr) // cfg.base_seq_len
        # Strided slice: each DDP rank handles a disjoint subset of samples.
        my_indices = list(range(ddp_rank, num_samples, ddp_world_size))
        # Batched decode: each test sample is decoded independently (no cross-sample
        # attention), so decoding B samples at once is mathematically identical to the
        # old per-sample loop but ~B times faster on the GPU. Deserialize/compare stays
        # per-row (cheap CPU). Set --eval_decode_batch_size 1 to restore the old path.
        for bstart in range(0, len(my_indices), bsz):
            chunk = my_indices[bstart:bstart + bsz]
            rows = [test_arr[i * cfg.base_seq_len:(i + 1) * cfg.base_seq_len].astype(np.int64) for i in chunk]
            prompts = np.stack([r[:cfg.quiz_size] for r in rows])
            x = torch.from_numpy(prompts).to(device=device, dtype=torch.long)
            with torch.no_grad():
                if mode == "parallel_index":
                    y = eval_model.generate_parallel_index(
                        x, max_new_tokens=cfg.response_size, temperature=0.01, top_k=None)
                else:
                    y = eval_model.generate_serialized_ar(
                        x, max_new_tokens=cfg.response_size * 2, temperature=0.01, top_k=None)
            y = y.tolist()
            for j, row in enumerate(rows):
                target_final = row.tolist()
                prompt = target_final[:cfg.quiz_size]
                try:
                    response = deserialize_indexed_response(y[j], cfg)
                    final_pred = prompt + response
                    is_correct = list(final_pred) == list(target_final)
                except ValueError:
                    malformed += 1
                    is_correct = False
                correct += int(is_correct)
                total += 1
    finally:
        if was_training:
            model.train()

    if ddp:
        stats = torch.tensor([correct, total, malformed], dtype=torch.long, device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        correct, total, malformed = stats.tolist()

    accuracy = correct / total if total else 0.0
    malformed_rate = malformed / total if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "malformed": malformed,
        "malformed_rate": malformed_rate,
    }


def update_results_table(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            table = json.load(f)
    else:
        table = {"results": []}

    key_fields = ["backbone", "config", "seed", "ckpt_iter", "test_file"]
    record_key = {field: record[field] for field in key_fields}
    results = table.setdefault("results", [])

    for idx, existing in enumerate(results):
        if all(existing.get(field) == value for field, value in record_key.items()):
            results[idx] = record
            break
    else:
        results.append(record)

    results.sort(key=lambda r: (r.get("backbone", ""), r.get("config", ""), r.get("mode", ""), r.get("ckpt_iter", 0)))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)


def record_round_test_results(iter_num, ckpt_path, test_path, test_arr, test_loss=None):
    # All ranks participate (evaluate_test_split slices across ranks and
    # all_reduces internally). Only master writes file output / logs.
    if test_arr is None:
        return

    local_results_path = Path(out_dir) / "serfox_test_results.json"

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        metrics_ar = evaluate_test_split(test_arr, mode="serialized_ar")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        metrics_pi = evaluate_test_split(test_arr, mode="parallel_index")

        if not master_process:
            return

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "backbone": f"Ser-FOX-V4-{loss_tag}-{shuf_tag}",
            "config": config_name,
            "seed": seed,
            "ckpt_iter": iter_num,
            "mode": "serialized_ar+parallel_index",
            "temperature": 0.01,
            "test_file": str(test_path),
            "checkpoint": str(ckpt_path),
            "out_dir": out_dir,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "test_loss": test_loss,
            "accuracy": metrics_ar["accuracy"],
            "correct": metrics_ar["correct"],
            "total": metrics_ar["total"],
            "malformed": metrics_ar["malformed"],
            "accuracy_pi": metrics_pi["accuracy"],
            "correct_pi": metrics_pi["correct"],
            "total_pi": metrics_pi["total"],
            "malformed_pi": metrics_pi["malformed"],
        }
        update_results_table(local_results_path, record)
        print(
            f"test at iter {iter_num}: "
            f"AR {metrics_ar['accuracy']:.6f} ({metrics_ar['correct']}/{metrics_ar['total']}), malformed {metrics_ar['malformed']} | "
            f"PI {metrics_pi['accuracy']:.6f} ({metrics_pi['correct']}/{metrics_pi['total']}), malformed {metrics_pi['malformed']}"
        )
        logger.info(
            f"test at iter {iter_num}: "
            f"AR {metrics_ar['accuracy']:.6f} ({metrics_ar['correct']}/{metrics_ar['total']}), malformed {metrics_ar['malformed']} | "
            f"PI {metrics_pi['accuracy']:.6f} ({metrics_pi['correct']}/{metrics_pi['total']}), malformed {metrics_pi['malformed']}"
        )
        append_loss_metrics(
            iter_num, phase="test_acc_AR", lr=None,
            test_accuracy=metrics_ar['accuracy'],
            test_malformed=metrics_ar['malformed'],
        )
        append_loss_metrics(
            iter_num, phase="test_acc_PI", lr=None,
            test_accuracy=metrics_pi['accuracy'],
            test_malformed=metrics_pi['malformed'],
        )
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def _select_best_round_ckpt_iter(round_start, round_end):
    """Highest-AR ckpt_iter in (round_start, round_end] from serfox_test_results.json.
    Overfit guard: stop scanning after 2 consecutive AR drops and keep the peak."""
    results_path = Path(out_dir) / "serfox_test_results.json"
    if not results_path.exists():
        return None
    try:
        data = json.load(open(results_path))
        recs = data.get("results", []) if isinstance(data, dict) else data
    except Exception:
        return None
    rnd = sorted(
        [r for r in recs if round_start < r.get("ckpt_iter", -1) <= round_end],
        key=lambda r: r["ckpt_iter"],
    )
    if not rnd:
        return None
    # best = highest-AR ckpt, ties broken toward the LATEST iter (>=). So a flat or
    # still-rising round warm-starts from the last ckpt (== warm-from-last, no harm),
    # while a true rise-then-fall (overfit) round warm-starts from the peak. This is
    # the robust realization of "stop at 2 consecutive drops -> keep the best".
    best_it, best_acc = rnd[-1]["ckpt_iter"], -1.0
    for r in rnd:
        # soft round-2+ has malformed serialized-AR (accuracy=0); fall back to PI
        # (accuracy_pi) so the overfit guard still finds the round's true peak ckpt.
        acc = max(r.get("accuracy", 0.0) or 0.0, r.get("accuracy_pi", 0.0) or 0.0)
        if acc >= best_acc:
            best_acc, best_it = acc, r["ckpt_iter"]
    return best_it


def _warm_start_from_best_round_ckpt(round_end_iter):
    """Reload the best-AR ckpt of the round into raw_model before regen+continue."""
    best_it = _select_best_round_ckpt_iter(round_end_iter - round_interval, round_end_iter)
    if best_it is None or best_it == round_end_iter:
        if master_process:
            print(f"[warm_from_best] round boundary {round_end_iter}: best == last (no reload)")
        return
    best_path = Path(out_dir) / f"{best_it}_ckpt.pt"
    if not best_path.exists():
        if master_process:
            print(f"[warm_from_best] best ckpt {best_it} file missing; keeping last")
        return
    try:
        ck = torch.load(str(best_path), map_location=device)
        sd = ck["model"]
        try:
            raw_model.load_state_dict(sd)  # ckpt saved from raw_model.state_dict(): direct match
        except Exception:
            # toggle the torch.compile "_orig_mod." prefix to match raw_model's keys
            want = any(k.startswith("_orig_mod.") for k in raw_model.state_dict().keys())
            fixed = {}
            for k, v in sd.items():
                if want and not k.startswith("_orig_mod."):
                    fixed["_orig_mod." + k] = v
                elif (not want) and k.startswith("_orig_mod."):
                    fixed[k[len("_orig_mod."):]] = v
                else:
                    fixed[k] = v
            raw_model.load_state_dict(fixed)
        if master_process:
            print(f"[warm_from_best] round boundary {round_end_iter}: reloaded best round ckpt {best_it} (AR-selected) before regen")
            logger.info(f"[warm_from_best] reloaded best round ckpt {best_it} at boundary {round_end_iter}")
    except Exception as exc:
        if master_process:
            print(f"[warm_from_best] failed to reload best ckpt {best_it}: {exc}; keeping last")


def _load_regen_override_model(ckpt_path_str):
    """Load a separate model from a checkpoint for trajectory regeneration."""
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.is_absolute():
        for candidate in (REPO_ROOT / ckpt_path, Path(out_dir) / ckpt_path):
            if candidate.exists():
                ckpt_path = candidate
                break
    ckpt = torch.load(str(ckpt_path), map_location=device)
    ckpt_model_args = ckpt['model_args']
    regen_gptconf = GPTConfig(**ckpt_model_args)
    regen_m = GPT(regen_gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    regen_m.load_state_dict(state_dict)
    regen_m.to(device)
    regen_m.eval()
    if master_process:
        print(f"Loaded regen override model from {ckpt_path}")
    return regen_m


@torch.inference_mode()
def regenerate_trajectory_data(start_block=None, end_block=None, override_model=None, return_soft_index=False, block_indices=None):
    m = override_model if override_model is not None else model
    was_training = m.training
    m.eval()

    raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')

    try:
        scoring_model = get_scoring_model(m)
        cfg = scoring_model.config
        num_blocks = len(raw_base) // cfg.base_seq_len
        if start_block is None:
            start_block = 0
        if end_block is None:
            end_block = num_blocks
        if block_indices is None:
            block_indices = np.arange(start_block, end_block, dtype=np.int64)
        base_2d = raw_base[: num_blocks * cfg.base_seq_len].reshape(num_blocks, cfg.base_seq_len)
        blocks_per_step = regen_blocks_per_step
        out_chunks = []
        soft_chunks = []
        internal_score_mode = _soft_index_score_mode_internal()

        for start in range(0, len(block_indices), blocks_per_step):
            chunk_idx = block_indices[start : start + blocks_per_step]

            z = torch.from_numpy(base_2d[chunk_idx].astype(np.int64))

            z = z.to(device)
            a, _ = z.shape

            z_basic = z[:, :cfg.quiz_size]
            z_result = z[:, cfg.quiz_size : cfg.quiz_size + cfg.response_size]

            index_tokens = torch.arange(0, cfg.num_index_tokens, dtype=torch.long, device=device).unsqueeze(0)
            index_tokens = (index_tokens + cfg.index_token_start).expand(a, -1)
            kv_cache = None
            if use_regen_kv_cache:
                with ctx:
                    kv_cache = scoring_model.build_prefix_kv_cache(z_basic)

            soft_steps = []
            for _ in range(cfg.response_size):
                with ctx:
                    if kv_cache is None:
                        z_basic_app = torch.cat([z_basic, index_tokens], dim=1)
                        index_logits = scoring_model.score_parallel_indices(
                            z_basic_app, cfg.num_index_tokens
                        )
                    else:
                        index_logits = scoring_model.score_parallel_indices_cached(index_tokens, kv_cache)

                decode = scoring_model.unused_index_mask(z_basic)

                # Single distribution P = softmax(logit_margin) over the remaining
                # positions. The SAME P is used both as the soft index LABEL and as the
                # sampling distribution for the regenerated trajectory, so the generated
                # order is self-consistent with the supervision (true easy-to-hard).
                # (Replaces the old split design: pick-by-p_gt/confidence + label-by-logit_margin.)
                soft_dist = soft_index_distribution_from_value_logits(
                    index_logits,
                    decode,
                    score_mode=internal_score_mode,
                    temperature=soft_index_temperature,
                    gt_values=z_result,
                )
                if return_soft_index:
                    soft_steps.append(soft_dist.detach().cpu().float())

                # Pick the next position FROM the same P: argmax(P) when deterministic,
                # else sample from P (regen_position_temperature>0 -> diverse trajectories).
                if regen_position_temperature is None or regen_position_temperature <= 0:
                    max_idx = soft_dist.masked_fill(~decode.bool(), float("-inf")).argmax(dim=1)
                else:
                    max_idx = torch.multinomial(soft_dist, num_samples=1).squeeze(-1)
                # value is bound to the chosen position's ground-truth token.
                max_token = z_result.gather(dim=1, index=max_idx.unsqueeze(-1))
                append_tokens = torch.cat(
                    [
                        max_idx.unsqueeze(-1) + cfg.index_token_start,
                        max_token,
                    ],
                    dim=1,
                )

                z_basic = torch.cat([z_basic, append_tokens], dim=1)
                if kv_cache is not None:
                    with ctx:
                        kv_cache = scoring_model.append_to_kv_cache(kv_cache, append_tokens)

            out_chunks.append(z_basic.detach().cpu().reshape(-1).numpy())
            if return_soft_index:
                soft_chunks.append(torch.stack(soft_steps, dim=1).numpy())

        tokens = np.concatenate(out_chunks, axis=0) if out_chunks else np.array([], dtype=np.uint16)
        if not return_soft_index:
            return tokens
        soft_targets = (
            np.concatenate(soft_chunks, axis=0).astype(np.float32, copy=False)
            if soft_chunks
            else np.empty((0, cfg.response_size, cfg.response_size), dtype=np.float32)
        )
        return tokens, soft_targets
    finally:
        if was_training:
            m.train()


def _serialized_file_expected_bytes(cfg):
    raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')
    num_blocks = len(raw_base) // cfg.base_seq_len
    del raw_base
    return num_blocks * cfg.train_seq_len * np.dtype(np.uint16).itemsize


def _serialized_file_ready(path, cfg):
    path = Path(path)
    try:
        if not path.exists():
            return False
        size = path.stat().st_size
        if regen_max_blocks > 0:
            # subsampled regen writes fewer blocks; any nonzero whole number of
            # serialized rows is valid (atomic tmp+rename rules out partials)
            row_bytes = cfg.train_seq_len * np.dtype(np.uint16).itemsize
            return size > 0 and size % row_bytes == 0
        return size == _serialized_file_expected_bytes(cfg)
    except OSError:
        return False


def _atomic_write_uint16_file(path, array):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        np.asarray(array, dtype=np.uint16).tofile(str(tmp_path))
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _ensure_l2r_trajectory_file(path, cfg, description):
    """Create a serialized L2R trajectory on rank 0, then sync readers."""
    path = Path(path)
    if master_process and not _serialized_file_ready(path, cfg):
        print(f"{description} not found or incomplete; generating {path}")
        logger.info(f"{description} not found or incomplete; generating {path}")
        raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')
        l2r_data = serialize_base_array_for_loss(raw_base, cfg)
        _atomic_write_uint16_file(path, l2r_data)
        print(f"Generated {path.name} ({len(l2r_data)} tokens)")
        logger.info(f"Generated {path.name} ({len(l2r_data)} tokens)")
        del l2r_data, raw_base
    if ddp:
        torch.distributed.barrier()
    if not _serialized_file_ready(path, cfg):
        raise FileNotFoundError(f"{description} is missing or incomplete after generation: {path}")


def regenerate_trajectory_data_parallel(out_path, override_model=None, soft_index_out_path=None):
    """Regenerate trajectory data using all DDP ranks in parallel, then write."""
    m = override_model if override_model is not None else model
    scoring_model = get_scoring_model(m)
    cfg = scoring_model.config
    raw_base = np.memmap(base_path, dtype=np.uint16, mode='r')
    num_blocks = len(raw_base) // cfg.base_seq_len

    if ddp:
        rank, world = ddp_rank, ddp_world_size
    else:
        rank, world = 0, 1

    if regen_max_blocks > 0 and regen_max_blocks < num_blocks:
        # per-round random subset, seeded by iter so every rank draws the same
        # selection (and reruns of the same boundary are reproducible)
        sel = np.random.default_rng(1234 + int(iter_num)).choice(
            num_blocks, size=regen_max_blocks, replace=False
        )
        sel.sort()
    else:
        sel = np.arange(num_blocks, dtype=np.int64)

    per_rank = (len(sel) + world - 1) // world
    my_indices = sel[rank * per_rank : (rank + 1) * per_rank]

    if master_process:
        print(f"Regenerating trajectory data across {world} GPU(s) "
              f"({len(sel)}/{num_blocks} blocks, ~{per_rank} per rank)...", flush=True)

    return_soft_index = soft_index_out_path is not None
    shard = regenerate_trajectory_data(
        override_model=override_model,
        return_soft_index=return_soft_index,
        block_indices=my_indices,
    )
    if return_soft_index:
        shard, soft_shard = shard
    else:
        soft_shard = None

    if world > 1:
        shard_path = out_path + f".shard{rank}"
        _atomic_write_uint16_file(shard_path, shard)
        if return_soft_index:
            soft_shard_path = str(soft_index_out_path) + f".shard{rank}"
            _atomic_write_float32_file(soft_shard_path, soft_shard)
        torch.distributed.barrier()
        if master_process:
            out_path_obj = Path(out_path)
            tmp_path = out_path_obj.with_name(f"{out_path_obj.name}.tmp.{os.getpid()}")
            soft_tmp_path = None
            try:
                with open(tmp_path, 'wb') as fout:
                    for r in range(world):
                        sp = out_path + f".shard{r}"
                        with open(sp, 'rb') as fin:
                            fout.write(fin.read())
                        os.remove(sp)
                os.replace(str(tmp_path), out_path)
                if return_soft_index:
                    soft_out_path_obj = Path(soft_index_out_path)
                    soft_tmp_path = soft_out_path_obj.with_name(
                        f"{soft_out_path_obj.name}.tmp.{os.getpid()}"
                    )
                    with open(soft_tmp_path, 'wb') as fout:
                        for r in range(world):
                            sp = str(soft_index_out_path) + f".shard{r}"
                            with open(sp, 'rb') as fin:
                                fout.write(fin.read())
                            os.remove(sp)
                    os.replace(str(soft_tmp_path), str(soft_index_out_path))
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                if soft_tmp_path is not None and soft_tmp_path.exists():
                    try:
                        soft_tmp_path.unlink()
                    except OSError:
                        pass
        torch.distributed.barrier()
    else:
        _atomic_write_uint16_file(out_path, shard)
        if return_soft_index:
            _atomic_write_float32_file(soft_index_out_path, soft_shard)
    if not _serialized_file_ready(out_path, cfg):
        raise FileNotFoundError(f"Regenerated trajectory is missing or incomplete after write: {out_path}")
    if return_soft_index and not _soft_index_file_ready(soft_index_out_path, cfg):
        raise FileNotFoundError(
            f"Regenerated soft-index targets are missing or incomplete after write: {soft_index_out_path}"
        )





def _rollout_test_loss(eval_model, quiz_arr, gt_resp_arr, cfg, rollout_batch=None):
    """Rollout test loss: model predicts both index and value autoregressively.

    Value CE loss is matched by the model's predicted index → ground truth position.
    """
    if rollout_batch is None:
        # the rollout is per-sample independent (same as the accuracy decode), so
        # reuse the decode batch size instead of the old hardcoded 64
        rollout_batch = max(1, eval_decode_batch_size)
    num_samples = quiz_arr.shape[0]
    if ddp:
        per_rank = (num_samples + ddp_world_size - 1) // ddp_world_size
        my_start = ddp_rank * per_rank
        my_end = min(my_start + per_rank, num_samples)
    else:
        my_start, my_end = 0, num_samples

    total_loss = 0.0
    total_tokens = 0

    for bs in range(my_start, my_end, rollout_batch):
        be = min(bs + rollout_batch, my_end)
        b = be - bs

        quiz = torch.from_numpy(quiz_arr[bs:be].astype(np.int64)).to(device)
        gt_resp = torch.from_numpy(gt_resp_arr[bs:be].astype(np.int64)).to(device)

        prefix = quiz.clone()
        seen = torch.zeros(b, cfg.response_size, dtype=torch.bool, device=device)

        for _step in range(cfg.response_size):
            with ctx:
                idx_logits = eval_model.forward_ar(prefix)[0][:, -1, :]
            pred_idx = idx_logits.argmax(dim=-1)
            prefix = torch.cat([prefix, pred_idx.unsqueeze(1)], dim=1)

            with ctx:
                val_logits = eval_model.forward_ar(prefix)[0][:, -1, :]
            pred_val = val_logits.argmax(dim=-1)

            pos = pred_idx - cfg.index_token_start
            valid = (pos >= 0) & (pos < cfg.response_size)
            pos_c = pos.clamp(0, cfg.response_size - 1)
            already = seen.gather(1, pos_c.unsqueeze(1)).squeeze(1)
            valid = valid & ~already

            if valid.any():
                vi = valid.nonzero(as_tuple=True)[0]
                matched_gt = gt_resp[vi, pos_c[vi]]
                total_loss += F.cross_entropy(val_logits[vi], matched_gt, reduction='sum').item()
                total_tokens += vi.shape[0]
                seen[vi, pos_c[vi]] = True

            prefix = torch.cat([prefix, pred_val.unsqueeze(1)], dim=1)

    if ddp:
        agg = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
        total_loss, total_tokens = agg[0].item(), int(agg[1].item())

    return total_loss / total_tokens if total_tokens > 0 else float('nan')


# helps estimate an arbitrarily accurate loss over either split using many batches
# In DDP, all ranks compute independently then all_reduce for a better estimate.
@torch.inference_mode()
def estimate_loss():
    out = {}
    # Skip the (expensive) loss eval entirely when requested: under soft-index
    # mode the val split has no precomputed soft targets, so run_forward_train
    # falls back to online build_soft_index_targets per eval batch (very slow,
    # ~24min/eval). Returning NaNs keeps only the AR/PI accuracy eval.
    if skip_loss_eval:
        return {'train': float('nan'), 'val': float('nan')}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y, soft_Y = get_batch(split)
            with ctx:
                _, loss = run_forward_train(model, X, Y, soft_Y)
            losses.append(loss.detach())
        mean_loss = torch.stack(losses).mean() if losses else torch.tensor(float("nan"), device=device)
        if ddp:
            torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
            mean_loss = mean_loss / ddp_world_size
        out[split] = mean_loss.item()

    if test_quiz_arr is not None:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            eval_model = get_scoring_model(model)
            cfg = eval_model.config
            out['test'] = _rollout_test_loss(eval_model, test_quiz_arr, test_gt_response_arr, cfg)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)



#new_data = regenerate_trajectory_data()
#new_data.astype(np.uint16).tofile(os.path.join(out_dir, f'train_{iter_num}.bin'))
train_data_iter = iter_num
if init_from == 'resume':
    train_data_iter = (iter_num // round_interval) * round_interval
train_data_path = Path(out_dir) / f'train_{train_data_iter}.bin'
train_data_cfg = get_scoring_model(model).config
train_soft_index_path = _soft_index_file_for_train_path(train_data_path)
if init_from == 'scratch' and iter_num == 0:
    # Initial train_0.bin is ALWAYS built in L2R base order; --shuffle_order then
    # decides whether round-1 trains on it as-is (L2R) or randomly shuffled.
    _ensure_l2r_trajectory_file(train_data_path, train_data_cfg, "Initial trajectory file")
elif not _serialized_file_ready(train_data_path, train_data_cfg):
    if init_from == 'resume' and iter_num > 0 and iter_num % round_interval == 0:
        prev_train_path = Path(out_dir) / f'train_{iter_num - round_interval}.bin'
        if _serialized_file_ready(prev_train_path, train_data_cfg):
            train_data_path = prev_train_path
            _resume_needs_boundary_regen = True
            _skip_resume_eval = False
            if master_process:
                print(f"WARNING: Missing round-start trajectory {Path(out_dir) / f'train_{iter_num}.bin'}; "
                      f"using previous trajectory {prev_train_path} and regenerating at resume boundary.")
                logger.warning(f"Missing round-start trajectory at resume boundary; "
                               f"using {prev_train_path} and forcing regeneration.")
        elif iter_num - round_interval == 0:
            # train_0.bin missing (out_dir was cleaned). Safe to rebuild ONLY at the
            # round-1 boundary: _ensure_l2r_trajectory_file reproduces the L2R canonical
            # byte-for-byte from base (== the original round-1 train_0.bin), so resuming
            # round-2 from a hard round-1 ckpt stays equivalent to an uninterrupted run.
            _ensure_l2r_trajectory_file(prev_train_path, train_data_cfg, "rebuilt round-1 L2R train_0.bin")
            train_data_path = prev_train_path
            _resume_needs_boundary_regen = True
            _skip_resume_eval = False
            if master_process:
                print(f"Rebuilt missing round-1 L2R trajectory {prev_train_path}; "
                      f"regenerating round-2 at resume boundary.")
        else:
            raise FileNotFoundError(
                f"Training data for iter {iter_num} not found: {train_data_path}, "
                f"and previous round trajectory is also missing: {prev_train_path} "
                f"(only the round-1 boundary can auto-rebuild L2R; later rounds need the regen .bin)."
            )
    else:
        raise FileNotFoundError(
            f"Training data for iter {iter_num} not found: {train_data_path}. "
            "For resume, make sure the current round-start train_<round_start>.bin is present."
        )
train_soft_index_path = _soft_index_file_for_train_path(train_data_path)
if not _serialized_file_ready(train_data_path, train_data_cfg):
    raise FileNotFoundError(f"Training data is missing or incomplete after generation: {train_data_path}")
train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')

# --- Canonical trajectory (L2R) for warm+mix ---
canonical_traj_path = Path(out_dir) / 'canonical_traj.bin'
_canon_cfg = get_scoring_model(model).config
_ensure_l2r_trajectory_file(canonical_traj_path, _canon_cfg, "canonical_traj.bin")
data_canonical_traj = np.memmap(str(canonical_traj_path), dtype=np.uint16, mode='r')

# --- Multi-source data tracking for warm+mix ---
data_main = train_data
data_main_soft = _map_soft_index_file(train_soft_index_path, train_data_cfg)
if index_loss_mode == 'soft' and master_process:
    if data_main_soft is None:
        print("Soft-index cache not found for current main trajectory; using dynamic soft targets as fallback.")
        logger.info("Soft-index cache missing for main trajectory; dynamic soft targets fallback enabled.")
    else:
        print(f"Mapped soft-index cache: {train_soft_index_path}")
        logger.info(f"Mapped soft-index cache: {train_soft_index_path}")
# Truly uniform prev sampling: hold ALL prior-round trajectory files in a list.
# Per training step, each 'prev' batch element is drawn uniformly across the
# union of (file, row) pairs. New files are appended at every round boundary.
#
# Memory note: np.memmap with mode='r' does NOT load the file into RAM upfront.
# It maps virtual address space; pages are paged in by the OS on access and
# evicted under memory pressure. Holding many memmaps costs ~1 fd each and
# virtual address space (huge on 64-bit). Resident RAM is bounded by working
# set, not by total bytes mapped. We log totals so it stays visible.
data_prev_list = []
data_prev_soft_list = []
if iter_num >= round_interval:
    current_round = iter_num // round_interval
    _missing_prev = []
    _total_bytes = 0
    for r in range(current_round):
        candidate = Path(out_dir) / f'train_{r * round_interval}.bin'
        if candidate.exists():
            mm = np.memmap(str(candidate), dtype=np.uint16, mode='r')
            data_prev_list.append(mm)
            data_prev_soft_list.append(
                _map_soft_index_file(_soft_index_file_for_train_path(candidate), train_data_cfg)
            )
            _total_bytes += mm.nbytes
        else:
            _missing_prev.append(str(candidate))
    if master_process:
        print(f"Prev source: mapped {len(data_prev_list)} prior-round trajectory file(s) "
              f"({_total_bytes / 1e9:.2f} GB virtual; resident RAM grows only on access).")
        logger.info(f"Prev source: mapped {len(data_prev_list)} files, "
                    f"{_total_bytes / 1e9:.2f} GB total virtual.")
        for p in _missing_prev:
            print(f"  WARNING: missing prior trajectory: {p}")
            logger.warning(f"Missing prior trajectory: {p}")
        if not data_prev_list and (mix_ratios[1] > 0):
            print(f"  WARNING: mix_ratios prev share ({mix_ratios[1]}) will be merged into main.")

val_bin_path = data_dir / "val.bin"
if val_bin_path.exists():
    _val_cfg = get_scoring_model(model).config
    _val_raw = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
    val_data = serialize_base_array_for_loss(_val_raw, _val_cfg)
    if master_process:
        print(f"Loaded independent val data: {val_bin_path} ({len(_val_raw) // _val_cfg.base_seq_len} samples)")
    del _val_raw, _val_cfg
else:
    val_data = train_data
    if master_process:
        print(f"WARNING: val.bin not found in {data_dir}; using train_data as val_data")
# training loop
if online_regen_sample and iter_num >= round_interval:
    _refresh_online_regen_model(f'initial iter {iter_num}')
X, Y, soft_Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed

test_path = None
test_arr = None
test_loss_data = None
test_quiz_arr = None
test_gt_response_arr = None
if args.test_file is not None:
    test_path = resolve_optional_test_path(args.test_file)
    eval_cfg = get_scoring_model(model).config
    _test_raw = load_test_array(test_path, meta, eval_cfg)
    test_loss_data = serialize_base_array_for_loss(_test_raw, eval_cfg)
    _rows = np.asarray(_test_raw, dtype=np.uint16).reshape(-1, eval_cfg.base_seq_len)
    test_quiz_arr = _rows[:, :eval_cfg.quiz_size].copy()
    test_gt_response_arr = _rows[:, eval_cfg.quiz_size:eval_cfg.quiz_size + eval_cfg.response_size].copy()
    test_arr = _test_raw
    if master_process:
        print(f"Test eval enabled: {test_path} ({len(test_arr) // eval_cfg.base_seq_len} samples)")
        logger.info(f"Test eval enabled: {test_path}")
    del _test_raw, _rows
loss_metrics_path = Path(out_dir) / "loss_metrics.csv"

def append_loss_metrics(metric_iter, phase, lr=None, batch_train_loss=None,
                        eval_train_loss=None, val_loss=None, test_loss=None,
                        mfu_percent=None, elapsed_sec=None,
                        test_accuracy=None, test_malformed=None):
    if not master_process:
        return
    fieldnames = ["timestamp", "iter", "phase", "batch_train_loss",
                  "eval_train_loss", "val_loss", "test_loss", "lr",
                  "mfu_percent", "elapsed_sec", "test_accuracy", "test_malformed"]
    write_header = not loss_metrics_path.exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iter": metric_iter, "phase": phase,
        "batch_train_loss": "" if batch_train_loss is None else f"{batch_train_loss:.8g}",
        "eval_train_loss": "" if eval_train_loss is None else f"{float(eval_train_loss):.8g}",
        "val_loss": "" if val_loss is None else f"{float(val_loss):.8g}",
        "test_loss": "" if test_loss is None else f"{float(test_loss):.8g}",
        "lr": "" if lr is None else f"{lr:.8g}",
        "mfu_percent": "" if mfu_percent is None else f"{mfu_percent:.8g}",
        "elapsed_sec": "" if elapsed_sec is None else f"{elapsed_sec:.8g}",
        "test_accuracy": "" if test_accuracy is None else f"{test_accuracy:.8g}",
        "test_malformed": "" if test_malformed is None else str(test_malformed),
    }
    with open(loss_metrics_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

if master_process:
    print(f"Loss monitor: CSV {loss_metrics_path}")
    logger.info(f"Loss monitor: CSV {loss_metrics_path}")

running_mfu = -1.0
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num % round_interval) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    is_round_boundary = iter_num % round_interval == 0
    is_checkpoint_boundary = iter_num % checkpoint_interval == 0
    is_loss_eval = iter_num % eval_interval == 0 or is_checkpoint_boundary or is_round_boundary

    if is_loss_eval and _skip_resume_eval:
        _skip_resume_eval = False
        if master_process:
            print(f"Skipping redundant eval at resume point (iter {iter_num})")
            logger.info(f"Skipping redundant eval at resume point (iter {iter_num})")
        is_loss_eval = False

    if is_loss_eval:
        next_train_path = os.path.join(out_dir, f'train_{iter_num}.bin')

        # All ranks participate in loss estimation (all_reduce inside)
        losses = estimate_loss()

        should_regen = False
        should_save_checkpoint = iter_num > 0 and (is_checkpoint_boundary or is_round_boundary)
        ckpt_path = os.path.join(out_dir, f'{iter_num}_ckpt.pt') if should_save_checkpoint else None
        if master_process:
            loss_msg = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            if 'test' in losses:
                loss_msg += f", test loss {losses['test']:.4f}"
            print(loss_msg)
            logger.info(loss_msg)
            append_loss_metrics(
                iter_num, phase="eval", lr=lr,
                eval_train_loss=losses['train'],
                val_loss=losses['val'],
                test_loss=losses.get('test'),
                mfu_percent=None if running_mfu < 0 else running_mfu * 100,
            )
            if wandb_log:
                wandb_metrics = {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                }
                if 'test' in losses:
                    wandb_metrics["test/loss"] = losses['test']
                wandb.log(wandb_metrics)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
            if should_save_checkpoint:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                logger.info(f"saving checkpoint to {out_dir}")
                tmp_ckpt_path = f"{ckpt_path}.tmp.{os.getpid()}"
                try:
                    torch.save(checkpoint, tmp_ckpt_path)
                    os.replace(tmp_ckpt_path, ckpt_path)
                finally:
                    if os.path.exists(tmp_ckpt_path):
                        try:
                            os.remove(tmp_ckpt_path)
                        except OSError:
                            pass

            should_regen = is_round_boundary and 0 < iter_num < max_iters

        # Distributed test eval: all ranks evaluate disjoint slices then
        # all_reduce inside. Master alone writes results to disk.
        if should_save_checkpoint:
            record_round_test_results(iter_num, ckpt_path, test_path, test_arr, losses.get('test'))

        if _resume_needs_boundary_regen:
            should_regen = True
            _resume_needs_boundary_regen = False

        if first_round_only and is_round_boundary and iter_num > 0:
            if master_process:
                msg = f"First round complete at iter {iter_num}; stopping (first_round_only=True)."
                print(msg)
                logger.info(msg)
            break

        # Broadcast the regen decision from master to all ranks
        if ddp:
            regen_flag = torch.tensor([int(should_regen)], device=device)
            torch.distributed.broadcast(regen_flag, src=0)
            should_regen = bool(regen_flag.item())

        # All ranks participate at round boundaries. Online mode freezes a fresh
        # round-start sampler instead of writing one fixed train_<iter>.bin.
        if should_regen:
            if online_regen_sample:
                _refresh_online_regen_model(f'round boundary iter {iter_num}')
                X, Y, soft_Y = get_batch('train')
            else:
                if warm_from_best_round:
                    _warm_start_from_best_round_ckpt(iter_num)
                regen_override = None
                if args.regen_ckpt is not None:
                    regen_override = _load_regen_override_model(args.regen_ckpt)
                next_soft_index_path = _soft_index_file_for_train_path(next_train_path)
                regenerate_trajectory_data_parallel(
                    next_train_path,
                    override_model=regen_override,
                    soft_index_out_path=str(next_soft_index_path) if index_loss_mode == 'soft' else None,
                )
                if regen_override is not None:
                    del regen_override
                    torch.cuda.empty_cache()
                # The just-finished round's main goes into the prev pool for true
                # uniform-across-all-prior-rounds sampling next round.
                data_prev_list.append(data_main)
                data_prev_soft_list.append(data_main_soft)
                data_main = np.memmap(next_train_path, dtype=np.uint16, mode='r')
                data_main_soft = _map_soft_index_file(next_soft_index_path, get_scoring_model(model).config)
                train_data = data_main
                if master_process:
                    if index_loss_mode == 'soft' and data_main_soft is not None:
                        print(f"Mapped soft-index cache: {next_soft_index_path}")
                        logger.info(f"Mapped soft-index cache: {next_soft_index_path}")
                    print(f"Prev pool size now {len(data_prev_list)} after round boundary at iter {iter_num}.")
                    logger.info(f"Prev pool size now {len(data_prev_list)} after iter {iter_num}.")

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = run_forward_train(model, X, Y, soft_Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        X, Y, soft_Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
        append_loss_metrics(
            iter_num, phase="train_step", lr=lr,
            batch_train_loss=lossf,
            mfu_percent=None if running_mfu < 0 else running_mfu * 100,
            elapsed_sec=dt,
        )
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
