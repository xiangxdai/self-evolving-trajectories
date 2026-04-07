"""
Unified Tom-CAT training entrypoint.

Supports both scratch and resume training across reasoning-task datasets.
Dataset families provide profile defaults, with the main distinction being
write-only supervision versus read-and-write supervision on the second segment.
"""

import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tomcat_model import GPT, GPTConfig
from logger import get_logger

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "out"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# Command-line arguments.
parser = argparse.ArgumentParser(description="Train Tom-CAT on a reasoning task")
parser.add_argument("--dataset", type=str, default="cd5", help="Dataset name (folder under data/)")
parser.add_argument("--meta_file", type=str, default=None, help="Optional metadata file path or name")
parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint path to resume from")
parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
parser.add_argument("--n_embd", type=int, default=768, help="Embedding size")
parser.add_argument("--max_iters", type=int, default=None, help="Total training iterations for scratch mode")
parser.add_argument("--additional_iters", type=int, default=None, help="Additional iterations for resume mode")
parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
parser.add_argument("--compile", type=parse_bool, default=True, help="Use torch.compile")
parser.add_argument("--suffix", type=str, default=None, help="Suffix appended to the output directory name")
parser.add_argument("--out_dir", type=str, default=None, help="Optional explicit output directory")
args = parser.parse_args()


def infer_training_profile(dataset):
    # Keep dataset-family defaults in one place so the rest of the script can stay
    # task-agnostic. One important distinction is whether we supervise only the
    # write space or both the partially masked second segment and the write space.
    profile = {
        "meta_candidates": [
            "meta.pkl",
            "cd_meta.pkl",
            "sat_meta.pkl",
            f"{dataset}meta.pkl",
            f"{dataset}_meta.pkl",
        ],
        "default_max_iters": 1_000_000,
        "default_batch_size": 512,
        "supervision_mode": "write_only",
        "out_dir_name": "tomcat_write_only_train",
        "wandb_project": dataset,
    }
    if "sat" in dataset:
        profile["meta_candidates"] = ["sat_meta.pkl"] + profile["meta_candidates"]
        profile["wandb_project"] = "sat"
    elif dataset.startswith("cd4"):
        profile["meta_candidates"] = ["cd_meta.pkl"] + profile["meta_candidates"]
        profile["default_max_iters"] = 50_000
        profile["supervision_mode"] = "read_and_write"
        profile["out_dir_name"] = "tomcat_read_write_train"
        profile["wandb_project"] = "cd4"
    elif dataset.startswith("cd"):
        profile["meta_candidates"] = ["cd_meta.pkl"] + profile["meta_candidates"]
        profile["wandb_project"] = "cd"
    return profile


def apply_checkpoint_profile_overrides(profile, checkpoint_config):
    merged_profile = dict(profile)
    if "supervision_mode" in checkpoint_config:
        merged_profile["supervision_mode"] = checkpoint_config["supervision_mode"]
    elif "target_mode" in checkpoint_config:
        # Older checkpoints stored the same concept under target_mode.
        merged_profile["supervision_mode"] = checkpoint_config["target_mode"]
    return merged_profile


def resolve_path(candidates):
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def resolve_meta_path(dataset, meta_file, profile):
    if meta_file is not None:
        meta_path = Path(meta_file).expanduser()
        candidates = [
            meta_path,
            DATA_ROOT / meta_path,
            DATA_ROOT / dataset / meta_path,
        ]
        meta_path = resolve_path(candidates)
        if meta_path is None:
            raise FileNotFoundError(f"Could not find meta file from candidates: {candidates}")
        return meta_path

    candidates = [DATA_ROOT / dataset / name for name in profile["meta_candidates"]]
    meta_path = resolve_path(candidates)
    if meta_path is None:
        raise FileNotFoundError(f"Could not infer meta file for dataset '{dataset}'. Tried: {candidates}")
    return meta_path


def resolve_resume_checkpoint(checkpoint_path):
    # Accept the current output layout and the older TeachlessARThree naming so
    # historical checkpoints from that family remain resumable.
    checkpoint = Path(checkpoint_path).expanduser()
    candidates = [checkpoint]
    if not checkpoint.is_absolute():
        candidates.append(REPO_ROOT / checkpoint)

    corrected_candidates = []
    for candidate in candidates:
        corrected_candidates.append(candidate)
        candidate_text = str(candidate)
        corrected_candidates.append(Path(candidate_text.replace("TeachlessARThree3Train", "TeachlessARThree/3Train")))
        corrected_candidates.append(Path(candidate_text.replace("/TeachlessARThree3Train", "/TeachlessARThree/3Train")))

    resolved_checkpoint = resolve_path(corrected_candidates)
    if resolved_checkpoint is not None:
        if resolved_checkpoint != checkpoint:
            print(f"Found checkpoint at corrected path: {resolved_checkpoint}")
        return resolved_checkpoint
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def default_out_dir(dataset, n_layer, n_head, n_embd, suffix, max_iters, profile):
    return OUT_ROOT / profile["out_dir_name"] / f"{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}"


def strip_unwanted_prefix(state_dict):
    # torch.compile can save state_dict keys with an `_orig_mod.` prefix.
    # Strip it so compiled checkpoints load into the plain GPT module.
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    return state_dict


def validate_resume_metadata(meta, checkpoint_model_args):
    # Resume mode loads model weights from the checkpoint, so the on-disk dataset
    # metadata must still agree with the saved model shape assumptions.
    expected_block_size = meta["quiz_size"] + meta["response_size"] * 2 - 1
    checks = {
        "quiz_size": (meta["quiz_size"], checkpoint_model_args["quiz_size"]),
        "response_size": (meta["response_size"], checkpoint_model_args["response_size"]),
        "block_size": (expected_block_size, checkpoint_model_args["block_size"]),
        "vocab_size": (meta["vocab_size"], checkpoint_model_args["vocab_size"]),
    }
    mismatches = [
        f"{name}: meta={meta_value}, checkpoint={checkpoint_value}"
        for name, (meta_value, checkpoint_value) in checks.items()
        if meta_value != checkpoint_value
    ]
    if mismatches:
        raise ValueError(
            "Resolved dataset metadata does not match the checkpoint model_args: "
            + "; ".join(mismatches)
        )


resume_mode = args.resume_from is not None
initial_profile = infer_training_profile(args.dataset)

if resume_mode:
    # Resume mode reuses the checkpoint architecture and optimizer state, while
    # still allowing run-length and batch-size overrides from the command line.
    checkpoint_path = resolve_resume_checkpoint(args.resume_from)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_model_args = dict(checkpoint["model_args"])
    checkpoint_config = dict(checkpoint["config"])
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    dataset = checkpoint_config["dataset"]
    profile = apply_checkpoint_profile_overrides(infer_training_profile(dataset), checkpoint_config)
    meta_path = resolve_meta_path(dataset, args.meta_file, profile)

    n_layer = checkpoint_model_args["n_layer"]
    n_head = checkpoint_model_args["n_head"]
    n_embd = checkpoint_model_args["n_embd"]
    block_size = checkpoint_model_args["block_size"]
    bias = checkpoint_model_args["bias"]
    dropout = checkpoint_model_args["dropout"]
    quiz_size = checkpoint_model_args["quiz_size"]
    response_size = checkpoint_model_args["response_size"]
    vocab_size = checkpoint_model_args["vocab_size"]

    suffix = args.suffix or "resume-1"
    additional_iters = args.additional_iters if args.additional_iters is not None else 180_000
    max_iters = iter_num + additional_iters
    if args.batch_size is not None:
        train_batch_size = args.batch_size
        val_batch_size = max(1, train_batch_size // 2)
    else:
        train_batch_size = checkpoint_config.get("train_batch_size", 256)
        val_batch_size = checkpoint_config.get("val_batch_size", max(1, train_batch_size // 2))
else:
    # Scratch mode builds model/config defaults directly from the requested dataset.
    checkpoint = None
    checkpoint_config = {}
    checkpoint_model_args = {}
    iter_num = 0
    best_val_loss = 1e9
    dataset = args.dataset
    profile = initial_profile
    meta_path = resolve_meta_path(dataset, args.meta_file, profile)

    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    bias = False
    dropout = 0.1

    suffix = args.suffix or "0-1"
    additional_iters = None
    max_iters = args.max_iters if args.max_iters is not None else profile["default_max_iters"]
    train_batch_size = args.batch_size if args.batch_size is not None else profile["default_batch_size"]
    val_batch_size = max(1, train_batch_size // 2)

data_dir = DATA_ROOT / dataset

# Load dataset metadata.
with open(meta_path, "rb") as handle:
    meta = pickle.load(handle)

if resume_mode:
    validate_resume_metadata(meta, checkpoint_model_args)

stoi, itos = meta["stoi"], meta["itos"]
data_size = meta["data_size"]
quiz_size = meta["quiz_size"]
response_size = meta["response_size"]
block_size = quiz_size + response_size * 2 - 1
vocab_size = meta["vocab_size"]

# Resolve the special tokens used by Tom-CAT training.
pad_token_id = stoi["<PAD>"]
sep_token_id = stoi.get("<SEP>")
if sep_token_id is None:
    raise KeyError("The dataset vocabulary does not define <SEP>, which is required for Tom-CAT training.")

if "<MASK>" not in stoi:
    raise KeyError("The dataset vocabulary does not define <MASK>, which is required for Tom-CAT training.")
mask_token_id = stoi["<MASK>"]

if "$" not in stoi:
    raise KeyError("The dataset vocabulary does not define the '$' placeholder token required by Tom-CAT.")
dollar_token_id = stoi["$"]

if resume_mode:
    existing_mask_token_id = checkpoint_model_args.get("mask_token_id")
    if existing_mask_token_id is not None and existing_mask_token_id != mask_token_id:
        raise ValueError(
            f"Resolved mask_token_id {mask_token_id} does not match checkpoint mask_token_id {existing_mask_token_id}."
        )
    existing_dollar_token_id = checkpoint_model_args.get("dollar_token_id")
    if existing_dollar_token_id is not None and existing_dollar_token_id != dollar_token_id:
        raise ValueError(
            f"Resolved dollar_token_id {dollar_token_id} does not match checkpoint dollar_token_id {existing_dollar_token_id}."
        )
    checkpoint_model_args.setdefault("mask_token_id", mask_token_id)
    checkpoint_model_args.setdefault("dollar_token_id", dollar_token_id)
    model_args = checkpoint_model_args
else:
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
        mask_token_id=mask_token_id,
        dollar_token_id=dollar_token_id,
    )

# Linear learning-rate scaling rule: keep the effective step size roughly
# comparable when the batch size changes from the 512-example reference setup.
base_batch_size = 512
base_learning_rate = 3e-4
learning_rate = base_learning_rate * (train_batch_size / base_batch_size)
print(f"Using scaled learning rate = {learning_rate:.6f} for batch size = {train_batch_size}")

weight_decay = checkpoint_config.get("weight_decay", 1e-1)
beta1 = checkpoint_config.get("beta1", 0.9)
beta2 = checkpoint_config.get("beta2", 0.95)
grad_clip = checkpoint_config.get("grad_clip", 1.0)
decay_lr = checkpoint_config.get("decay_lr", True)
gradient_accumulation_steps = checkpoint_config.get("gradient_accumulation_steps", 1)
backend = checkpoint_config.get("backend", "nccl")
dtype = checkpoint_config.get("dtype", "bfloat16")
compile_model = args.compile

target_interval_span = additional_iters if resume_mode else max_iters
eval_interval = max(1, target_interval_span // 10)
log_interval = max(1, target_interval_span // 100)
# Evaluate on a fixed small window by default; using max_iters // 10 here would
# make long runs spend far too much wall-clock time inside evaluation.
eval_iters = checkpoint_config.get("eval_iters", max(1, min(200, target_interval_span)))
always_save_checkpoint = True
eval_only = False
wandb_log = False
wandb_project = profile["wandb_project"]
wandb_run_name = f"tomcat-{dataset}-{suffix}"

warmup_iters = checkpoint_config.get("warmup_iters", max(1, max_iters // 20))
lr_decay_iters = max_iters
min_lr = learning_rate / 10

if args.out_dir is not None:
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
else:
    out_dir = default_out_dir(
        dataset,
        n_layer,
        n_head,
        n_embd,
        suffix,
        max_iters,
        profile,
    )

config = dict(
    dataset=dataset,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    max_iters=max_iters,
    train_batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    grad_clip=grad_clip,
    decay_lr=decay_lr,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    gradient_accumulation_steps=gradient_accumulation_steps,
    dtype=dtype,
    backend=backend,
    compile=compile_model,
    supervision_mode=profile["supervision_mode"],
    eval_iters=eval_iters,
)

# DDP setup.
ddp = int(os.environ.get("RANK", -1)) != -1
device = "cuda" if torch.cuda.is_available() else "cpu"

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    # Treat gradient_accumulation_steps in config as the global value.
    # Under DDP, convert it to per-rank micro-steps.
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    ddp_world_size = 1
    master_process = True
    seed_offset = 0

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
if ddp:
    # Only rank 0 creates the output directory; other ranks wait here before
    # touching the logger or any files inside that directory.
    torch.distributed.barrier()

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
# Use autocast on GPU and a no-op context on CPU.
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader.
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    batch_size = train_batch_size if split == "train" else val_batch_size
    ignore_index = -100

    # Packed layout:
    #   x = [quiz] [masked_response] [response[:-1]]
    #   y = [ignore on most quiz positions] [optional response supervision on the
    #       masked-response segment] [response supervision on the write-space segment]
    #
    # Labels are aligned for next-token prediction, so supervision starts at
    # quiz_size - 1: the last quiz token predicts response[0].

    # Make sure the sampled window stays within one serialized example.
    max_index = len(data) - data_size
    if max_index < 0:
        raise ValueError(f"Data size {len(data)} is smaller than data_size {data_size}")

    # Randomly sample example start indices; each example occupies data_size tokens.
    ix = torch.randint(0, (max_index // data_size) + 1, (batch_size,)) * data_size

    # Split each packed example into the quiz segment and the target response segment.
    quizzes = torch.stack([torch.from_numpy(data[i:i + quiz_size].astype(np.int64)) for i in ix])
    responses = torch.stack([torch.from_numpy(data[i + quiz_size:i + data_size].astype(np.int64)) for i in ix])

    # Each sample draws its own mask ratio in [0.001, 1).
    mask_ratios = torch.empty(batch_size).uniform_(0.001, 1)

    # Initialize the response mask as all False, then mark per-sample positions.
    mask = torch.zeros((batch_size, response_size), dtype=torch.bool)
    for batch_index in range(batch_size):
        # Determine how many response tokens to mask for this sample.
        num_to_mask = max(1, int(mask_ratios[batch_index].item() * response_size))

        # Randomly choose the response positions to mask.
        mask_positions = torch.randperm(response_size)[:num_to_mask]
        mask[batch_index, mask_positions] = True

    # The second segment is the partially masked response. Its final token is
    # always forced to <SEP> so the segment boundary stays explicit.
    response_masked = responses.clone()
    response_masked[mask] = mask_token_id
    response_masked[:, -1] = sep_token_id

    # Construct x = quiz + response_masked + response[:-1].
    x = torch.zeros((batch_size, block_size), dtype=torch.int64)
    x[:, :quiz_size] = quizzes
    x[:, quiz_size:quiz_size + response_size] = response_masked
    x[:, quiz_size + response_size:block_size] = responses[:, :-1]

    # Construct y for next-token prediction. Most of the quiz prefix is ignored,
    # but the last quiz token predicts response[0]. In read_and_write mode, the
    # partially masked second segment also predicts the full response; the final
    # write-space segment always predicts the full response.
    y = torch.full((batch_size, block_size), ignore_index, dtype=torch.int64)
    if profile["supervision_mode"] == "read_and_write":
        y[:, quiz_size - 1:quiz_size - 1 + response_size] = responses
    y[:, quiz_size - 1 + response_size:block_size] = responses

    # Move to the active device; pin memory on CUDA for faster transfer.
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Logger and model initialization.
logger = get_logger(os.path.join(out_dir, "train.log")) if master_process else None

if resume_mode:
    print(f"Resuming training from iteration {iter_num} with best val loss {best_val_loss:.4f}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(strip_unwanted_prefix(dict(checkpoint["model"])))
else:
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
last_val_loss = checkpoint.get("last_val_loss", best_val_loss) if resume_mode else None
if resume_mode:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if compile_model:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    # Wrap only the training model. Evaluation uses raw_model so only rank 0
    # runs eval forwards while the other ranks wait at barriers.
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss(eval_model):
    # Evaluate on both train and val splits using the unwrapped model.
    out = {}
    eval_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for loss_index in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = eval_model(X, Y)
            losses[loss_index] = loss.item()
        out[split] = losses.mean()
    eval_model.train()
    return out


def get_lr(step):
    # Cosine decay with linear warmup.
    if step < warmup_iters:
        return learning_rate * step / warmup_iters
    if step > lr_decay_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop.
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if resume_mode and master_process:
    start_message = f"Resuming {dataset} training from iteration {iter_num} to {max_iters}"
    print(start_message)
    logger.info(start_message)

while iter_num < max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        if ddp:
            # Synchronize before and after evaluation so non-master ranks do not
            # race ahead into backward passes while rank 0 is still evaluating.
            torch.distributed.barrier()
        if master_process:
            losses = estimate_loss(raw_model)
            eval_message = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            print(eval_message)
            logger.info(eval_message)

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })

            last_val_loss = losses["val"].item()
            save_due_to_improvement = last_val_loss < best_val_loss
            if save_due_to_improvement:
                best_val_loss = last_val_loss
            if save_due_to_improvement or always_save_checkpoint:
                if iter_num > 0:
                    checkpoint_to_save = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "last_val_loss": last_val_loss,
                        "config": config,
                    }
                    save_message = f"saving checkpoint to {out_dir}"
                    print(save_message)
                    logger.info(save_message)
                    torch.save(checkpoint_to_save, os.path.join(out_dir, f"{iter_num}_ckpt.pt"))
        if ddp:
            torch.distributed.barrier()

    if iter_num == 0 and eval_only:
        break

    # Gradient accumulation across micro-steps.
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            # Skip the first few iterations before estimating MFU to avoid noisy startup timing.
            mfu = raw_model.estimate_mfu(train_batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        log_message = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        print(log_message)
        logger.info(log_message)

    iter_num += 1
    local_iter_num += 1

if ddp:
    torch.distributed.barrier()

if master_process:
    # Always save a final checkpoint, even if the last step was not an eval step.
    final_checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "last_val_loss": last_val_loss,
        "config": config,
    }
    final_save_message = f"Training completed. Saving final checkpoint to {out_dir}"
    print(final_save_message)
    logger.info(final_save_message)
    torch.save(final_checkpoint, os.path.join(out_dir, f"{iter_num}_final_ckpt.pt"))

if ddp:
    torch.distributed.barrier()
    destroy_process_group()
