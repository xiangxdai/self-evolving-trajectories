"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python serfox_train.py --n_layer 6 --n_head 6 --n_embd 210 --max_iters 1000000

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 serfox_train.py --n_layer 6 --n_head 6 --n_embd 210 --max_iters 1000000

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 serfox_train.py --n_layer 6 --n_head 6 --n_embd 210 --max_iters 1000000
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 serfox_train.py --n_layer 6 --n_head 6 --n_embd 210 --max_iters 1000000
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse
from pathlib import Path


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from serfox_model import GPTConfig, GPT
from logger import get_logger
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# the input parameters

parser = argparse.ArgumentParser(description='Train Ser-FOX.')

parser.add_argument('--n_layer', type=int, default=6, help='Number of layers (default: 1)')  
parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads (default: 1)')  
parser.add_argument('--n_embd', type=int, default=210, help='Size of the embeddings (default: 384)')
parser.add_argument('--max_iters', type=int, default=1000000, help='Number of Iterations (default: 100000)')



args = parser.parse_args()

n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters

num_rounds = 10

seed = 1337

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_ROOT = REPO_ROOT / "out" / "serfox_train"

data_dir = REPO_ROOT / 'data'
with open(REPO_ROOT / 'meta.pkl', 'rb') as f:
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
vocab_size = value_vocab_size + response_size
block_size = quiz_size + 3 * response_size

out_dir = str(OUT_ROOT / f'{n_layer}_{n_head}_{n_embd}_{seed}')

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = max_iters // num_rounds
log_interval =  max_iters // (num_rounds*10)
eval_iters = max_iters // 10000

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = f'serfox-{n_layer}_{n_head}_{n_embd}_{seed}'
# data
#dataset = 'reasoning'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
train_batch_size = 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
val_batch_size = 1024
batch_size = train_batch_size
#block_size = 64
# model
#n_layer = 1 #12
#n_head = 1 #12
#n_embd = 384 #768


dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 2e-4 # max learning rate 
#max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = max_iters//(20*num_rounds) # how many steps to warm up for
lr_decay_iters = max_iters//num_rounds # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

'''check_type = 'shortest'
max_path_len = 10
max_new_tokens = 200
flag = 0
test_interval = 100'''
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
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
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * train_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
###torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


base_data = np.memmap(REPO_ROOT / 'base.bin', dtype=np.uint16, mode='r')
#print(len(base_data))
#print(xxx)

# poor man's data loader




def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size

    prompt_len = quiz_size
    target_len = response_size
    data_size = prompt_len + 2 * target_len

    ix = torch.randint(len(data)//data_size , (batch_size,)) *data_size
    z = torch.stack([torch.from_numpy((data[i:i+data_size]).astype(np.int64)) for i in ix])

    x = z[:,:-1].clone()
    y = z[:,1:].clone()
    y[:, :prompt_len - 1] = -100


    '''

    a, b = y.shape

    # 生成所有列的下标 0..b-1
    cols = torch.arange(b)

    # 1) 前 k-1 位
    mask1 = cols < (k - 1)

    # 2) k, k+2, k+4, ...
    mask2 = ((cols - k+1) % 2 == 0) & (cols >= k-1)
    #print(mask1)
    #print(mask2)

    # 合并 mask
    mask = mask1 | mask2        # shape (b,)

    # 扩展到 (a, b)
    mask = mask.unsqueeze(0).expand(a, b)

    # 设成 -1
    y = y.masked_fill(mask, -100)
    '''


    #print(x[0])
    #print(y[0])
    #print(z[0])
    #print(xxx)
    
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# logger
logger = get_logger(os.path.join(out_dir, "train.log"))






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



if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

if init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, '0_ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
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
    #iter_num = 0
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
'''
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
'''
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


@torch.no_grad()
def regenerate_trajectory_data():
    was_training = model.training
    model.eval()

    train_data = np.memmap(REPO_ROOT / 'base.bin', dtype=np.uint16, mode='r')

    try:
        scoring_model = get_scoring_model(model)
        cfg = scoring_model.config
        num_blocks = len(train_data) // cfg.base_seq_len
        blocks_per_step = 1000
        out_chunks = []

        for start in range(0, num_blocks, blocks_per_step):
            end = min(start + blocks_per_step, num_blocks)

            z = torch.from_numpy(
                train_data[start * cfg.base_seq_len : end * cfg.base_seq_len].astype(np.int64)
            ).reshape(-1, cfg.base_seq_len)

            z = z.to(device)
            a, _ = z.shape

            z_basic = z[:, :cfg.quiz_size]
            z_result = z[:, cfg.quiz_size : cfg.quiz_size + cfg.response_size]

            index_tokens = torch.arange(0, cfg.num_index_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(a, 1)
            index_tokens = index_tokens + cfg.index_token_start

            for _ in range(cfg.response_size):
                z_basic_app = torch.cat([z_basic, index_tokens], dim=1)

                with ctx:
                    index_logits = scoring_model.score_parallel_indices(
                        z_basic_app, cfg.num_index_tokens
                    )

                probs = F.softmax(index_logits, dim=-1)

                eq = z_basic_app[:, :, None] == z_basic_app[:, None, :]
                has_previous_same = eq.tril(diagonal=-1).any(dim=-1)
                decode = ~has_previous_same
                decode = decode[:, -cfg.num_index_tokens:]

                p = probs.gather(dim=2, index=z_result.unsqueeze(-1)).squeeze(-1)
                p = torch.where(decode, p, -1)

                _, max_idx = p.max(dim=1)
                max_token = z_result.gather(dim=1, index=max_idx.unsqueeze(-1))

                z_basic = torch.cat(
                    [
                        z_basic,
                        max_idx.unsqueeze(-1) + cfg.index_token_start,
                        max_token,
                    ],
                    dim=1,
                )

            out_chunks.append(z_basic.detach().cpu().reshape(-1).numpy())
            if start == 0:
                preview_len = cfg.train_seq_len
                preview_count = min(5, z.shape[0])
                for preview_idx in range(preview_count):
                    print(z[preview_idx])
                    start_offset = preview_idx * preview_len
                    end_offset = start_offset + preview_len
                    print(out_chunks[0][start_offset:end_offset])

        return np.concatenate(out_chunks, axis=0)
    finally:
        if was_training:
            model.train()





# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = run_forward_ar(model, X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
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
train_data = np.memmap(os.path.join(out_dir, f'train_{iter_num}.bin'), dtype=np.uint16, mode='r')
val_data = train_data
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
accuracy = []
corrects = []
totals = []
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num % (max_iters//num_rounds)) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        refresh_train_data = False
        next_train_path = os.path.join(out_dir, f'train_{iter_num}.bin')

        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
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
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))

                    new_data = regenerate_trajectory_data()
                    new_data.astype(np.uint16).tofile(next_train_path)
                    refresh_train_data = True

        if ddp:
            refresh_flag = torch.tensor([int(refresh_train_data)], device=device)
            torch.distributed.broadcast(refresh_flag, src=0)
            refresh_train_data = bool(refresh_flag.item())

        if refresh_train_data:
            train_data = np.memmap(next_train_path, dtype=np.uint16, mode='r')
            val_data = train_data

    # if iter_num % test_interval == 0 and master_process:
    #     correct, tot = test_model()
    #     corrects.append(correct)
    #     totals.append(tot)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = run_forward_ar(model, X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        X, Y = get_batch('train')
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
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        #new_data = regenerate_trajectory_data()
        #new_data.astype(np.uint16).tofile(f'train_{iter_num}.bin')
        break

torch.save(torch.tensor(corrects).cpu(), os.path.join(out_dir, f'corrects.pt'))
torch.save(torch.tensor(totals).cpu(), os.path.join(out_dir, f'totals.pt'))

if ddp:
    destroy_process_group()
