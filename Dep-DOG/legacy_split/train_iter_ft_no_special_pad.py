import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

SCRIPT_DIR = Path(__file__).resolve().parent
DTO_DIR = SCRIPT_DIR.parent
if str(DTO_DIR) not in sys.path:
    sys.path.insert(0, str(DTO_DIR))

from depdog_model import GPTConfig, GPT
from logger import get_logger
import re

# -----------------------------------------------------------------------------
# Command-line arguments
parser = argparse.ArgumentParser(description='Training of NanoGPT for cd with MDM - Iterative Round')
parser.add_argument('--dataset', type=str, default='cd4$', help='Name of the dataset to use')
parser.add_argument('--meta_file', type=str, default='cd_meta.pkl', help='Metadata pickle file') # Add this
parser.add_argument('--n_layer', type=int, default=3, help='Number of layers')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=384, help='Size of the embeddings')
parser.add_argument('--max_iters', type=int, default=1000000, help='Number of iterations')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--gen_batch_size', type=int, default=1024, help='Batch size for generation phase')
parser.add_argument('--compile', type=bool, default=True, help='Use PyTorch 2.0 compilation')
parser.add_argument('--suffix', type=str, default='round3_FT_mix', help='Current Round Suffix') 
parser.add_argument('--prev_round_file', type=str, default='train_gen_round3.bin', help='Filename of data from previous round')
# Previous-round checkpoint path
parser.add_argument('--prev_round_ckpt', type=str, default='out/MDMx0/cd4$_3_12_384_round2_FT_1000000/1000000_ckpt.pt', help='Path to the checkpoint from the previous round to warm-start from')

args = parser.parse_args()

# Extract arguments
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
train_batch_size = args.batch_size
gen_batch_size = args.gen_batch_size
val_batch_size = args.batch_size // 2
compile = args.compile
suffix = args.suffix
prev_round_file = args.prev_round_file
prev_round_ckpt = args.prev_round_ckpt  # Previous-round checkpoint path

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', dataset)

# Load metadata
meta_file = args.meta_file  # Extract here
with open(os.path.join(data_dir, meta_file), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
meta_data_size = meta['data_size'] 
quiz_size = meta['quiz_size']   # 15
response_size = meta['response_size']  # 35
vocab_size = meta['vocab_size']  # 18
mask_token_id = stoi['<MASK>'] 

# Round 2+ data block size (because it includes indices)
current_block_size = quiz_size + 2 * response_size

out_dir = f'out/MDMx0/{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}'

# -----------------------------------------------------------------------------
# Config
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 1000
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # Default to scratch and override it later if arguments require it
wandb_log = False
wandb_project = 'cd-mdm'
wandb_run_name = f'gpt-cd-mdm-{suffix}'
gradient_accumulation_steps = 1
dropout = 0.1
bias = False

# Auto LR Scaling
base_batch_size = 512
base_learning_rate = 3e-4
learning_rate = base_learning_rate * (train_batch_size / base_batch_size)
print(f"Using scaled learning rate = {learning_rate:.6f}")

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * current_block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
train_data_path = os.path.join(data_dir, prev_round_file)
if not os.path.exists(train_data_path):
    print(f"Warning: {train_data_path} not found, falling back to base train.bin")
    train_data_path = os.path.join(data_dir, 'train.bin')
    # Note: if we fall back to train.bin, block_size may no longer match; this assumes the user-provided path is normally correct

print(f"Loading training data from: {train_data_path}")
train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
# Assume validation data either reuses training data or is handled separately
val_data = train_data 

# ==============================================================================
# Batch Logic
# ==============================================================================
def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    
    max_index = len(data) - current_block_size
    ix = torch.randint(0, (max_index // current_block_size) + 1, (batch_size,)) * current_block_size
    
    z = torch.stack([torch.from_numpy((data[i:i+current_block_size]).astype(np.int64)) for i in ix])
    z = z.to(device, non_blocking=True)

    z_quiz = z[:, :quiz_size]
    z_generated = z[:, quiz_size:] 
    
    z_indices = z_generated[:, ::2]  
    z_values  = z_generated[:, 1::2] 
    
    rel_indices = z_indices - quiz_size 
    rel_indices_safe = rel_indices.clamp(0, response_size - 1)
    
    sorted_gather_idx = torch.argsort(rel_indices_safe, dim=1)
    y_response = torch.gather(z_values, 1, sorted_gather_idx)
    
    num_visible = torch.randint(0, response_size, (batch_size, 1), device=device)
    x_response = y_response.clone()
    mask_canvas = torch.zeros((batch_size, response_size), dtype=torch.bool, device=device)
    range_row = torch.arange(response_size, device=device).unsqueeze(0)
    should_mask_in_list = range_row >= num_visible
    
    mask_canvas.scatter_(1, rel_indices_safe, should_mask_in_list)
    x_response[mask_canvas] = mask_token_id

    zero_mask_rows = (x_response == mask_token_id).sum(dim=1) == 0
    if zero_mask_rows.any().item():
        raise RuntimeError(
            f"Dep-DOG/train_iter_ft_no_special_pad.py get_batch produced {zero_mask_rows.sum().item()} sample(s) without any masked response tokens; loss normalization would divide by zero."
        )
    
    x = torch.cat([z_quiz, x_response], dim=1)
    y = torch.cat([z_quiz, y_response], dim=1)
    
    is_masked = (x == mask_token_id)
    y = torch.where(is_masked, y, torch.tensor(-100, device=device))
    
    x = x.clamp(0, vocab_size - 1)
    y_is_ignored = (y == -100)
    y = y.clamp(0, vocab_size - 1)
    y = torch.where(y_is_ignored, torch.tensor(-100, device=device), y)
    
    return x, y

# ==============================================================================
# GENERATION FUNCTION (Re-Ordering)
# ==============================================================================
@torch.no_grad()
def generate_data(model):
    print(f"Starting data generation (Re-Ordering) for NEXT round...")
    model.eval()
    
    num_samples = len(train_data) // current_block_size
    generated_chunks = []
    
    if ddp:
        my_indices = np.array_split(np.arange(num_samples), ddp_world_size)[ddp_rank]
        if len(my_indices) == 0:
            total_range = []
        else:
            start_idx_base = my_indices[0]
            end_idx_base = my_indices[-1] + 1
            total_range = range(start_idx_base, end_idx_base, gen_batch_size)
    else:
        total_range = range(0, num_samples, gen_batch_size)

    for i, start_idx in enumerate(total_range):
        end_idx = min(start_idx + gen_batch_size, num_samples if not ddp else end_idx_base)
        current_bs = end_idx - start_idx
        if current_bs <= 0: break

        batch_indices = np.arange(start_idx, end_idx) * current_block_size
        batch_data_raw = [train_data[idx : idx + current_block_size] for idx in batch_indices]
        z_raw = torch.tensor(np.array(batch_data_raw).astype(np.int64)).to(device)
        
        quiz_part = z_raw[:, :quiz_size]
        gen_part = z_raw[:, quiz_size:]
        old_indices = gen_part[:, ::2] - quiz_size 
        old_values = gen_part[:, 1::2]
        
        sort_idx = torch.argsort(old_indices, dim=1)
        response_gt = torch.gather(old_values, 1, sort_idx)
        
        z_clean = torch.cat([quiz_part, response_gt], dim=1)
        
        z_order = quiz_part.clone()
        z_mask = z_clean.clone()
        z_mask[:, quiz_size:] = mask_token_id 
        
        for step in range(response_size):
            with ctx:
                logits, _ = model(z_mask)
            
            probs = F.softmax(logits, dim=-1)
            decode = (z_mask == mask_token_id)
            
            p = probs.gather(dim=2, index=z_clean.unsqueeze(-1)).squeeze(-1)
            p = torch.where(decode, p, torch.tensor(-1.0, device=device))
            
            _, max_idx = p.max(dim=1) 
            max_token = z_clean.gather(dim=1, index=max_idx.unsqueeze(-1))
            
            scatter_mask = torch.zeros_like(z_mask, dtype=torch.bool)
            scatter_mask.scatter_(1, max_idx.unsqueeze(-1), True)
            z_mask = torch.where(scatter_mask, z_clean, z_mask)
            
            z_order = torch.cat([z_order, max_idx.unsqueeze(-1), max_token], dim=1)

        generated_chunks.append(z_order.detach().cpu().numpy().astype(np.uint16))
        
        if i % 10 == 0:
            print(f"[Rank {ddp_rank}] Re-ordering batch {i}/{len(total_range)}")

    if len(generated_chunks) > 0:
        local_data = np.concatenate(generated_chunks, axis=0)
    else:
        local_data = np.array([], dtype=np.uint16)

    output_file = os.path.join(data_dir, f'train_{max_iters}_rank{ddp_rank}.bin')
    local_data.tofile(output_file)
    
    if ddp:
        torch.distributed.barrier()
        
    if master_process:
        print("Master process merging re-ordered files...")
        all_data_list = []
        for r in range(ddp_world_size):
            fname = os.path.join(data_dir, f'train_{max_iters}_rank{r}.bin')
            if os.path.exists(fname):
                d = np.fromfile(fname, dtype=np.uint16).reshape(-1, current_block_size)
                all_data_list.append(d)
                try:
                    os.remove(fname) 
                except:
                    pass
        
        if all_data_list:
            full_new_data = np.concatenate(all_data_list, axis=0)
            
            numbers = re.findall(r'\d+', suffix)
            if numbers:
                current_round_num = int(numbers[-1])
                next_round_num = current_round_num + 1
            else:
                next_round_num = 'next'
            
            final_path = os.path.join(data_dir, f'train_gen_round{next_round_num}.bin')
            full_new_data.tofile(final_path)
            print(f"Successfully generated NEXT ROUND dataset: {final_path}")
            print(f"To run next round, update your script to use: --prev_round_file=train_gen_round{next_round_num}.bin --prev_round_ckpt={os.path.join(out_dir, f'{max_iters}_ckpt.pt')}")
            
    model.train()

# -----------------------------------------------------------------------------
# Logger & Setup
logger = get_logger(os.path.join(out_dir, 'train.log'))
log_file_name = os.path.join(out_dir, 'train.log')

# Model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=meta_data_size, 
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    quiz_size=quiz_size,
    response_size=response_size,
    mask_token_id=mask_token_id
)

# Core logic for choosing the initialization path
# 1. If prev_round_ckpt is provided, prefer it for a warm start
# 2. Otherwise, if resume is requested (usually continuing the same run), use ckpt.pt
# 3. Otherwise, start from scratch
if prev_round_ckpt is not None and os.path.exists(prev_round_ckpt):
    init_from = 'prev_round'
elif init_from == 'resume':
    pass # Keep the existing resume behavior
else:
    init_from = 'scratch'

# Model Creation
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'prev_round':
    print(f"Initializing model from previous round checkpoint: {prev_round_ckpt}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load checkpoint
    checkpoint = torch.load(prev_round_ckpt, map_location=device)
    state_dict = checkpoint['model']
    
    # Fix keys (remove _orig_mod. prefix from compiled models)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # Load weights
    msg = model.load_state_dict(state_dict, strict=False) # strict=False allows small mismatches, such as vocabulary changes
    print(f"Weights loaded. {msg}")
    print("Resetting optimizer and iter_num for new round training.")
    # Intentionally do not load optimizer state or iter_num here because this is a fresh training round

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    # Resume mode should restore iter_num (for simplicity, this script still starts counting from 0 below,
    # a full resume should overwrite the global iter_num; this script mainly demonstrates the prev_round flow)

model.to(device)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Optimizer is ALWAYS fresh for 'prev_round' and 'scratch'
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Estimate loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# LR Scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def open_and_append(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
iter_num = 0
best_val_loss = 1e9

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
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
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))
                # Also save the latest ckpt.pt for resume
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
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
            mfu = raw_model.estimate_mfu(train_batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
        open_and_append(log_file_name, f"iter {iter_num}: loss {lossf:.4f}")
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        print("Training finished. Starting Data Re-ordering for next round...")
        
        optimizer = None
        scaler = None
        torch.cuda.empty_cache()
        
        generate_data(raw_model)
        
        break

if ddp:
    destroy_process_group()
