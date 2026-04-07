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
from torch.nn import functional as F  # F is required here
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_gather

SCRIPT_DIR = Path(__file__).resolve().parent
DTO_DIR = SCRIPT_DIR.parent
REPO_ROOT = DTO_DIR.parent
if str(DTO_DIR) not in sys.path:
    sys.path.insert(0, str(DTO_DIR))

from depdog_model import GPTConfig, GPT
from logger import get_logger

# Legacy split-entry training script kept for reference.

# -----------------------------------------------------------------------------
# Command-line arguments
parser = argparse.ArgumentParser(description='Training of NanoGPT for cd with MDM')
parser.add_argument('--dataset', type=str, default='cd4$', help='Name of the dataset to use')
parser.add_argument('--meta_file', type=str, default='cd_meta.pkl', help='Metadata pickle file') # Add this
parser.add_argument('--n_layer', type=int, default=3, help='Number of layers')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=384, help='Size of the embeddings')
parser.add_argument('--max_iters', type=int, default=100000, help='Number of iterations')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--max_time_step', type=int, default=20, help='Number of max time steps for MDM generation')
parser.add_argument('--compile', type=bool, default=True, help='Use PyTorch 2.0 compilation')
parser.add_argument('--suffix', type=str, default='1024', help='Suffix to add to output directory')
parser.add_argument('--gen_batch_size', type=int, default=1024, help='Batch size for generation phase') # Batch size for the generation phase
parser.add_argument('--out_dir', type=str, default='out/MDMx0', help='Base output directory')
args = parser.parse_args()

# Extract arguments
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
train_batch_size = args.batch_size
val_batch_size = args.batch_size // 2
gen_batch_size = args.gen_batch_size # Batch size used for generation
max_time_step = args.max_time_step
compile = args.compile
suffix = args.suffix

# Get the script directory
script_dir = str(SCRIPT_DIR)
# Data directory
data_dir = os.path.join(str(REPO_ROOT), 'data', dataset)

# Load metadata
meta_file = args.meta_file  # Extract here
with open(os.path.join(data_dir, meta_file), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
data_size = meta['data_size']   # 50
quiz_size = meta['quiz_size']   # 15
response_size = meta['response_size']  # 35
vocab_size = meta['vocab_size']  # 18
pad_token_id = stoi['<PAD>']
eos_token_id = stoi['<EOS>']  
sep_token_id = stoi['<SEP>']
mask_token_id = stoi['<MASK>']
dollar_token_id = stoi['$']

out_dir = os.path.join(args.out_dir, f'{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}')

# -----------------------------------------------------------------------------
# Default config values
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 1000
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'cd-mdm'
wandb_run_name = f'gpt-cd-mdm-{suffix}'
gradient_accumulation_steps = 1
dropout = 0.1
bias = False
learning_rate = 3e-4

# -----------------------------------------------------------------------------
# Automatic learning-rate scaling (linear scaling rule)
base_batch_size = 512
base_learning_rate = 3e-4
learning_rate = base_learning_rate * (train_batch_size / base_batch_size)
print(f"Using scaled learning rate = {learning_rate:.6f} for batch size = {train_batch_size}")

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

# Collect configuration for logging
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
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # Single-GPU mode
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0        # Required for the non-DDP path
    ddp_local_rank = 0  # Also set this in case later code relies on it

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * data_size
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
# Data loader
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    max_index = len(data) - data_size
    
    ix = torch.randint(0, (max_index // data_size) + 1, (batch_size,)) * data_size
    z = torch.stack([torch.from_numpy((data[i:i+data_size]).astype(np.int64)) for i in ix])
    
    quiz = z[:, :quiz_size]        
    solution = z[:, quiz_size:]    

    x_solution = solution.clone()
    
    for i in range(batch_size):
        mask_ratio = torch.rand(1).item() 
        num_to_mask = max(1, int(response_size * mask_ratio))
        mask_indices = torch.randperm(response_size)[:num_to_mask]
        x_solution[i, mask_indices] = mask_token_id

    x = torch.cat([quiz, x_solution], dim=1)
    y = torch.cat([quiz, solution], dim=1) 

    zero_mask_rows = (x_solution == mask_token_id).sum(dim=1) == 0
    if zero_mask_rows.any().item():
        raise RuntimeError(
            f"Dep-DOG/legacy_split/train_round1.py get_batch produced {zero_mask_rows.sum().item()} sample(s) without any masked response tokens; loss normalization would divide by zero."
        )

    is_masked = (x == mask_token_id)
    y = torch.where(is_masked, y, torch.tensor(-100))
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    # === Final safety guard ===
    # Ensure x and y remain valid after they are moved onto the GPU
    x = x.clamp(0, vocab_size - 1)
    
    y_is_ignored = (y == -100)
    y = y.clamp(0, vocab_size - 1)
    y = torch.where(y_is_ignored, torch.tensor(-100, device=device), y)
        
    return x, y
# -----------------------------------------------------------------------------
# Logger
logger = get_logger(os.path.join(out_dir, 'train.log'))
log_file_name = os.path.join(out_dir, 'train.log')

# Load vocab size
meta_path = os.path.join(data_dir, meta_file)
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=data_size,  # 50
    bias=bias,
    vocab_size=meta_vocab_size,
    dropout=dropout,
    quiz_size=quiz_size,  # 15
    response_size=response_size,  # 35,
    mask_token_id=mask_token_id
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Prefer ckpt_latest.pt first
    ckpt_path = os.path.join(out_dir, 'ckpt_latest.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(out_dir, 'ckpt.pt') # Backward-compatible fallback
       
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # ... (the remaining loading logic stays the same) ...
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

# GradScaler for mixed precision
# The first argument explicitly specified the 'cuda' device
#scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
# Updated version (compatible with older releases)
# torch.cuda.amp.GradScaler does not need the first 'cuda' argument
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrapper
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Estimate loss
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

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# File append utility
def open_and_append(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

# -----------------------------------------------------------------------------
# GENERATION FUNCTION (ported and adapted core logic)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_data(model):
    print(f"Starting data generation (Ordering Phase) for next round...")
    model.eval()
   
    # Original mentor-script logic: read the real train.bin data and re-order it
    num_samples = len(train_data) // data_size
   
    # Prepare a list to store generated data chunks
    generated_chunks = []
   
    # Split work across DDP ranks
    if ddp:
        my_indices = np.array_split(np.arange(num_samples), ddp_world_size)[ddp_rank]
        start_idx_base = my_indices[0] if len(my_indices) > 0 else 0
        end_idx_base = my_indices[-1] + 1 if len(my_indices) > 0 else 0
        total_range = range(start_idx_base, end_idx_base, gen_batch_size)
    else:
        total_range = range(0, num_samples, gen_batch_size)

    for i, start_idx in enumerate(total_range):
        end_idx = min(start_idx + gen_batch_size, num_samples if not ddp else end_idx_base)
        current_batch_size = end_idx - start_idx
       
        if current_batch_size <= 0: break

        # 1. Build the batch data (including ground truth)
        batch_indices = np.arange(start_idx, end_idx) * data_size
        batch_data_raw = []
        for idx in batch_indices:
             batch_data_raw.append(train_data[idx : idx + data_size])
       
        # z is the full [Quiz, Solution] ground-truth sequence
        z = torch.tensor(np.array(batch_data_raw).astype(np.int64)).to(device)
       
        # 2. Initialize working tensors
        # z_order starts with the quiz portion
        z_order = z[:, :quiz_size].clone()
       
        # z_mask starts with the quiz kept visible and the entire response masked
        z_mask = z.clone()
        z_mask[:, quiz_size:] = mask_token_id
       
        # Core mentor loop: iterate k times, where k is the response length
        # response_size is 35 in your metadata, corresponding to k in the mentor code
        for step in range(response_size):
           
            # Forward pass through the model
            with ctx:
                logits, _ = model(z_mask) # logits shape: (B, T, V)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (B, T, V)
           
            # Key point 1: find which positions are currently masked
            # decode shape: (B, T); True means the position is masked
            decode = (z_mask == mask_token_id)
           
            # Key point 2: gather the model probability of each ground-truth token
            # z.unsqueeze(-1) shape: (B, T, 1)
            # After gather, p has shape (B, T) and stores the ground-truth token probability at each position
            p = probs.gather(dim=2, index=z.unsqueeze(-1)).squeeze(-1)
           
            # Key point 3: consider only masked positions and set non-masked positions to -1 (ignored)
            p = torch.where(decode, p, torch.tensor(-1.0, device=device))
           
            # Key point 4: find the position with the largest probability (max_idx is an index into the full sequence)
            # vals, max_idx shape: (B,)
            _, max_idx = p.max(dim=1)
           
            # Key point 5: fetch the ground-truth token at that position
            # max_token shape: (B, 1)
            max_token = z.gather(dim=1, index=max_idx.unsqueeze(-1))
           
            # Update the mask by filling the most confident position with its ground-truth value
            # Build a scatter mask
            scatter_mask = torch.zeros_like(z_mask, dtype=torch.bool)
            scatter_mask.scatter_(1, max_idx.unsqueeze(-1), True)
           
            # Replace positions marked True in scatter_mask with the real value from z
            z_mask = torch.where(scatter_mask, z, z_mask)
           
            # Record the decoding order:
            # Mentor output format: [Quiz, (Idx_1, Token_1), (Idx_2, Token_2), ...]
            # Note: should this index be relative to the response segment?
            # Your mentor code outputs max_idx as an absolute position together with max_token.
            # We keep that behavior for consistency.
            z_order = torch.cat([z_order, max_idx.unsqueeze(-1), max_token], dim=1)

        # After the loop, z_order contains the re-ordered data
        generated_chunks.append(z_order.detach().cpu().numpy().astype(np.uint16))
       
        if i % 10 == 0:
            print(f"[Rank {ddp_rank}] Processed batch {i}/{len(total_range)}")

    # Merge shards and save the result (same logic as before)
    if len(generated_chunks) > 0:
        local_data = np.concatenate(generated_chunks, axis=0)
    else:
        local_data = np.array([], dtype=np.uint16)

    output_file = os.path.join(data_dir, f'train_{max_iters}_rank{ddp_rank}.bin')
    local_data.tofile(output_file)
    print(f"[Rank {ddp_rank}] Saved ordered data to {output_file}")
   
    if ddp:
        torch.distributed.barrier()
       
    if master_process:
        print("Master process merging ordered files...")
        all_data_list = []
        for r in range(ddp_world_size):
            fname = os.path.join(data_dir, f'train_{max_iters}_rank{r}.bin')
            if os.path.exists(fname):
                # Note: the reshape dimension changes here
                # New length = quiz_size + 2 * response_size
                new_block_size = quiz_size + 2 * response_size
                d = np.fromfile(fname, dtype=np.uint16).reshape(-1, new_block_size)
                all_data_list.append(d)
       
        if all_data_list:
            full_new_data = np.concatenate(all_data_list, axis=0)
            final_path = os.path.join(data_dir, 'train_gen_round2.bin')
            full_new_data.tofile(final_path)
            print(f"Successfully generated ORDERED dataset: {final_path}")
            print(f"New data shape: {full_new_data.shape}. New block size: {new_block_size}")
        else:
            print("No data generated?")
           
    model.train()

# -----------------------------------------------------------------------------
# WandB logging
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
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            # --- Change: during training only ckpt_latest.pt is kept ---
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving latest checkpoint to {out_dir}")
            # Use a fixed filename here instead of f'{iter_num}_ckpt.pt'
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))
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

    # ==========================================
    # When max iterations are reached, trigger generation and exit
    # ==========================================
    # ==========================================
    # When max iterations are reached, trigger generation and exit
    # ==========================================
    if iter_num > max_iters:
        # --- Addition: ensure the final checkpoint is saved before exiting ---
        if master_process:
            final_checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': max_iters, # Force this field to max_iters
                'best_val_loss': best_val_loss,
                'config': config,
            }
            # Key point: save as "{max_iters}_ckpt.pt" so the Bash script can detect it
            final_path = os.path.join(out_dir, f'{max_iters}_ckpt.pt')
            print(f"Max iters reached. Saving FINAL checkpoint to {final_path}")
            torch.save(final_checkpoint, final_path)
            # Leave ckpt_latest.pt untouched here so it remains the penultimate backup as a safety measure
        # ------------------------------------------------

        print("Starting Generation Phase...")
       
        # Release some GPU memory
        optimizer = None
        scaler = None
        torch.cuda.empty_cache()
       
        # Call the generation function
        generate_data(raw_model) 
       
        break

if ddp:
    destroy_process_group()
