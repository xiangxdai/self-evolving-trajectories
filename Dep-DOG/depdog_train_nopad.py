import os
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
from depdog_model import GPTConfig, GPT
from logger import get_logger
import re

# =============================================================================
# Command-line arguments
# =============================================================================
parser = argparse.ArgumentParser(description='Unified training of NanoGPT for Dep-DOG (all rounds, no-pad variant)')


# Core control and path settings
parser.add_argument('--round', type=int, default=1, help='Current training round (1 for Base, 2+ for Iterative)')
parser.add_argument('--dataset', type=str, default='sat/3sat7/k1', help='Name of the dataset path relative to data/')
parser.add_argument('--meta_file', type=str, default='meta.pkl', help='Metadata pickle file')
parser.add_argument('--out_dir', type=str, default=None, help='Base output directory (default: out/depdog_train_nopad)')
parser.add_argument('--suffix', type=str, default='run', help='Current Round Suffix') 

# Model and training parameters
parser.add_argument('--n_layer', type=int, default=3, help='Number of layers')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=384, help='Size of the embeddings')
parser.add_argument('--max_iters', type=int, default=100000, help='Number of iterations')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--gen_batch_size', type=int, default=1024, help='Batch size for generation phase')
parser.add_argument('--compile', type=bool, default=True, help='Use PyTorch 2.0 compilation')

# Iteration and mixing parameters (Round 2+ only)
parser.add_argument('--canonical_file', type=str, default='train.bin', help='Original Ground Truth data (Round 1)')
parser.add_argument('--prev_round_file', type=str, default='train_gen_round2.bin', help='Main data for current round') 
parser.add_argument('--prev_files', type=str, default='', help='Comma separated list of historical round files')
parser.add_argument('--prev_round_ckpt', type=str, default='', help='Path to previous round checkpoint')
parser.add_argument('--mix_ratios', type=str, default='1.0,0.0,0.0', help='Sampling ratios for [Main, Prev, Canonical]')

# Masking and shuffle strategy parameters
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle response pairs in canonical batch')
parser.add_argument('--window_size', type=int, default=1, help='Window size for future masking in Z-order (1 means use response_size)')

args = parser.parse_args()

# Extract arguments
round_num = args.round
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
train_batch_size = args.batch_size
gen_batch_size = args.gen_batch_size
val_batch_size = max(1, args.batch_size // 2)
compile = args.compile
suffix = args.suffix
# Newly extracted arguments
shuffle_canonical = args.shuffle
custom_window_size = args.window_size

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
data_dir = str(REPO_ROOT / 'data' / dataset)

# Replace slashes in the dataset name to avoid accidental nested subdirectories
safe_dataset_name = dataset.replace('/', '_')
if args.out_dir is None:
    base_out_dir = REPO_ROOT / 'out' / 'depdog_train_nopad'
else:
    base_out_dir = Path(args.out_dir).expanduser()
    if not base_out_dir.is_absolute():
        base_out_dir = REPO_ROOT / base_out_dir
out_dir = str(base_out_dir / f'{safe_dataset_name}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}')

# Load metadata
with open(os.path.join(data_dir, args.meta_file), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
quiz_size = meta['quiz_size']      # 15
response_size = meta['response_size'] # 35
vocab_size = meta['vocab_size']    # 18
mask_token_id = stoi['<MASK>'] 

# Block Sizes
# Model input block size is always 50 (Quiz + Response physical tokens)
model_block_size = quiz_size + response_size 
# Disk storage block size
block_size_can = quiz_size + response_size      # 50 (Canonical Data)
block_size_gen = quiz_size + 2 * response_size  # 85 (Re-ordered Data with Indices)

# =============================================================================
# Configuration & DDP Setup
# =============================================================================
eval_interval = max(1, max_iters // 10)
log_interval = max(1, max_iters // 100)
eval_iters = max(1, max_iters // 1000)
eval_only = False
always_save_checkpoint = True

gradient_accumulation_steps = 1
dropout = 0.1
bias = False


# ================= 1. DDP initialization and environment setup =================
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    backend = 'nccl'
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

if master_process: 
    os.makedirs(out_dir, exist_ok=True)

if ddp:
    torch.distributed.barrier() # Let non-master ranks wait here until the master process creates the output directory

# ================= 2. Hyperparameter calculation that depends on environment variables =================
# Auto LR Scaling Foundation
base_batch_size = 512
initial_base_lr = 3e-4  # Original base learning rate

global_batch_size = train_batch_size * ddp_world_size * gradient_accumulation_steps
lr_scaling_factor = global_batch_size / base_batch_size

# --- Core change: adjust LR and warmup dynamically by round ---
if round_num == 1:
    # Round 1: cold start with the full LR and standard warmup
    learning_rate = initial_base_lr * lr_scaling_factor
    warmup_iters = max_iters // 20  # 5% warmup
else:
    # Round 2+: fine-tuning / continuous-training stage
    # Strategy 1: exponentially decay the peak LR by round (for example, 70% of the previous round)
    # decay_rate = 0.7 ** (round_num - 1)
    
    # Strategy 2: use one smaller fine-tuning LR for all later rounds (recommended and more stable)
    # Because the mix_ratios for R3-R10 are fixed, their task difficulty is fairly similar
    decay_rate = 0.1  # Reduce directly to 1/10 as the fine-tuning LR
    
    learning_rate = (initial_base_lr * decay_rate) * lr_scaling_factor
    warmup_iters = max_iters // 100 # Greatly shorten warmup or effectively remove it because pretrained weights already exist

weight_decay = 1e-1
beta1 = 0.9; beta2 = 0.95; grad_clip = 1.0
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10
dtype = 'bfloat16'

# ================= 3. Save configuration state =================
# At this point all global variables, including ddp_world_size and the computed learning_rate, are ready
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# =============================================================================
# Data Loading & Mixing
# =============================================================================
# Canonical Data (Always loaded, needed for generate_data and Round 1 training)
can_path = os.path.join(data_dir, args.canonical_file)
data_canonical = np.memmap(can_path, dtype=np.uint16, mode='r')

if round_num > 1:
    mix_ratios = [float(x) for x in args.mix_ratios.split(',')]
    assert abs(sum(mix_ratios) - 1.0) < 1e-5, "Mix ratios must sum to 1.0"
    
    main_path = os.path.join(data_dir, args.prev_round_file)
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Main data {main_path} not found!")
    data_main = np.memmap(main_path, dtype=np.uint16, mode='r')

    data_prev_list = []
    if args.prev_files:
        for fname in args.prev_files.split(','):
            if not fname.strip(): continue
            fpath = os.path.join(data_dir, fname.strip())
            if os.path.exists(fpath):
                data_prev_list.append(np.memmap(fpath, dtype=np.uint16, mode='r'))

canonical_indices_template = torch.arange(response_size, dtype=torch.int64, device=device).unsqueeze(0)

def prepare_canonical_batch(ix, batch_size, shuffle=shuffle_canonical):
    z_raw = torch.stack([torch.from_numpy((data_canonical[i:i+block_size_can]).astype(np.int64)) for i in ix])
    z_raw = z_raw.to(device, non_blocking=True)
    z_quiz = z_raw[:, :quiz_size]
    z_resp_values = z_raw[:, quiz_size:]
    z_indices = canonical_indices_template.repeat(batch_size, 1)
    z_pairs = torch.stack((z_indices, z_resp_values), dim=2)
    if shuffle:
        rand_perm = torch.rand(batch_size, response_size, device=device).argsort(dim=1)
        rand_perm_expanded = rand_perm.unsqueeze(-1).expand(-1, -1, 2)
        z_pairs = torch.gather(z_pairs, 1, rand_perm_expanded)
    z_generated = z_pairs.flatten(1, 2)
    return torch.cat((z_quiz, z_generated), dim=1)

def get_batch(split):
    is_train = (split == 'train')
    batch_size = train_batch_size if is_train else val_batch_size
    split_ratio = 0.95 

    if round_num == 1:
        # ---------------------------------------------------------------------
        # Round 1 Logic: Pure Canonical Data + Random Masking
        # ---------------------------------------------------------------------
        data = data_canonical
        total_blocks = len(data) // block_size_can
        split_idx = int(total_blocks * split_ratio)
        idx_start, idx_end = (0, split_idx) if is_train else (split_idx, total_blocks)
        if idx_end <= idx_start: idx_start = 0 
        
        ix = torch.randint(idx_start, idx_end, (batch_size,)) * block_size_can
        z = torch.stack([torch.from_numpy((data[i:i+block_size_can]).astype(np.int64)) for i in ix])
        z = z.to(device, non_blocking=True)
        
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
        is_masked = (x == mask_token_id)
        y = torch.where(is_masked, y, torch.tensor(-100, device=device))
        
    else:
        # ---------------------------------------------------------------------
        # Round 2+ Logic: Mixed Data + Z-order Curriculum Masking
        # ---------------------------------------------------------------------
        window_size = max(1, response_size //custom_window_size)
        
        if is_train:
            source_idx = np.random.choice([0, 1, 2], p=mix_ratios)
            if source_idx == 1 and not data_prev_list: source_idx = 0
        else:
            source_idx = 0 

        if source_idx == 0:
            data, block_size = data_main, block_size_gen
        elif source_idx == 1:
            data, block_size = data_prev_list[np.random.randint(len(data_prev_list))], block_size_gen
        else:
            data, block_size = data_canonical, block_size_can

        total_blocks = len(data) // block_size
        split_idx = int(total_blocks * split_ratio)
        idx_start, idx_end = (0, split_idx) if is_train else (split_idx, total_blocks)
        if idx_end <= idx_start: idx_start = 0 

        ix = torch.randint(idx_start, idx_end, (batch_size,)) * block_size
        
        if source_idx == 2:
            z = prepare_canonical_batch(ix, batch_size)
        else:
            z = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
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
        range_row = torch.arange(response_size, device=device).unsqueeze(0)
        should_mask_in_z_order = range_row >= num_visible

        mask_canvas = torch.zeros((batch_size, response_size), dtype=torch.bool, device=device)
        mask_canvas.scatter_(1, rel_indices_safe, should_mask_in_z_order)

        x_response = y_response.clone()
        x_response[mask_canvas] = mask_token_id

        y_target = y_response.clone()
        y_target = torch.where(mask_canvas, y_target, torch.tensor(-100, device=device))

        should_ignore_future_z = range_row >= (num_visible + window_size) 
        ignore_future_canvas = torch.zeros((batch_size, response_size), dtype=torch.bool, device=device)
        ignore_future_canvas.scatter_(1, rel_indices_safe, should_ignore_future_z)
        y_target = torch.where(ignore_future_canvas, torch.tensor(-100, device=device), y_target)

        x = torch.cat([z_quiz, x_response], dim=1)
        y = torch.cat([z_quiz, y_target], dim=1)
        y[:, :quiz_size] = -100

    zero_mask_rows = (x[:, quiz_size:] == mask_token_id).sum(dim=1) == 0
    if zero_mask_rows.any().item():
        raise RuntimeError(
            f"Dep-DOG/depdog_train_nopad.py get_batch produced {zero_mask_rows.sum().item()} sample(s) without any masked response tokens; loss normalization would divide by zero."
        )

    # -------------------------------------------------------------------------
    # Unified Safety Clamp (For Both Branches)
    # -------------------------------------------------------------------------
    x = x.clamp(0, vocab_size - 1)
    y_is_ignored = (y == -100)
    y = y.clamp(0, vocab_size - 1)
    y = torch.where(y_is_ignored, torch.tensor(-100, device=device), y)
        
    return x, y

# =============================================================================
# Generation Function (Trajectory/Ordering Search)
# =============================================================================
@torch.no_grad()
def generate_data(model):
    print(f"\n[Generation Phase] Starting Trajectory Search for NEXT round...")
    model.eval()
    
    # --- Fix: choose the data source and block size based on the current round ---
    if round_num == 1:
        # Round 1: use the original physical-order data as the base
        current_data_source = data_canonical
        current_block_size = block_size_can # 50
    else:
        # Round 2+: use the previous round's generated trajectories as the base
        current_data_source = data_main
        current_block_size = block_size_gen # 85
    
    num_samples = len(current_data_source) // current_block_size
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

        # Load the batch data
        batch_indices = np.arange(start_idx, end_idx) * current_block_size
        batch_data_raw = [current_data_source[idx : idx + current_block_size] for idx in batch_indices]
        z_raw = torch.tensor(np.array(batch_data_raw).astype(np.int64)).to(device)
        
        # --- Data reconstruction logic ---
        if round_num == 1:
            # Physical order: split directly
            quiz_part = z_raw[:, :quiz_size]
            response_gt = z_raw[:, quiz_size:]
        else:
            # Trajectory order: restore physical order first
            quiz_part = z_raw[:, :quiz_size]
            gen_part = z_raw[:, quiz_size:]
            old_indices = gen_part[:, ::2] - quiz_size 
            old_values = gen_part[:, 1::2]
            sort_idx = torch.argsort(old_indices, dim=1)
            response_gt = torch.gather(old_values, 1, sort_idx)
        
        # Concatenate into the reference target sequence (open-book)
        z_clean = torch.cat([quiz_part, response_gt], dim=1)
        
        # --- Re-ordering search (Trajectory Search) ---
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

    # Keep the downstream file-save and merge logic unchanged...
    local_data = np.concatenate(generated_chunks, axis=0) if generated_chunks else np.array([], dtype=np.uint16)
    output_file = os.path.join(data_dir, f'train_{max_iters}_rank{ddp_rank}.bin')
    local_data.tofile(output_file)
    
    if ddp: torch.distributed.barrier()
    
    if master_process:
        print("Master process merging files...")
        all_data_list = []
        for r in range(ddp_world_size):
            fname = os.path.join(data_dir, f'train_{max_iters}_rank{r}.bin')
            if os.path.exists(fname):
                d = np.fromfile(fname, dtype=np.uint16).reshape(-1, block_size_gen)
                all_data_list.append(d)
                try: os.remove(fname) 
                except: pass
        
        if all_data_list:
            full_new_data = np.concatenate(all_data_list, axis=0)
            numbers = re.findall(r'\d+', suffix)
            current_round_num = int(numbers[-1]) if numbers else 1
            next_round_num = current_round_num + 1
            final_path = os.path.join(data_dir, f'train_gen_round{next_round_num}.bin')
            full_new_data.tofile(final_path)
            
            # Suggested prev_files string for the next round
            prev_files_str = args.prev_files if args.prev_files else ""
            current_main = args.prev_round_file if round_num > 1 else args.canonical_file
            new_prev_files = f"{current_main},{prev_files_str}".strip(',')

            print(f"\n[SUCCESS] Generated dataset for Round {next_round_num}: {final_path}")
            print(f"To run next round, ensure --prev_round_file={os.path.basename(final_path)}")
            
    if ddp: torch.distributed.barrier()
    model.train()
    
# =============================================================================
# Initialization & Setup
# =============================================================================
log_file_name = os.path.join(out_dir, 'train.log')
if master_process:
    logger = get_logger(log_file_name)
else:
    logger = None

# Note: regardless of the round, the model always receives the 50-token physical sequence
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=model_block_size, # 50
    bias=bias, vocab_size=vocab_size, dropout=dropout,
    quiz_size=quiz_size, response_size=response_size, mask_token_id=mask_token_id
)

if round_num > 1 and args.prev_round_ckpt and os.path.exists(args.prev_round_ckpt):
    print(f"Initializing model from previous round checkpoint: {args.prev_round_ckpt}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    checkpoint = torch.load(args.prev_round_ckpt, map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = 0
    best_val_loss = 1e9
else:
    print("Initializing a new model from scratch (Base Training)")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    iter_num = 0
    best_val_loss = 1e9

model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# =============================================================================
# Training Loop
# =============================================================================
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

def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def open_and_append(filename, text):
    with open(filename, 'a') as file: file.write(text + '\n')

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss: best_val_loss = losses['val']
            
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_args': model_args, 'iter_num': iter_num,
                'best_val_loss': best_val_loss, 'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))

    for micro_step in range(gradient_accumulation_steps):
        if ddp: model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
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
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        if master_process:
            final_checkpoint = {
                'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_args': model_args, 'iter_num': max_iters,
                'best_val_loss': best_val_loss, 'config': config,
            }
            final_path = os.path.join(out_dir, f'{max_iters}_ckpt.pt')
            print(f"Max iters reached. Saving FINAL checkpoint to {final_path}")
            torch.save(final_checkpoint, final_path)

        # Enter the trajectory-generation phase
        optimizer = None; scaler = None; torch.cuda.empty_cache()
        generate_data(raw_model) 
        break

if ddp: destroy_process_group()
