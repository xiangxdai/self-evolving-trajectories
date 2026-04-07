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
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--gen_batch_size', type=int, default=1024, help='Batch size for generation phase')
parser.add_argument('--compile', type=bool, default=True, help='Use PyTorch 2.0 compilation')


# Iteration and path parameters
parser.add_argument('--suffix', type=str, default='round3_FT_pad_mix', help='Current Round Suffix') 
parser.add_argument('--prev_round_file', type=str, default='train_gen_round3.bin', help='Filename of data from previous round, current mainly data for training') 
parser.add_argument('--prev_round_ckpt', type=str, default='out/MDMx0/cd4$_3_12_384_round2_FT_pad_final_1000000/1000000_ckpt.pt', help='Path to the checkpoint from the previous round to warm-start from')


# Data-mixing parameters
parser.add_argument('--canonical_file', type=str, default='train.bin', help='Original Ground Truth data (Round 0)')
parser.add_argument('--prev_files', type=str, default='train_gen_round2.bin', help='Comma separated list of historical round files (e.g. round2.bin,round3.bin)')
parser.add_argument('--mix_ratios', type=str, default='0.5,0.3,0.2', help='Sampling ratios for [Main, Prev, Canonical]')
parser.add_argument('--out_dir', type=str, default='out/MDMx0', help='Base output directory') 

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
prev_round_ckpt = args.prev_round_ckpt
canonical_file = args.canonical_file
prev_files_str = args.prev_files

# Parse the mixing ratios
mix_ratios = [float(x) for x in args.mix_ratios.split(',')]
assert len(mix_ratios) == 3 and abs(sum(mix_ratios) - 1.0) < 1e-5, "Mix ratios must sum to 1.0"

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

# Block Sizes
# Generated Data: [Quiz + Index + Value] -> 15 + 35*2 = 85
block_size_gen = quiz_size + 2 * response_size
# Canonical Data: [Quiz + Value] -> 15 + 35 = 50 (Need runtime adaptation)
block_size_can = quiz_size + response_size

out_dir = os.path.join(args.out_dir, f'{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}')

# -----------------------------------------------------------------------------
# Config
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

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * block_size_gen
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
# Data Loading (Mixed Sources)
# -----------------------------------------------------------------------------

# 1. Main data (the primary data source for this round)
main_path = os.path.join(data_dir, prev_round_file)
if not os.path.exists(main_path):
    # Fail fast and stop the run immediately
    raise FileNotFoundError(f"CRITICAL ERROR: Main data file not found: {main_path}. Please generate it first!")

print(f"Loading MAIN data: {main_path}")
data_main = np.memmap(main_path, dtype=np.uint16, mode='r')

# 2. Historical data (from earlier rounds)
data_prev_list = []
if prev_files_str:
    for fname in prev_files_str.split(','):
        if not fname.strip(): continue
        fpath = os.path.join(data_dir, fname.strip())
        if os.path.exists(fpath):
            print(f"Loading HISTORICAL data: {fname}")
            data_prev_list.append(np.memmap(fpath, dtype=np.uint16, mode='r'))
        else:
            print(f"Warning: Historical file {fname} not found, skipping.")

# 3. Canonical data (original ground truth)
can_path = os.path.join(data_dir, canonical_file)
print(f"Loading CANONICAL data: {can_path}")
data_canonical = np.memmap(can_path, dtype=np.uint16, mode='r')

# -----------------------------------------------------------------------------
# Helper: Canonical Data Alignment
# -----------------------------------------------------------------------------
# Precompute a fixed index tensor [0, 1, ..., 34]
canonical_indices_template = torch.arange(response_size, dtype=torch.int64, device=device).unsqueeze(0)

def prepare_canonical_batch(ix, batch_size, shuffle=True):
    """
    Convert the original [Quiz, Response] layout (Block=50) into [Quiz, Index, Response] (Block=85)
    
    Args:
        ix: sampled data indices
        batch_size: batch size
        shuffle (bool): 
            True  -> shuffled order (used for training to simulate arbitrary generation order) [(5, F), (0, A), ...]
            False -> sequential order (used for validation/debugging with a fixed order) [(0, A), (1, B), ...]
    """
    # 1. Load Raw Data
    # Assumes data_canonical, block_size_can, and device have already been defined
    z_raw = torch.stack([torch.from_numpy((data_canonical[i:i+block_size_can]).astype(np.int64)) for i in ix])
    z_raw = z_raw.to(device, non_blocking=True) # [B, 50]
    
    # 2. Split
    z_quiz = z_raw[:, :quiz_size]        # [B, 15]
    z_resp_values = z_raw[:, quiz_size:] # [B, 35]
    
    # 3. Create Indices (0..34)
    # Assumes canonical_indices_template has already been defined
    z_indices = canonical_indices_template.repeat(batch_size, 1) # [B, 35]
    
    # 4. Interleave [Index, Value] -> [B, 35, 2]
    # Pair indices and values first; this is required whether or not shuffling is enabled
    z_pairs = torch.stack((z_indices, z_resp_values), dim=2) 
    
    # --- Decide whether to shuffle based on the flag ---
    if shuffle:
        # Generate a random permutation index tensor [B, 35]
        # argsort returns sorted indices, which effectively gives us a random permutation
        rand_perm = torch.rand(batch_size, response_size, device=device).argsort(dim=1)
        
        # Expand dimensions for gather: [B, 35] -> [B, 35, 2]
        # This lets us move index and value together
        rand_perm_expanded = rand_perm.unsqueeze(-1).expand(-1, -1, 2)
        
        # Apply the shuffle
        z_pairs = torch.gather(z_pairs, 1, rand_perm_expanded)
        
    # 5. Flatten -> [B, 70]
    # At this point z_pairs is either ordered or shuffled, but still has the same [Index, Value, Index, Value...] structure
    z_generated = z_pairs.flatten(1, 2)
    
    # 6. Concat -> [B, 85]
    z = torch.cat((z_quiz, z_generated), dim=1)
    return z

# -----------------------------------------------------------------------------
# Mixed Batch Logic
# -----------------------------------------------------------------------------
def get_batch(split):
        
    # 1. Define the attention window size (for example, one quarter of the total length)
    # You can adjust this ratio as needed or replace it with a fixed value such as 5
    window_size = max(1, response_size // 2) 
    is_train = (split == 'train')
    batch_size = train_batch_size if is_train else val_batch_size
    
    # Define the train/validation split ratio (for example, 95% train and 5% validation)
    split_ratio = 0.95 

    # --- 1. Choose the data source ---
    if is_train:
        source_idx = np.random.choice([0, 1, 2], p=mix_ratios)
        if source_idx == 1 and not data_prev_list: source_idx = 0
    else:
        # Validation strategy:
        # Option A: validate only on Main Data (your original logic) to check whether the model has learned the current round
        # Option B: validate only on Canonical Data to check robustness against the original data distribution (recommended)
        # To fix the bug, keep your original Main-data validation logic for now but add proper splitting
        source_idx = 0 

    # --- 2. Prepare the data object and compute split boundaries ---
    if source_idx == 0: # Main
        data = data_main
        block_size = block_size_gen
        total_blocks = len(data) // block_size
    elif source_idx == 1: # Prev
        data = data_prev_list[np.random.randint(len(data_prev_list))]
        block_size = block_size_gen
        total_blocks = len(data) // block_size
    else: # Canonical
        data = data_canonical
        block_size = block_size_can
        total_blocks = len(data) // block_size

    # --- [Key change] Compute sampling start and end block indices from the split ---
    split_idx = int(total_blocks * split_ratio)
    
    if is_train:
        # Training: use the first 95%
        idx_start = 0
        idx_end = split_idx
    else:
        # Validation: use the last 5%
        # Note: if the dataset is too small, idx_end may equal idx_start, so keep a safety fallback
        idx_start = split_idx
        idx_end = total_blocks
        if idx_end <= idx_start: # Fallback for extremely small datasets
            idx_start = 0 
    
    # Ensure there is data available for sampling
    if idx_end <= idx_start:
        raise ValueError(f"Dataset too small for split ratio {split_ratio}")

    # --- 3. Sample blocks ---
    # Randomly sample within the [idx_start, idx_end) range
    ix = torch.randint(idx_start, idx_end, (batch_size,)) * block_size
    
    # --- 4. Load tensors (Canonical data needs special handling) ---
    if source_idx == 2: # Canonical requires special handling
        z = prepare_canonical_batch(ix, batch_size)
    else: # Main / Prev follow the regular path
        z = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        z = z.to(device, non_blocking=True)

    # --- 5. Downstream processing (keep your earlier N-step masking logic) ---
    z_quiz = z[:, :quiz_size]
    z_generated = z[:, quiz_size:] 
    
    z_indices = z_generated[:, ::2]  
    z_values  = z_generated[:, 1::2] 
    
# 1. Restore physical order to build the label and input base sequences
    rel_indices = z_indices - quiz_size 
    rel_indices_safe = rel_indices.clamp(0, response_size - 1)
    sorted_gather_idx = torch.argsort(rel_indices_safe, dim=1)
    y_response = torch.gather(z_values, 1, sorted_gather_idx) 
    
    # =======================================================
    # === [Core fix] Order-agnostic masking logic ===
    # =======================================================
    
    # 1. Decide how many tokens are visible in decoding order (num_visible)
    #    For example, num_visible=2 means we have only seen the first two pairs in the z sequence
    #    For Canonical data, that could mean seeing (5, F) and (0, A)
    num_visible = torch.randint(0, response_size, (batch_size, 1), device=device)
    
    # 2. Build the mask on the decoding-time axis (0..34 vs num_visible)
    #    Note that the comparison here is between range_row and num_visible
    #    should_mask_in_z_order[b, t] = True means the t-th token in Z order should be masked
    range_row = torch.arange(response_size, device=device).unsqueeze(0)
    should_mask_in_z_order = range_row >= num_visible # [B, 35] logical mask

    # 3. Map the decoding-order mask back onto physical sentence positions
    #    Use scatter to write the Z-order mask state back to physical sentence positions
    #    mask_canvas: [B, 35], corresponding to physical sentence positions 0..34
    mask_canvas = torch.zeros((batch_size, response_size), dtype=torch.bool, device=device)
    
    # scatter_ here means:
    # specifically at indices `rel_indices_safe` (physical positions),
    # put the value from `should_mask_in_z_order` (decoding visibility)
    mask_canvas.scatter_(1, rel_indices_safe, should_mask_in_z_order)

    # 4. Build the input sequence x
    x_response = y_response.clone()
    x_response[mask_canvas] = mask_token_id # Fill masked positions with [MASK]

    # 5. Build the label sequence y
    #    Combine this with the N-step logic:
    #    Loss should only be computed for positions that are masked and not too far in the future
    
    #    Rule A: positions that are not masked (already known) become -100
    y_target = y_response.clone()
    y_target = torch.where(mask_canvas, y_target, torch.tensor(-100, device=device))

    #    Rule B: positions too far in the future (optional, if you want to keep the N-step constraint)
    #    Note: N-step is less well-defined under shuffled ordering.
    #    Strict order-agnostic training usually does not need an N-step constraint because it predicts arbitrary missing parts.
    #    In your case, if the goal is to mimic block-wise generation at inference time,
    #    we can simply predict the next N tokens in Z order immediately after num_visible.
    
    #    [Advanced N-step implementation]:
    #    Compute loss only for Z-order indices in [num_visible, num_visible + window_size)
    should_ignore_future_z = range_row >= (num_visible + window_size) # Too far ahead in Z order
    
    #    Scatter this "too far ahead" marker back to physical positions as well
    ignore_future_canvas = torch.zeros((batch_size, response_size), dtype=torch.bool, device=device)
    ignore_future_canvas.scatter_(1, rel_indices_safe, should_ignore_future_z)
    
    #    Apply the N-step filter
    y_target = torch.where(ignore_future_canvas, torch.tensor(-100, device=device), y_target)

    # -------------------------------------------------------

    # 6. Concatenate and finalize (same as before)
    x = torch.cat([z_quiz, x_response], dim=1)
    y = torch.cat([z_quiz, y_target], dim=1)
    y[:, :quiz_size] = -100
    
    # Safety clamp
    x = x.clamp(0, vocab_size - 1)
    y_is_ignored = (y == -100)
    y = y.clamp(0, vocab_size - 1)
    y = torch.where(y_is_ignored, torch.tensor(-100, device=device), y)
    
    return x, y
# ==============================================================================
# GENERATION FUNCTION (Re-Ordering for Next Round)
# ==============================================================================
@torch.no_grad()
def generate_data(model):
    print(f"Starting data generation (Re-Ordering) for NEXT round...")
    # Note: during generation we usually re-order only Main Data because the entire dataset needs to be processed
    model.eval()
    
    num_samples = len(data_main) // block_size_gen
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

        batch_indices = np.arange(start_idx, end_idx) * block_size_gen
        batch_data_raw = [data_main[idx : idx + block_size_gen] for idx in batch_indices]
        z_raw = torch.tensor(np.array(batch_data_raw).astype(np.int64)).to(device)
        
        # Extract and restore the ground-truth values
        quiz_part = z_raw[:, :quiz_size]
        gen_part = z_raw[:, quiz_size:]
        old_indices = gen_part[:, ::2] - quiz_size 
        old_values = gen_part[:, 1::2]
        
        sort_idx = torch.argsort(old_indices, dim=1)
        response_gt = torch.gather(old_values, 1, sort_idx)
        
        z_clean = torch.cat([quiz_part, response_gt], dim=1)
        
        # Re-ordering Logic
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
                d = np.fromfile(fname, dtype=np.uint16).reshape(-1, block_size_gen)
                all_data_list.append(d)
                try: os.remove(fname) 
                except: pass
        
        if all_data_list:
            full_new_data = np.concatenate(all_data_list, axis=0)
            
            numbers = re.findall(r'\d+', suffix)
            current_round_num = int(numbers[-1]) if numbers else 0
            next_round_num = current_round_num + 1
            
            final_path = os.path.join(data_dir, f'train_gen_round{next_round_num}.bin')
            full_new_data.tofile(final_path)
            
            # Automatically suggest the command for the next round
            next_prev_files = f"{prev_round_file},{prev_files_str}".strip(',')
            print(f"\n[SUCCESS] Generated dataset for Round {next_round_num}: {final_path}")
            print("="*60)
            print(f"To run Round {next_round_num}, use:")
            print(f"python train_mdm_iterative_mixed.py \\")
            print(f"  --suffix=round{next_round_num} \\")
            print(f"  --prev_round_file=train_gen_round{next_round_num}.bin \\")
            print(f"  --prev_files={next_prev_files} \\")
            print(f"  --mix_ratios=0.7,0.2,0.1 \\")
            print(f"  --prev_round_ckpt={os.path.join(out_dir, f'{max_iters}_ckpt.pt')}")
            print("="*60)
            
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
    block_size=meta_data_size, # This is 85 in meta
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    quiz_size=quiz_size,
    response_size=response_size,
    mask_token_id=mask_token_id
)

# -----------------------------------------------------------------------------
# Initialization Strategy (Warm Start)
# -----------------------------------------------------------------------------
if prev_round_ckpt is not None and os.path.exists(prev_round_ckpt):
    init_from = 'prev_round'
elif init_from == 'resume':
    pass 
else:
    init_from = 'scratch'

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'prev_round':
    print(f"Initializing model from previous round checkpoint: {prev_round_ckpt}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    checkpoint = torch.load(prev_round_ckpt, map_location=device)
    state_dict = checkpoint['model']
    
    # Remove '_orig_mod.' prefix
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded. Resetting optimizer/iter_num for fine-tuning.")

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Prefer ckpt_latest.pt first, then fall back to the legacy ckpt.pt
    ckpt_path = os.path.join(out_dir, 'ckpt_latest.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(out_dir, 'ckpt.pt') # Backward-compatible fallback
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)

model.to(device)
#scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
# Remove the 'cuda' argument and use torch.cuda.amp instead
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer (Reset unless resuming exact run)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

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
iter_num = 0 if init_from != 'resume' else checkpoint['iter_num']
best_val_loss = 1e9 if init_from != 'resume' else checkpoint['best_val_loss']

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Even if a separate best checkpoint is not saved, keep this variable updated for logging
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            
        # --- Revised save logic: keep only Latest + Final ---
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        # 1. Always overwrite 'ckpt_latest.pt' for resume support
        print(f"saving latest checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))
        
        # # 2. Optionally save an extra 'ckpt_final.pt' on the last step for archiving
        # if iter_num >= max_iters:
        #     print(f"saving final checkpoint to {out_dir}")
        #     torch.save(checkpoint, os.path.join(out_dir, 'ckpt_final.pt'))

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

    # === Revised save logic when leaving the training loop ===
    if iter_num >= max_iters:
        if master_process:
            # 1. Build the final checkpoint
            final_checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': max_iters, # Force this field to max_iters so it stays aligned
                'best_val_loss': best_val_loss,
                'config': config,
            }
            
            # 2. Key point: save as "{max_iters}_ckpt.pt" to match the Bash script expectation
            # For example: out/.../500_ckpt.pt
            final_ckpt_name = f'{max_iters}_ckpt.pt'
            final_path = os.path.join(out_dir, final_ckpt_name)
            
            print(f"Training finished. Saving FINAL checkpoint: {final_path}")
            torch.save(final_checkpoint, final_path)
            
        # ------------------------------------------------
        
        print("Starting Data Re-ordering for next round...")
        
        # Clear memory
        optimizer = None
        scaler = None
        torch.cuda.empty_cache()
        
        generate_data(raw_model)
        break

if ddp:
    destroy_process_group()
