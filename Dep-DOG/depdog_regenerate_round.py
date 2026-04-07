import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import re
from pathlib import Path
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from depdog_model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Command-line arguments (kept aligned with train_round1.py)
parser = argparse.ArgumentParser(description='Regenerate the next Dep-DOG round dataset from an earlier checkpoint')
parser.add_argument('--dataset', type=str, default='cd4$', help='Name of the dataset to use')
parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=512, help='Size of the embeddings')
parser.add_argument('--max_iters', type=int, default=900000, help='Number of iterations (used to find ckpt)')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (not used for training here)')
parser.add_argument('--gen_batch_size', type=int, default=1024, help='Batch size for generation phase')
parser.add_argument('--suffix', type=str, default='round2_FT_pad_final', help='Suffix to add to output directory')# Read path configuration

# Set default to None so the code can infer the output filename when the user does not provide one
parser.add_argument('--output_file', type=str, default=None, 
                    help='Filename for the generated output data.')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Infer the output filename automatically (revised version)
# -----------------------------------------------------------------------------
if args.output_file is None:
    # Strategy 1: prefer matching the pattern "round" + number (for example, round2_FT -> round3)
    # This usually produces a cleaner generated filename without the FT suffix
    match_round = re.search(r'round(\d+)', args.suffix)
    
    if match_round:
        current_num = int(match_round.group(1))  # Extract 2
        next_num = current_num + 1               # Increment to 3
        # Use a fixed generated filename format to keep the data naming clean
        args.output_file = f'train_gen_round{next_num}.bin'
    
    # Strategy 2: if there is no "round" marker, try matching a trailing number as a fallback
    elif re.search(r'(\d+)$', args.suffix):
        match_end = re.search(r'(\d+)$', args.suffix)
        current_num = int(match_end.group(1))
        next_num = current_num + 1
        # Replace the trailing number
        next_suffix = re.sub(r'\d+$', str(next_num), args.suffix)
        args.output_file = f'train_gen_{next_suffix}.bin'
        
    # Strategy 3: if no number can be found, append _next
    else:
        args.output_file = f'train_gen_{args.suffix}_next.bin'

print(f"Input Suffix: {args.suffix}")
print(f"Output File : {args.output_file}")
# Extract arguments
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
gen_batch_size = args.gen_batch_size
suffix = args.suffix

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
data_dir = os.path.join(str(REPO_ROOT), 'data', dataset)
safe_dataset_name = dataset.replace('/', '_')
out_dir = os.path.join(str(REPO_ROOT), 'out', 'depdog_train', f'{safe_dataset_name}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}')

# Load metadata
meta_file = 'cd_meta.pkl'
with open(os.path.join(data_dir, meta_file), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
data_size = meta['data_size']   # 50
quiz_size = meta['quiz_size']   # 15
response_size = meta['response_size']  # 35
vocab_size = meta['vocab_size']  # 18
mask_token_id = stoi['<MASK>']

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    master_process = True
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loader (read only train.bin for generation)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

# Context setup
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = torch.bfloat16 # Use bfloat16 by default; change to float16 if the hardware does not support it
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# GENERATION FUNCTION (ported from train_round1.py)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_data(model):
    if master_process:
        print(f"Starting data generation (Ordering Phase)...")
        print(f"Strategy: Enforce 'Content First, PAD Last' constraints.") # Log
    model.eval()
    
    # --- Addition 1: fetch the PAD ID ---
    if '<PAD>' not in stoi:
        raise ValueError("Critical Error: '<PAD>' token not found in stoi vocabulary!")
    pad_token_id = stoi['<PAD>']
    # ---------------------------

    num_samples = len(train_data) // data_size
    generated_chunks = []
    
    # Split work across DDP ranks
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
        current_batch_size = end_idx - start_idx
        if current_batch_size <= 0: break

        # 1. Build the batch
        batch_indices = np.arange(start_idx, end_idx) * data_size
        batch_data_raw = []
        for idx in batch_indices:
             batch_data_raw.append(train_data[idx : idx + data_size])
        
        # Here z is the ground-truth sequence, including both Quiz and Response
        z = torch.tensor(np.array(batch_data_raw).astype(np.int64)).to(device)
        
        # 2. Initialize working tensors
        z_order = z[:, :quiz_size].clone()
        z_mask = z.clone()
        z_mask[:, quiz_size:] = mask_token_id
        
        # --- Addition 2: precompute PAD positions in the ground-truth sequence ---
        # Shape: [Batch, data_size]; True where the ground-truth token is PAD
        is_pad_gt = (z == pad_token_id)
        # -----------------------------------------------
        
        # 3. Re-ordering loop
        for step in range(response_size):
            with ctx:
                logits, _ = model(z_mask) 
            
            probs = F.softmax(logits, dim=-1)
            decode = (z_mask == mask_token_id) # Positions that still need to be decoded (the masked portion)
            
            # Gather the probability assigned to the ground-truth token
            p = probs.gather(dim=2, index=z.unsqueeze(-1)).squeeze(-1)
            # Set positions that no longer need decoding to -1.0
            p = torch.where(decode, p, torch.tensor(-1.0, device=device))
            
            # ==========================================================
            # [Key change] Port the Content First logic
            # ==========================================================
            
            # 1. Check whether each sample still has non-PAD content waiting to be decoded
            # Logic: the position is masked (decode=True) and its ground-truth value is not PAD (~is_pad_gt)
            content_pending_mask = decode & (~is_pad_gt)
            
            # For each sample, if any content_pending entry is True, PAD must not be selected yet
            has_content_left = content_pending_mask.any(dim=1, keepdim=True) # [Batch, 1]
            
            # 2. Identify PAD positions that are still pending (the PAD candidate set)
            # Logic: the position is masked (decode=True) and its ground-truth value is PAD (is_pad_gt)
            pad_pending_mask = decode & is_pad_gt
            
            # 3. Build a suppression mask: if content remains, suppress all PAD candidates
            suppress_pads = has_content_left & pad_pending_mask
            
            # 4. Apply suppression by forcing PAD probabilities to a very small value
            p = torch.where(suppress_pads, torch.tensor(-10.0, device=device), p)
            
            # ==========================================================
            # End of Modification
            # ==========================================================
            
            _, max_idx = p.max(dim=1)
            max_token = z.gather(dim=1, index=max_idx.unsqueeze(-1))
            
            scatter_mask = torch.zeros_like(z_mask, dtype=torch.bool)
            scatter_mask.scatter_(1, max_idx.unsqueeze(-1), True)
            z_mask = torch.where(scatter_mask, z, z_mask)
            
            z_order = torch.cat([z_order, max_idx.unsqueeze(-1), max_token], dim=1)

        generated_chunks.append(z_order.detach().cpu().numpy().astype(np.uint16))
        
        if i % 10 == 0:
            print(f"[Rank {ddp_rank}] Processed batch {i}/{len(total_range)}")

    # Save this rank's shard
    if len(generated_chunks) > 0:
        local_data = np.concatenate(generated_chunks, axis=0)
    else:
        local_data = np.array([], dtype=np.uint16)

    output_file = os.path.join(data_dir, f'train_gen_temp_rank{ddp_rank}.bin')
    local_data.tofile(output_file)
    print(f"[Rank {ddp_rank}] Saved chunk to {output_file}")
    
    if ddp:
        torch.distributed.barrier()
        
    # Merge shards
    if master_process:
        print("Master process merging files...")
        all_data_list = []
        for r in range(ddp_world_size):
            fname = os.path.join(data_dir, f'train_gen_temp_rank{r}.bin')
            if os.path.exists(fname):
                # The block size here is computed as Quiz + 2 * Response (Index + Token)
                new_block_size = quiz_size + 2 * response_size
                d = np.fromfile(fname, dtype=np.uint16).reshape(-1, new_block_size)
                all_data_list.append(d)
                # os.remove(fname)  # Optional: delete the temporary file
        
        if all_data_list:
            full_new_data = np.concatenate(all_data_list, axis=0)
            
            final_path = os.path.join(data_dir, args.output_file)
            
            full_new_data.tofile(final_path)
            print(f"Successfully RE-GENERATED dataset: {final_path}")
            print(f"New data shape: {full_new_data.shape}")
        else:
            print("Error: No data generated.")

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------

# 1. Initialize the model structure
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=data_size,
    bias=False,
    vocab_size=vocab_size,
    dropout=0.0, # Dropout is not needed during generation
    quiz_size=quiz_size,
    response_size=response_size,
    mask_token_id=mask_token_id
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# 2. Load the checkpoint
ckpt_path = os.path.join(out_dir, f'{max_iters}_ckpt.pt')
if not os.path.exists(ckpt_path):
    # Try the checkpoint name without an iteration suffix
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

if os.path.exists(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    
    # Strip the _orig_mod prefix
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
else:
    raise FileNotFoundError(f"Cannot find checkpoint at {out_dir}")

model.to(device)

# DDP wrapper needed for generation function logic (if running with torchrun)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 3. Run generation
raw_model = model.module if ddp else model
generate_data(raw_model)

if ddp:
    destroy_process_group()
