import os
import argparse
import pickle
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
parser = argparse.ArgumentParser(description='Inspect generated Dep-DOG binary data (full steps)')
parser.add_argument('--dataset', type=str, default='cd/cd3/k0', help='Dataset name (e.g., cipher17)')
parser.add_argument('--bin_file', type=str, default='train_gen_round2.bin', help='Name of the bin file to read')
parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to display') # Default to 5 samples because the output is quite long
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 1. Load metadata dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', args.dataset)
meta_file = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(meta_file):
    print(f"Error: Meta file not found at {meta_file}")
    exit(1)

with open(meta_file, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']
quiz_size = meta['quiz_size']
response_size = meta['response_size']

print(f"Loaded Meta from: {args.dataset}")
print(f"  - Quiz Size: {quiz_size}")
print(f"  - Response Size: {response_size}")
print(f"  - Vocab Size: {vocab_size}")

# -----------------------------------------------------------------------------
# 2. Compute the block size and load the data
# Dep-DOG layout: [Quiz (N) | Index_1, Value_1, ... Index_M, Value_M]
block_size = quiz_size + (2 * response_size)
print(f"  - Calculated Block Size: {block_size} (Quiz + 2*Response)")

bin_path = os.path.join(data_dir, args.bin_file)
if not os.path.exists(bin_path):
    print(f"Error: Bin file not found at {bin_path}")
    exit(1)

data = np.fromfile(bin_path, dtype=np.uint16)
total_tokens = len(data)
num_rows = total_tokens // block_size

print(f"\nReading file: {args.bin_file}")
print(f"  - Total samples: {num_rows}")

data = data[:num_rows*block_size].reshape(num_rows, block_size)

# -----------------------------------------------------------------------------
# 3. Display each sample in detail
print("\n" + "="*60)
print(f"Displaying first {args.num_samples} samples (ALL STEPS)...")
print("="*60)

for i in range(min(args.num_samples, num_rows)):
    row = data[i]
    
    quiz_part = row[:quiz_size]
    gen_part = row[quiz_size:]
    
    # Even positions are global indices and odd positions are token values
    gen_indices = gen_part[::2]
    gen_values = gen_part[1::2]
    
    # --- Decode the quiz portion ---
    quiz_str = "".join([itos.get(idx, f"<{idx}>") for idx in quiz_part])
    quiz_clean = quiz_str.replace('<PAD>', '_').replace('<SEP>', '|')
    
    print(f"\nSample #{i}:")
    print(f"  [Quiz]: {quiz_clean}")
    
    # --- Print generation steps in detail ---
    print(f"  [Generation Order (Confidence High -> Low)]:")
    
    reconstructed_response = ['<MASK>'] * response_size
    valid_sample = True
    
    for step, (g_idx, g_val) in enumerate(zip(gen_indices, gen_values)):
        # 1. Compute the array-storage index (must start from 0, not 1)
        array_idx = int(g_idx) - quiz_size
        
        val_char = itos.get(g_val, '?')
        
        # Check for out-of-bounds indices
        if 0 <= array_idx < response_size:
            # Store the value using array_idx
            reconstructed_response[array_idx] = val_char
            # 2. Display a human-readable index when printing (add 1 here)
            display_idx = array_idx + 1
            print(f"    Step {step:02d}: Index {g_idx:3d} (Rel {display_idx:2d}) -> Val '{val_char}'")
        else:
            valid_sample = False
            print(f"    Step {step:02d}: [ERROR] Index {g_idx} out of bounds (Rel {rel_idx})")

    # --- Final reconstruction ---
    full_response_str = "".join(reconstructed_response)
    print(f"  [Reconstructed Result]: {full_response_str}")

print("\nDone.")
