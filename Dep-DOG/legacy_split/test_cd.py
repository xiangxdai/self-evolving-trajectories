import os
import sys
import torch
import pandas as pd
import argparse
import pickle
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DTO_DIR = SCRIPT_DIR.parent
if str(DTO_DIR) not in sys.path:
    sys.path.insert(0, str(DTO_DIR))

from depdog_model import GPTConfig, GPT
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test cd model')
parser.add_argument('--ckpt_iter', type=int, default=200000)
parser.add_argument('--config', type=str, default='3_12_384_round1_2000000')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_time_step', type=int, default=20, help='Number of decoding steps') 
parser.add_argument('--dataset', type=str, default='cd5$', help='Dataset name (folder under data/)')
parser.add_argument('--test_file', type=str, default='cd5_test.jsonl', help='Test file name under data/')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: out/MDMx0/{dataset}Q_{config})')
args = parser.parse_args()

# ==========================================
# Core change: resolve paths dynamically to avoid FileNotFoundError
# ==========================================
# 1. Get the absolute directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Locate the data root directory (the parent of MDM)
data_root = os.path.join(script_dir, '..', 'data')
print(f"Data root located at: {os.path.abspath(data_root)}")

# 3. Build the default output directory to match the Bash OUT_DIR layout
if args.out_dir is None:
    safe_dataset_name = args.dataset.replace('/', '_')
    # Build the same base output path used by the Bash scripts: out/cd/cd3/k0NoShuffle
    base_out = os.path.join(script_dir, 'out', f'{args.dataset}NoShuffle') 
    out_dir = os.path.join(base_out, f'{safe_dataset_name}_{args.config}')
else:
    out_dir = args.out_dir

# Ensure the output directory exists
os.makedirs(out_dir, exist_ok=True)
# ==========================================


# Load metadata using data_root
meta_path = os.path.join(data_root, args.dataset, 'meta.pkl')
print(f"Loading meta from: {meta_path}")

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Meta file not found at {meta_path}. Check your dataset name and directory structure.")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
data_size = meta['data_size']  # Use data_size instead of block_size
quiz_size = meta['quiz_size']
response_size = meta['response_size']
max_quiz_len = quiz_size-1  # Maximum raw quiz length, excluding the trailing <SEP>
max_response_len = response_size-1  # Maximum raw response length, excluding the trailing <EOS>
vocab_size = meta['vocab_size']

# Get special-token IDs
pad_token_id = stoi['<PAD>']
sep_token_id = stoi['<SEP>']
eos_token_id = stoi['<EOS>']
mask_token_id = stoi['<MASK>']

assert data_size == quiz_size + response_size, f"Expected data_size={quiz_size + response_size}, got {data_size}"

# Load the model
ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
print(f"Loading checkpoint from: {ckpt_path}")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Check --out_dir or --config.")

checkpoint = torch.load(ckpt_path, map_location=args.device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# Handle the _orig_mod. prefix
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

# Load test data using data_root
test_file_path = os.path.join(data_root, args.test_file)
print(f"Loading test data from: {test_file_path}")

test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line.strip()))

correct = 0
total = 0

# Open the output file for test results
output_file = os.path.join(out_dir, f'pred_test_{args.ckpt_iter}_{args.max_time_step}_details.txt')
with open(output_file, 'w') as f:
    for idx, sample in enumerate(tqdm(test_data, total=len(test_data), desc="Testing")):
        quiz = sample['input']
        sol = sample['output']
        # Validate input lengths
        if len(quiz) > quiz_size -1  or len(sol) > response_size - 1:
            f.write(f"Sample {idx}: Skipped due to excessive length (quiz={len(quiz)}, sol={len(sol)})\n")
            continue
        if not all(c in stoi for c in quiz + sol):
            f.write(f"Sample {idx}: Skipped due to invalid characters\n")
            continue
        try:
            # Encode the quiz, pad it, and append <SEP>
            quiz_tokens = [stoi[c] for c in quiz]
            quiz_padded = quiz_tokens + [stoi["<PAD>"]] * (quiz_size - 1 - len(quiz)) + [sep_token_id]
            
            # Initialize the answer entirely with <MASK> tokens
            solution_tokens = [mask_token_id] * (response_size ) #all masked, include + [eos_token_id] 

            input_seq = torch.tensor(quiz_padded + solution_tokens, dtype=torch.long, device=args.device).unsqueeze(0)
            assert input_seq.shape == (1, data_size), f"Expected input shape (1, {data_size}), got {input_seq.shape}"
            assert input_seq.max().item() < vocab_size, f"Input_seq contains invalid tokens: {input_seq.max().item()}"

            # Generate the answer
            with torch.no_grad():
                output = model.generate(input_seq, max_time_step=args.max_time_step, temperature=args.temperature)
            pred_tokens = output[0, -response_size:]  # Take the last response_size positions
            assert pred_tokens.max().item() < vocab_size, f"Generated tokens contain invalid values: {pred_tokens.max().item()}"

            # Decode the prediction
            pred = ''.join([itos[i.item()] for i in pred_tokens])
            pred_clean = pred.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")   # Remove padding and special tokens before comparison
            # Decode and pad the ground-truth answer
            sol_tokens = [stoi[c] for c in sol]
            sol_padded = sol_tokens + [stoi["<PAD>"]] * (response_size -1- len(sol)) + [eos_token_id]
            sol_decoded = ''.join([itos[i] for i in sol_padded])
            sol_clean = sol_decoded.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")  # Optionally remove <PAD> for display

            # Compare the prediction and target after cleaning
            is_correct = (pred_clean == sol_clean)
            if is_correct:
                correct += 1
            total += 1

            # Write the cleaned result to the file
            f.write(f"Sample {idx}: Predicted {pred_clean}, Actual {sol_clean}, Correct: {is_correct}\n")

            # Optionally also log the raw version with special tokens
            f.write(f"  Raw Predicted: '{pred}'\n")
            f.write(f"  Raw Actual: '{sol_decoded}'\n\n")
        except Exception as e:
            f.write(f"Sample {idx}: Error during processing: {str(e)}\n")
            continue

# Save the aggregate result
result_file = os.path.join(out_dir, f'pred_test_{args.ckpt_iter}_{args.max_time_step}.txt')
with open(result_file, 'w') as f:
    f.write(f'Accuracy: {correct/total:.4f} ({correct}/{total})\n')
    f.write(f'Model: {args.config}, Checkpoint: {args.ckpt_iter}\n')
    f.write(f'Quiz size: {quiz_size}, Response size: {response_size}\n')

print(f"Accuracy: {correct/total:.4f}")
print(f"Details saved to {output_file}")
print(f"Summary saved to {result_file}")
