import os
from pathlib import Path
import torch
import pandas as pd
import argparse
import pickle
import json
from depdog_model import GPTConfig, GPT
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate Dep-DOG')
parser.add_argument('--ckpt_iter', type=int, default=100000, help='Iteration to load. Set to -1 for ckpt_latest.pt')
parser.add_argument('--config', type=str, default='3_7_224_round10_FT_pad_mix_100000')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_time_step', type=int, default=20, help='Number of decoding steps') 
parser.add_argument('--dataset', type=str, default='cd4k=0', help='Dataset name (folder under data/)')
parser.add_argument('--meta_file', type=str, default='meta.pkl', help='Metadata file name under the dataset directory')
parser.add_argument('--test_file', type=str, default='cd4_test.jsonl', help='Test file name under data/')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: out/depdog_train/{dataset}_{config}, with automatic search under out/ for matching checkpoints)')
parser.add_argument('--save_suffix', type=str, default='', help='Suffix for the output file names')
# ==========================================
# Verbose flag
# When enabled: print Input/Target and pass itos into generate to enable step-by-step logging
# When disabled: show only the progress bar
# ==========================================
parser.add_argument('--verbose', action='store_true', help='Enable detailed logging. If set, passes itos to generate function.')

parser.add_argument('--delay_pad', action='store_true', help='Enable Content First, PAD Last strategy.')
args = parser.parse_args()

# ==========================================
# Path handling
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / 'data'
OUT_ROOT = REPO_ROOT / 'out'
print(f"Data root located at: {DATA_ROOT.resolve()}")

if args.ckpt_iter == -1:
    ckpt_name = 'ckpt_latest.pt'
else:
    ckpt_name = f'{args.ckpt_iter}_ckpt.pt'


def resolve_out_dir():
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        return out_dir

    safe_dataset_name = args.dataset.replace('/', '_')
    candidate_dirs = [
        OUT_ROOT / 'depdog_train' / f'{safe_dataset_name}_{args.config}',
        OUT_ROOT / 'depdog_train_nopad' / f'{safe_dataset_name}_{args.config}',
    ]
    for candidate in candidate_dirs:
        if (candidate / ckpt_name).exists():
            return candidate

    matches = []
    if OUT_ROOT.exists():
        for candidate in OUT_ROOT.rglob(f'{safe_dataset_name}_{args.config}'):
            if candidate.is_dir() and (candidate / ckpt_name).exists():
                matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple matching checkpoint directories for dataset='{args.dataset}', config='{args.config}', ckpt='{ckpt_name}': "
            f"{[str(match) for match in matches]}. Please pass --out_dir explicitly."
        )

    return candidate_dirs[0]


out_dir = resolve_out_dir()

# Load metadata
meta_path = DATA_ROOT / args.dataset / args.meta_file
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Meta file not found at {meta_path}")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
data_size = meta['data_size']
quiz_size = meta['quiz_size']
response_size = meta['response_size']
vocab_size = meta['vocab_size']

pad_token_id = stoi['<PAD>']
sep_token_id = stoi['<SEP>']
eos_token_id = stoi['<EOS>']
mask_token_id = stoi['<MASK>']

ckpt_path = out_dir / ckpt_name
print(f"Loading checkpoint from: {ckpt_path}")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=args.device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

# Load test data
test_file_path = DATA_ROOT / args.test_file
print(f"Loading test data from: {test_file_path}")
test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line.strip()))

correct = 0
total = 0
output_file = out_dir / f'depdog_eval_{args.ckpt_iter}_{args.max_time_step}{args.save_suffix}_details.txt'

print(f"Starting testing... (Verbose: {args.verbose})")

with open(output_file, 'w') as f:
    # Iterate over the test set
    for idx, sample in enumerate(tqdm(test_data, total=len(test_data), desc="Testing")):
        quiz = sample['input']
        sol = sample['output']
        
        # Pre-checks
        if len(quiz) > quiz_size - 1 or len(sol) > response_size - 1:
            f.write(f"Sample {idx}: Skipped (length)\n")
            continue
        if not all(c in stoi for c in quiz + sol):
            f.write(f"Sample {idx}: Skipped (invalid char)\n")
            continue

        try:
            # Prepare the input sequence
            quiz_tokens = [stoi[c] for c in quiz]
            quiz_padded = quiz_tokens + [stoi["<PAD>"]] * (quiz_size - 1 - len(quiz)) + [sep_token_id]
            solution_tokens = [mask_token_id] * (response_size)
            input_seq = torch.tensor(quiz_padded + solution_tokens, dtype=torch.long, device=args.device).unsqueeze(0)

            # ==========================================
            # Key change: control whether itos is passed in
            # ==========================================
            if args.verbose:
                # Mode 1: verbose mode
                # 1. Print the input and target
                print(f"\n--- Sample {idx} ---")
                print(f"Input:  {quiz}")
                print(f"Target: {sol}")
                # 2. Pass the real itos dictionary so generate can print intermediate steps
                itos_arg = itos
            else:
                # Mode 2: silent mode (tqdm only)
                # Pass None so generate stays silent internally
                itos_arg = None

            # Decide whether to pass pad_token_id based on the CLI flag to enable delayed PAD decoding
            current_pad_id = pad_token_id if args.delay_pad else None

            with torch.no_grad():
                output = model.generate(
                    input_seq, 
                    max_time_step=args.max_time_step, 
                    temperature=args.temperature,
                    itos=itos_arg,
                    pad_token_id=current_pad_id  # Core addition
                )
            
            # Post-process the result
            pred_tokens = output[0, -response_size:]
            pred = ''.join([itos[i.item()] for i in pred_tokens])
            pred_clean = pred.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")
            
            sol_tokens = [stoi[c] for c in sol]
            sol_padded = sol_tokens + [stoi["<PAD>"]] * (response_size - 1 - len(sol)) + [eos_token_id]
            sol_decoded = ''.join([itos[i] for i in sol_padded])
            sol_clean = sol_decoded.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")

            is_correct = (pred_clean == sol_clean)
            if is_correct:
                correct += 1
            total += 1

            # Write the detailed log file regardless of whether verbose mode is enabled
            f.write(f"Sample {idx}: Predicted {pred_clean}, Actual {sol_clean}, Correct: {is_correct}\n")
            f.write(f"  Raw Predicted: '{pred}'\n")
            f.write(f"  Raw Actual: '{sol_decoded}'\n\n")

        except Exception as e:
            f.write(f"Sample {idx}: Error: {str(e)}\n")
            continue

# Save final summary statistics
result_file = out_dir / f'depdog_eval_{args.ckpt_iter}_{args.max_time_step}{args.save_suffix}.txt'
with open(result_file, 'w') as f:
    f.write(f'Accuracy: {correct/total:.4f} ({correct}/{total})\n')
    f.write(f'Model: {args.config}, Checkpoint: {args.ckpt_iter}\n')

print(f"\nAccuracy: {correct/total:.4f}")
print(f"Details saved to {output_file}")
