"""
Evaluation entrypoint for the 2Seg-Teacherless ablation.

Evaluation uses the same packed quiz prefix as the standard AR baseline, while
the model itself enforces teacherless behavior by replacing response-side
content with placeholder tokens inside the forward pass.
"""

import os
import torch
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from two_seg_teacherless_model import GPTConfig, GPT

parser = argparse.ArgumentParser(description='Evaluate the 2Seg-Teacherless ablation')
parser.add_argument('--ckpt_iter', type=int, default=500000)
parser.add_argument('--config', type=str, default='12_12_768_1000000')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cd5$', help='Dataset name (folder under data/)')
parser.add_argument('--test_file', type=str, default='cd5_test.jsonl', help='Test file name under data/')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: out/tomcat_ablations/two_seg_teacherless/{dataset}_{config})')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

def resolve_out_dir():
    if args.out_dir is not None:
        return args.out_dir

    return os.path.join(
        repo_root,
        'out',
        'tomcat_ablations',
        'two_seg_teacherless',
        f'{args.dataset}_{args.config}',
    )


out_dir = resolve_out_dir()

# Load dataset metadata.
with open(os.path.join(repo_root, 'data', args.dataset, 'cd_meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
meta_vocab_size = meta['vocab_size']
quiz_size = meta['quiz_size']
response_size = meta['response_size']

# Resolve special tokens.
pad_token_id = stoi['<PAD>']
sep_token_id = stoi['<SEP>']
eos_token_id = stoi['<EOS>']
mask_token_id = stoi['<MASK>']

# Load the model checkpoint.
ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=args.device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# Strip the torch.compile prefix if present.
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

# Load evaluation data.
test_path = os.path.join(repo_root, 'data', args.test_file)
test_data = pd.read_json(test_path, lines=True)

correct = 0
total = 0

# Save detailed evaluation results.
output_file = os.path.join(out_dir, f'two_seg_teacherless_eval_{args.ckpt_iter}_details.txt')
with open(output_file, 'w') as f:
    for idx, row in enumerate(tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")):
        quiz = str(row[1]['input'])
        sol = str(row[1]['output'])

        # Build the fixed-length quiz prefix used for autoregressive decoding.
        quiz_tokens = [stoi[c] for c in quiz]
        quiz_padded = quiz_tokens + [stoi["<PAD>"]] * (quiz_size - len(quiz) - 1) + [sep_token_id]
        input_seq = torch.tensor(quiz_padded, dtype=torch.long, device=args.device).unsqueeze(0)

        # Decode the response left-to-right.
        with torch.no_grad():
            output = model.generate(input_seq, max_new_tokens=response_size, temperature=args.temperature)
        pred_tokens = output[0][-response_size:]

        # Decode the prediction and strip padding/control tokens for exact-match comparison.
        pred = ''.join([itos[i.item()] for i in pred_tokens])
        pred_clean = pred.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")

        # Build the padded reference response with the same fixed-length protocol.
        sol_tokens = [stoi[c] for c in sol]
        sol_padded = sol_tokens + [stoi["<PAD>"]] * (response_size - len(sol))
        sol_decoded = ''.join([itos[i] for i in sol_padded])
        sol_clean = sol_decoded.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "")

        # Compare cleaned prediction and target strings.
        is_correct = (pred_clean == sol_clean)
        if is_correct:
            correct += 1
        total += 1

        # Write both cleaned and raw forms for debugging.
        f.write(f"Sample {idx}: Predicted {pred_clean}, Actual {sol_clean}, Correct: {is_correct}\n")
        f.write(f"  Raw Predicted: '{pred}'\n")
        f.write(f"  Raw Actual: '{sol_decoded}'\n\n")

# Save the final summary.
with open(os.path.join(out_dir, f'two_seg_teacherless_eval_{args.ckpt_iter}.txt'), 'w') as f:
    f.write(f'Accuracy: {correct/total:.4f} ({correct}/{total})\n')
    f.write(f'Model: {args.config}, Checkpoint: {args.ckpt_iter}\n')
    f.write(f'Quiz size: {quiz_size}, Response size: {response_size}\n')

print(f"Accuracy: {correct/total:.4f}")
print(f"Results saved to {output_file}")
