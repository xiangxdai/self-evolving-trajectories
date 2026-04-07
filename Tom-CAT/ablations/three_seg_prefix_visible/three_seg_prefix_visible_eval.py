"""
Evaluation entrypoint for the 3Seg-PrefixVisible ablation.

This ablation keeps Tom-CAT's three-segment packed input at inference time, but
the model uses the standard causal mask instead of Tom-CAT's teacherless
write-space masking.
"""

import os
import torch
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from three_seg_prefix_visible_model import GPTConfig, GPT

parser = argparse.ArgumentParser(description='Evaluate the 3Seg-PrefixVisible ablation')
parser.add_argument('--ckpt_iter', type=int, default=300000)
parser.add_argument('--config', type=str, default='12_12_768_0-1_1000000')
parser.add_argument('--temperature', type=float, default=0.0001)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cd5$', help='Dataset name (folder under data/)')
parser.add_argument('--test_file', type=str, default='cd5_test.jsonl', help='Test file name under data/')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory for this Tom-CAT ablation (default: out/tomcat_ablations/three_seg_prefix_visible/{dataset}_{config})')
parser.add_argument('--decoding_step', type=int, default=1, help='Number of decoding steps')
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
        'three_seg_prefix_visible',
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
max_quiz_len = meta['max_input_len']
max_response_len = meta['max_output_len']

# Resolve special tokens.
pad_token_id = stoi['<PAD>']
sep_token_id = stoi['<SEP>']
eos_token_id = stoi['<EOS>']
mask_token_id = stoi['<MASK>']

# Load the model checkpoint.
ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=args.device)
model_args = checkpoint['model_args']
# Backfill missing token ids for older checkpoints.
if 'mask_token_id' not in model_args:
    model_args['mask_token_id'] = mask_token_id
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
output_file = os.path.join(out_dir, f'three_seg_prefix_visible_eval_{args.ckpt_iter}_details_{args.decoding_step}_{args.temperature}.txt')
with open(output_file, 'w') as f:
    for idx, row in enumerate(tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")):
        quiz = str(row[1]['input'])
        sol = str(row[1]['output'])
        
        # Build the first segment and append the masked second segment.
        quiz_encoded = [stoi[c] for c in quiz]
        quiz_padded = quiz_encoded + [pad_token_id] * (max_quiz_len - len(quiz_encoded)) + [sep_token_id]
        
        response_masked = [mask_token_id] * response_size
        
        # Evaluation input layout:
        #   [quiz_padded] [fully masked second segment]
        #
        # Iterative decoding then appends the third segment and feeds confident
        # predictions back into the second segment across refinement rounds.
        input_seq = torch.tensor(quiz_padded + response_masked, dtype=torch.long, device=args.device).unsqueeze(0)
        
        # Run iterative refinement on the third segment.
        with torch.no_grad():
            output = model.generate(input_seq, max_new_tokens=response_size, temperature=args.temperature, decoding_step=args.decoding_step, topk=True)
        
        # Extract the generated third segment.
        pred_tokens = output[0][-response_size:]
        
        # Decode the prediction and strip control tokens for exact-match comparison.
        pred = ''.join([itos[i.item()] for i in pred_tokens])
        pred_clean = pred.replace("<PAD>", "").replace("<EOS>", "").replace("<SEP>", "").replace("<MASK>", "")
        
        # Build the padded reference response.
        sol_encoded = [stoi[c] for c in sol]
        sol_padded = sol_encoded + [pad_token_id] * (max_response_len - len(sol_encoded)) + [eos_token_id]
        sol_decoded = ''.join([itos[i] for i in sol_padded])
        sol_clean = sol_decoded.replace("<PAD>", "").replace("<SEP>", "").replace("<EOS>", "").replace("<MASK>", "")

        # Compare cleaned prediction and target strings.
        is_correct = (pred_clean == sol_clean)
        if is_correct:
            correct += 1
        total += 1

        # Write both cleaned and raw forms for debugging.
        f.write(f"Sample {idx}: Predicted '{pred_clean}', Actual '{sol_clean}', Correct: {is_correct}\n")
        
        f.write(f"  Raw Predicted: '{pred}'\n")
        f.write(f"  Raw Actual: '{sol_decoded}'\n\n")

# Save the final summary.
with open(os.path.join(out_dir, f'three_seg_prefix_visible_eval_{args.ckpt_iter}_{args.decoding_step}_{args.temperature}.txt'), 'w') as f:
    f.write(f'Accuracy: {correct/total:.4f} ({correct}/{total})\n')
    f.write(f'Configuration: decoding_step={args.decoding_step}, temperature={args.temperature}\n')
    f.write(f'Model: {args.config}, Checkpoint: {args.ckpt_iter}\n')
    f.write(f'Quiz size: {quiz_size} (max_len: {max_quiz_len}), Response size: {response_size} (max_len: {max_response_len})\n')

print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")
print(f"Results saved to {output_file}")
