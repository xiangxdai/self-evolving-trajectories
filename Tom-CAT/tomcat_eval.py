"""
Unified Tom-CAT evaluation entrypoint.

Supports checkpoint loading across reasoning-task datasets and legacy output
layouts. Builds [quiz][fully_masked_response] inputs for teacherless generation and
compares cleaned predictions against padded reference targets.
"""

import os
from pathlib import Path
import torch
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from tomcat_model import GPTConfig, GPT

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = REPO_ROOT / 'data'
OUT_ROOT = REPO_ROOT / 'out'

# Command-line arguments.
parser = argparse.ArgumentParser(description='Evaluate Tom-CAT on a reasoning task')
parser.add_argument('--ckpt_iter', type=int, default=1000000)
parser.add_argument('--config', type=str, default='12_12_768_0-1_1000000')
parser.add_argument('--temperature', type=float, default=0.0001)
parser.add_argument('--dataset', type=str, default='cd5', help='Dataset name (folder under data/)')
parser.add_argument('--test_file', type=str, default=None, help='Optional test file path or name')
parser.add_argument('--meta_file', type=str, default=None, help='Optional metadata file path or name')
parser.add_argument('--out_dir', type=str, default=None, help='Checkpoint/output directory; if omitted, Tom-CAT will try to infer it')
parser.add_argument('--decoding_step', type=int, default=None, help='Number of decoding refinement steps; defaults depend on dataset profile')
parser.add_argument('--save_suffix', type=str, default='', help='Optional suffix appended to output filenames')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def infer_dataset_profile(dataset):
    # Keep dataset-family defaults in one place so the rest of the script can stay
    # task-agnostic. The main distinction here is which metadata file names to
    # search first and which decoding_step value to use by default.
    dataset_lower = dataset.lower()

    def merge_candidates(*groups):
        merged = []
        seen = set()
        for group in groups:
            for item in group:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
        return merged

    # Generic fallback candidates that work for any dataset folder.
    generic_meta = [
        f"{dataset}_meta.pkl",
        f"{dataset}meta.pkl",
        "meta.pkl",
    ]
    generic_test = [
        f"{dataset}_test.jsonl",
        "test.jsonl",
    ]

    profile = {
        "task_family": "generic",
        "meta_candidates": generic_meta,
        "test_candidates": generic_test,
        "decoding_step": 1,
    }

    # Path / Planning
    if (
        dataset_lower.startswith(("path", "planning"))
        or "_path" in dataset_lower
        or "_planning" in dataset_lower
    ):
        profile["task_family"] = "path"
        profile["meta_candidates"] = merge_candidates(
            ["path_meta.pkl", "planning_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["path_test.jsonl", "planning_test.jsonl"],
            generic_test,
        )
        profile["decoding_step"] = 1

    # Sudoku
    elif dataset_lower.startswith(("sudoku", "sdk")) or "_sudoku" in dataset_lower:
        profile["task_family"] = "sudoku"
        profile["meta_candidates"] = merge_candidates(
            ["sudoku_meta.pkl", "sdk_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["sudoku_test.jsonl", "sdk_test.jsonl"],
            generic_test,
        )
        profile["decoding_step"] = 1

    # SAT / 3SAT / CNF
    elif (
        dataset_lower.startswith(("sat", "3sat", "cnf"))
        or "_sat" in dataset_lower
        or "3sat" in dataset_lower
    ):
        profile["task_family"] = "sat"
        profile["meta_candidates"] = merge_candidates(
            ["sat_meta.pkl", "3sat_meta.pkl", "cnf_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["sat_test.jsonl", "3sat_test.jsonl", "cnf_test.jsonl"],
            generic_test,
        )
        profile["decoding_step"] = 10

    # Countdown
    elif dataset_lower.startswith(("cd", "countdown")) or "_cd" in dataset_lower:
        profile["task_family"] = "countdown"
        profile["meta_candidates"] = merge_candidates(
            ["cd_meta.pkl", "countdown_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["cd_test.jsonl", "countdown_test.jsonl"],
            generic_test,
        )
        profile["decoding_step"] = 24 if dataset_lower.startswith("cd4") else 1

    # Cipher / Anchored Global Dependency
    elif (
        dataset_lower.startswith(("cipher", "agd"))
        or "_cipher" in dataset_lower
        or "anchored" in dataset_lower
    ):
        profile["task_family"] = "cipher"
        profile["meta_candidates"] = merge_candidates(
            ["cipher_meta.pkl", "agd_meta.pkl", "anchored_global_dependency_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["cipher_test.jsonl", "agd_test.jsonl"],
            generic_test,
        )
        profile["decoding_step"] = 1

    return profile

def resolve_path(candidates):
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None

def resolve_meta_path(dataset, meta_file, profile):
    if meta_file is not None:
        meta_path = Path(meta_file).expanduser()
        candidates = [
            meta_path,
            DATA_ROOT / meta_path,
            DATA_ROOT / dataset / meta_path,
        ]
        meta_path = resolve_path(candidates)
        if meta_path is None:
            raise FileNotFoundError(f"Could not find meta file from candidates: {candidates}")
        return meta_path

    candidates = [DATA_ROOT / dataset / name for name in profile['meta_candidates']]
    meta_path = resolve_path(candidates)
    if meta_path is None:
        raise FileNotFoundError(f"Could not infer meta file for dataset '{dataset}'. Tried: {candidates}")
    return meta_path

def resolve_test_path(dataset, test_file, profile):
    if test_file is not None:
        test_path = Path(test_file).expanduser()
        candidates = [
            test_path,
            DATA_ROOT / test_path,
            DATA_ROOT / dataset / test_path,
        ]
        test_path = resolve_path(candidates)
        if test_path is None:
            raise FileNotFoundError(f"Could not find test file from candidates: {candidates}")
        return test_path

    candidates = []
    for name in profile['test_candidates']:
        candidates.append(DATA_ROOT / dataset / name)
        candidates.append(DATA_ROOT / name)
    test_path = resolve_path(candidates)
    if test_path is None:
        raise FileNotFoundError(f"Could not infer test file for dataset '{dataset}'. Tried: {candidates}")
    return test_path

def resolve_checkpoint(dataset, config, ckpt_iter, out_dir):
    # Prefer the current output layout, while still recognizing the older
    # TeachlessARThree checkpoint directories.
    ckpt_name = f'{ckpt_iter}_ckpt.pt'
    if out_dir is not None:
        out_dir_path = Path(out_dir).expanduser()
        if not out_dir_path.is_absolute():
            out_dir_path = REPO_ROOT / out_dir_path
        ckpt_path = out_dir_path / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        return out_dir_path, ckpt_path

    candidate_dirs = [
        OUT_ROOT / 'tomcat_write_only_train' / f'{dataset}_{config}',
        OUT_ROOT / 'tomcat_read_write_train' / f'{dataset}_{config}',
        OUT_ROOT / 'TeachlessARThree' / '3Train' / f'{dataset}_{config}',
        OUT_ROOT / 'TeachlessARThree' / '23Train' / f'{dataset}_{config}',
        OUT_ROOT / f'TeachlessARThree2response23Train{dataset}_{config}',
        OUT_ROOT / 'TeachlessARThree' / f'23Train{dataset}_{config}',
        OUT_ROOT / 'TeachlessARThree' / f'3Train{dataset}_{config}',
    ]
    for candidate_dir in candidate_dirs:
        ckpt_path = candidate_dir / ckpt_name
        if ckpt_path.exists():
            return candidate_dir, ckpt_path

    matches = []
    out_root = OUT_ROOT
    if out_root.exists():
        for ckpt_path in out_root.rglob(ckpt_name):
            parent_text = str(ckpt_path.parent)
            if dataset in parent_text and config in parent_text:
                matches.append(ckpt_path)

    if len(matches) == 1:
        return matches[0].parent, matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple checkpoint matches for dataset='{dataset}', config='{config}', ckpt_iter={ckpt_iter}: "
            f"{[str(match) for match in matches]}. Please pass --out_dir explicitly."
        )
    raise FileNotFoundError(
        f"Could not infer checkpoint directory for dataset='{dataset}', config='{config}', ckpt_iter={ckpt_iter}. "
        "Please pass --out_dir explicitly."
    )

def encode_text(text, stoi):
    return [stoi[c] for c in text]

def decode_tokens(tokens, itos):
    return ''.join(itos[i.item()] for i in tokens)

def build_quiz_tokens(quiz, stoi, quiz_size, max_quiz_len, sep_token_id, pad_token_id):
    # If quiz_size reserves one extra slot beyond the maximum quiz content
    # length, place <SEP> in that final slot; otherwise pad directly to quiz_size.
    quiz_encoded = encode_text(quiz, stoi)
    if max_quiz_len is not None and len(quiz_encoded) > max_quiz_len:
        raise ValueError(f"Quiz length {len(quiz_encoded)} exceeds max_quiz_len {max_quiz_len}")

    if sep_token_id is not None and quiz_size == max_quiz_len + 1:
        return quiz_encoded + [pad_token_id] * (max_quiz_len - len(quiz_encoded)) + [sep_token_id]
    return quiz_encoded + [pad_token_id] * (quiz_size - len(quiz_encoded))

def build_target_tokens(solution, stoi, response_size, max_response_len, eos_token_id, pad_token_id):
    # If response_size reserves one extra slot beyond the maximum response content
    # length, place <EOS> in that final slot; otherwise pad directly to response_size.
    solution_encoded = encode_text(solution, stoi)
    if max_response_len is not None and len(solution_encoded) > max_response_len:
        raise ValueError(f"Response length {len(solution_encoded)} exceeds max_response_len {max_response_len}")

    if eos_token_id is not None and response_size == max_response_len + 1:
        return solution_encoded + [pad_token_id] * (max_response_len - len(solution_encoded)) + [eos_token_id]
    return solution_encoded + [pad_token_id] * (response_size - len(solution_encoded))

def clean_decoded_text(text):
    # Strip special tokens before comparing predictions against the reference.
    for token in ["<PAD>", "<EOS>", "<SEP>", "<MASK>"]:
        text = text.replace(token, "")
    return text

# Resolve dataset defaults and concrete file paths.
profile = infer_dataset_profile(args.dataset)
decoding_step = args.decoding_step if args.decoding_step is not None else profile['decoding_step']
meta_path = resolve_meta_path(args.dataset, args.meta_file, profile)
test_path = resolve_test_path(args.dataset, args.test_file, profile)
out_dir, ckpt_path = resolve_checkpoint(args.dataset, args.config, args.ckpt_iter, args.out_dir)

# Load metadata.
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
quiz_size = meta['quiz_size']
response_size = meta['response_size']
max_quiz_len = meta.get('max_input_len', meta.get('max_quiz_len', quiz_size - 1))
max_response_len = meta.get('max_output_len', meta.get('max_response_len', response_size - 1))

# Resolve the special tokens used to build the masked-response evaluation input.
pad_token_id = stoi['<PAD>']
sep_token_id = stoi.get('<SEP>')
eos_token_id = stoi.get('<EOS>')
mask_token_id = stoi['<MASK>']
dollar_token_id = stoi.get('$')

# Load the model checkpoint.
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
# Backfill token ids that may be missing from older checkpoints.
if 'mask_token_id' not in model_args:
    model_args['mask_token_id'] = mask_token_id
if 'dollar_token_id' not in model_args and dollar_token_id is not None:
    model_args['dollar_token_id'] = dollar_token_id
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# Strip the _orig_mod. prefix introduced by torch.compile so compiled
# checkpoints can load into the plain GPT module.
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Load the evaluation dataset and prepare the output file names.
test_data = pd.read_json(test_path, lines=True)

correct = 0
total = 0

suffix = args.save_suffix
summary_name = f'tomcat_eval_{args.dataset}_{args.ckpt_iter}_{decoding_step}_{args.temperature}{suffix}.txt'
details_name = f'tomcat_eval_{args.dataset}_{args.ckpt_iter}_details_{decoding_step}_{args.temperature}{suffix}.txt'
details_path = out_dir / details_name

with open(details_path, 'w') as f:
    for idx, row in enumerate(tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")):
        quiz = str(row[1]['input'])
        sol = str(row[1]['output'])

        # Evaluation input layout:
        #   [quiz_padded] [fully masked response segment]
        #
        # teacherless generation appends the write-space segment, and we compare
        # that final segment against the padded reference response.
        quiz_tokens = build_quiz_tokens(quiz, stoi, quiz_size, max_quiz_len, sep_token_id, pad_token_id)
        response_masked = [mask_token_id] * response_size
        input_seq = torch.tensor(quiz_tokens + response_masked, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            output = model.generate_teacherless(
                input_seq,
                max_new_tokens=response_size,
                temperature=args.temperature,
                decoding_step=decoding_step,
            )

        # generate_teacherless returns the full concatenated sequence, so the
        # final response_size tokens correspond to the write-space prediction.
        pred_tokens = output[0][-response_size:]
        pred = decode_tokens(pred_tokens, itos)
        pred_clean = clean_decoded_text(pred)

        # Build the padded reference response, then compare the cleaned strings.
        sol_padded = build_target_tokens(sol, stoi, response_size, max_response_len, eos_token_id, pad_token_id)
        sol_decoded = ''.join(itos[i] for i in sol_padded)
        sol_clean = clean_decoded_text(sol_decoded)

        is_correct = (pred_clean == sol_clean)
        if is_correct:
            correct += 1
        total += 1

        # Write both the cleaned comparison and the raw tokenized forms for inspection.
        f.write(f"Sample {idx}: Predicted '{pred_clean}', Actual '{sol_clean}', Correct: {is_correct}\n")
        f.write(f"  Raw Predicted: '{pred}'\n")
        f.write(f"  Raw Actual: '{sol_decoded}'\n\n")

# Save the summary statistics alongside the per-sample details.
summary_path = out_dir / summary_name
with open(summary_path, 'w') as f:
    f.write(f'Accuracy: {correct/total:.4f} ({correct}/{total})\n')
    f.write(f'Configuration: decoding_step={decoding_step}, temperature={args.temperature}\n')
    f.write(f'Device: {device}\n')
    f.write(f'Dataset: {args.dataset}\n')
    f.write(f'Meta file: {meta_path}\n')
    f.write(f'Test file: {test_path}\n')
    f.write(f'Model: {args.config}, Checkpoint: {args.ckpt_iter}\n')
    f.write(f'Quiz size: {quiz_size} (max_len: {max_quiz_len}), Response size: {response_size} (max_len: {max_response_len})\n')

print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")
print(f"Details saved to {details_path}")
print(f"Summary saved to {summary_path}")
