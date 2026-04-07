"""
Unified MDM evaluation entrypoint.

This script evaluates the vanilla masked diffusion baseline, supports both the
new `out/mdm_train/...` layout and the historical MDM checkpoint folders, and
normalizes metadata / test-file resolution across dataset families.
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from mdm_model import GPT, GPTConfig

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "out"


def infer_dataset_profile(dataset):
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
        "decoding_step": 20,
    }

    if (
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
    elif dataset_lower.startswith(("cd4",)):
        profile["task_family"] = "countdown"
        profile["meta_candidates"] = merge_candidates(
            ["cd_meta.pkl", "countdown_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["cd4_test.jsonl", "cd_test.jsonl", "countdown_test.jsonl"],
            generic_test,
        )
    elif dataset_lower.startswith(("cd", "countdown")) or "_cd" in dataset_lower:
        profile["task_family"] = "countdown"
        profile["meta_candidates"] = merge_candidates(
            ["cd_meta.pkl", "countdown_meta.pkl"],
            generic_meta,
        )
        profile["test_candidates"] = merge_candidates(
            ["cd5_test.jsonl", "cd_test.jsonl", "countdown_test.jsonl"],
            generic_test,
        )
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
    elif (
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
        resolved = resolve_path(candidates)
        if resolved is None:
            raise FileNotFoundError(f"Could not find meta file from candidates: {candidates}")
        return resolved

    candidates = [DATA_ROOT / dataset / name for name in profile["meta_candidates"]]
    resolved = resolve_path(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not infer meta file for dataset '{dataset}'. Tried: {candidates}")
    return resolved


def resolve_test_path(dataset, test_file, profile):
    if test_file is not None:
        test_path = Path(test_file).expanduser()
        candidates = [
            test_path,
            DATA_ROOT / test_path,
            DATA_ROOT / dataset / test_path,
        ]
        resolved = resolve_path(candidates)
        if resolved is None:
            raise FileNotFoundError(f"Could not find test file from candidates: {candidates}")
        return resolved

    candidates = []
    for name in profile["test_candidates"]:
        candidates.append(DATA_ROOT / dataset / name)
        candidates.append(DATA_ROOT / name)
    resolved = resolve_path(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not infer test file for dataset '{dataset}'. Tried: {candidates}")
    return resolved


def resolve_checkpoint_in_dir(out_dir, ckpt_iter):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None

    if ckpt_iter == -1:
        latest_alias = out_dir / "ckpt_latest.pt"
        if latest_alias.exists():
            return latest_alias

        numbered = []
        for path in out_dir.glob("*_ckpt.pt"):
            stem = path.name[:-8]
            if stem.isdigit():
                numbered.append((int(stem), path))
        if numbered:
            numbered.sort(key=lambda item: item[0])
            return numbered[-1][1]
        return None

    checkpoint_path = out_dir / f"{ckpt_iter}_ckpt.pt"
    return checkpoint_path if checkpoint_path.exists() else None


def resolve_checkpoint(dataset, config, ckpt_iter, out_dir):
    if out_dir is not None:
        out_dir_path = Path(out_dir).expanduser()
        if not out_dir_path.is_absolute():
            out_dir_path = REPO_ROOT / out_dir_path
        checkpoint_path = resolve_checkpoint_in_dir(out_dir_path, ckpt_iter)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found in {out_dir_path} for ckpt_iter={ckpt_iter}")
        return out_dir_path, checkpoint_path

    if config is None:
        raise ValueError("Please pass --config or --out_dir so mdm_eval can locate the checkpoint.")

    safe_dataset_name = dataset.replace("/", "_")
    candidate_dirs = [
        OUT_ROOT / "mdm_train" / f"{dataset}_{config}",
        OUT_ROOT / "MDMx0" / f"{dataset}_{config}",
        SCRIPT_DIR / "out" / f"{dataset}NoShuffle" / f"{safe_dataset_name}_{config}",
    ]

    matches = []
    for candidate_dir in candidate_dirs:
        checkpoint_path = resolve_checkpoint_in_dir(candidate_dir, ckpt_iter)
        if checkpoint_path is not None:
            matches.append((candidate_dir, checkpoint_path))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple checkpoint matches for dataset='{dataset}', config='{config}', ckpt_iter={ckpt_iter}: "
            f"{[str(match[1]) for match in matches]}. Please pass --out_dir explicitly."
        )

    wildcard_name = "*_ckpt.pt" if ckpt_iter == -1 else f"{ckpt_iter}_ckpt.pt"
    fallback_matches = []
    if OUT_ROOT.exists():
        for checkpoint_path in OUT_ROOT.rglob(wildcard_name):
            parent_text = str(checkpoint_path.parent)
            if dataset in parent_text and config in parent_text:
                fallback_matches.append((checkpoint_path.parent, checkpoint_path))

    if len(fallback_matches) == 1:
        return fallback_matches[0]
    if len(fallback_matches) > 1:
        raise FileNotFoundError(
            f"Found multiple checkpoint matches for dataset='{dataset}', config='{config}', ckpt_iter={ckpt_iter}: "
            f"{[str(match[1]) for match in fallback_matches]}. Please pass --out_dir explicitly."
        )

    raise FileNotFoundError(
        f"Could not infer checkpoint directory for dataset='{dataset}', config='{config}', ckpt_iter={ckpt_iter}. "
        "Please pass --out_dir explicitly."
    )


def encode_text(text, stoi):
    return [stoi[c] for c in text]


def token_to_text(token_id, itos):
    if isinstance(itos, dict):
        return itos[token_id]
    return itos[token_id]


def decode_tokens(tokens, itos):
    decoded = []
    for token in tokens:
        token_id = int(token.item()) if hasattr(token, "item") else int(token)
        decoded.append(token_to_text(token_id, itos))
    return "".join(decoded)


def build_quiz_tokens(quiz, stoi, quiz_size, max_quiz_len, sep_token_id, pad_token_id):
    quiz_encoded = encode_text(quiz, stoi)
    if max_quiz_len is not None and len(quiz_encoded) > max_quiz_len:
        raise ValueError(f"Quiz length {len(quiz_encoded)} exceeds max_quiz_len {max_quiz_len}")

    if sep_token_id is not None and quiz_size == max_quiz_len + 1:
        return quiz_encoded + [pad_token_id] * (max_quiz_len - len(quiz_encoded)) + [sep_token_id]
    return quiz_encoded + [pad_token_id] * (quiz_size - len(quiz_encoded))


def build_target_tokens(solution, stoi, response_size, max_response_len, eos_token_id, pad_token_id):
    solution_encoded = encode_text(solution, stoi)
    if max_response_len is not None and len(solution_encoded) > max_response_len:
        raise ValueError(f"Response length {len(solution_encoded)} exceeds max_response_len {max_response_len}")

    if eos_token_id is not None and response_size == max_response_len + 1:
        return solution_encoded + [pad_token_id] * (max_response_len - len(solution_encoded)) + [eos_token_id]
    return solution_encoded + [pad_token_id] * (response_size - len(solution_encoded))


def clean_decoded_text(text):
    for token in ["<PAD>", "<EOS>", "<SEP>", "<MASK>"]:
        text = text.replace(token, "")
    return text


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the vanilla MDM baseline")
    parser.add_argument("--ckpt_iter", type=int, default=200000, help="Checkpoint iteration to load; set to -1 to load the latest checkpoint in the run directory")
    parser.add_argument("--config", type=str, default=None, help="Config suffix used to infer the checkpoint directory when --out_dir is omitted")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for stochastic decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None, help="Device override, for example cuda:0 or cpu")
    parser.add_argument("--max_time_step", type=int, default=None, help="Number of iterative decoding refinement steps")
    parser.add_argument("--dataset", type=str, default="cd5$", help="Dataset name (folder under data/)")
    parser.add_argument("--test_file", type=str, default=None, help="Optional test file path or name")
    parser.add_argument("--meta_file", type=str, default=None, help="Optional metadata file path or name")
    parser.add_argument("--out_dir", type=str, default=None, help="Checkpoint directory; if omitted, mdm_eval will try to infer it")
    parser.add_argument("--save_suffix", type=str, default="", help="Optional suffix appended to output filenames")
    parser.add_argument("--verbose", action="store_true", help="Print iterative decode traces for each sample")
    parser.add_argument("--delay_pad", action="store_true", help="Delay PAD decoding so content tokens are revealed earlier")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = infer_dataset_profile(args.dataset)
    decoding_step = args.max_time_step if args.max_time_step is not None else profile["decoding_step"]

    meta_path = resolve_meta_path(args.dataset, args.meta_file, profile)
    test_path = resolve_test_path(args.dataset, args.test_file, profile)
    out_dir, ckpt_path = resolve_checkpoint(args.dataset, args.config, args.ckpt_iter, args.out_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)

    stoi, itos = meta["stoi"], meta["itos"]
    quiz_size = meta["quiz_size"]
    response_size = meta["response_size"]
    pad_token_id = stoi["<PAD>"]
    sep_token_id = stoi.get("<SEP>")
    eos_token_id = stoi.get("<EOS>")
    mask_token_id = stoi["<MASK>"]
    max_quiz_len = meta.get("max_input_len", meta.get("max_quiz_len", quiz_size - (1 if sep_token_id is not None else 0)))
    max_response_len = meta.get("max_output_len", meta.get("max_response_len", response_size - (1 if eos_token_id is not None else 0)))

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    model_args.setdefault("mask_token_id", mask_token_id)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_data = []
    with open(test_path, "r") as handle:
        for line in handle:
            if line.strip():
                test_data.append(json.loads(line))

    correct = 0
    total = 0

    suffix = args.save_suffix
    details_name = f"mdm_eval_{args.dataset}_{args.ckpt_iter}_details_{decoding_step}_{args.temperature}{suffix}.txt"
    summary_name = f"mdm_eval_{args.dataset}_{args.ckpt_iter}_{decoding_step}_{args.temperature}{suffix}.txt"
    details_path = out_dir / details_name
    summary_path = out_dir / summary_name

    with open(details_path, "w") as handle:
        for idx, sample in enumerate(tqdm(test_data, total=len(test_data), desc="Testing")):
            quiz = str(sample["input"])
            sol = str(sample["output"])

            if not all(char in stoi for char in quiz + sol):
                handle.write(f"Sample {idx}: Skipped due to invalid characters\n")
                continue

            try:
                quiz_tokens = build_quiz_tokens(quiz, stoi, quiz_size, max_quiz_len, sep_token_id, pad_token_id)
                response_masked = [mask_token_id] * response_size
                input_seq = torch.tensor(quiz_tokens + response_masked, dtype=torch.long, device=device).unsqueeze(0)

                if args.verbose:
                    print(f"\n--- Sample {idx} ---")
                    print(f"Input:  {quiz}")
                    print(f"Target: {sol}")
                    itos_arg = itos
                else:
                    itos_arg = None

                current_pad_id = pad_token_id if args.delay_pad else None

                with torch.no_grad():
                    output = model.generate(
                        input_seq.clone(),
                        max_time_step=decoding_step,
                        temperature=args.temperature,
                        itos=itos_arg,
                        pad_token_id=current_pad_id,
                    )

                pred_tokens = output[0, -response_size:]
                pred = decode_tokens(pred_tokens, itos)
                pred_clean = clean_decoded_text(pred)

                sol_padded = build_target_tokens(sol, stoi, response_size, max_response_len, eos_token_id, pad_token_id)
                sol_decoded = "".join(token_to_text(token_id, itos) for token_id in sol_padded)
                sol_clean = clean_decoded_text(sol_decoded)

                is_correct = pred_clean == sol_clean
                if is_correct:
                    correct += 1
                total += 1

                handle.write(f"Sample {idx}: Predicted '{pred_clean}', Actual '{sol_clean}', Correct: {is_correct}\n")
                handle.write(f"  Raw Predicted: '{pred}'\n")
                handle.write(f"  Raw Actual: '{sol_decoded}'\n\n")
            except Exception as exc:
                handle.write(f"Sample {idx}: Error during processing: {exc}\n")

    if total > 0:
        accuracy = correct / total
        accuracy_line = f"Accuracy: {accuracy:.4f} ({correct}/{total})"
    else:
        accuracy = None
        accuracy_line = "Accuracy: N/A (0/0 valid samples)"

    with open(summary_path, "w") as handle:
        handle.write(accuracy_line + "\n")
        handle.write(f"Configuration: decoding_step={decoding_step}, temperature={args.temperature}, delay_pad={args.delay_pad}\n")
        handle.write(f"Device: {device}\n")
        handle.write(f"Dataset: {args.dataset}\n")
        handle.write(f"Meta file: {meta_path}\n")
        handle.write(f"Test file: {test_path}\n")
        handle.write(f"Checkpoint: {ckpt_path}\n")
        handle.write(f"Quiz size: {quiz_size} (max_len: {max_quiz_len}), Response size: {response_size} (max_len: {max_response_len})\n")

    print(accuracy_line)
    print(f"Details saved to {details_path}")
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
