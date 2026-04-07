"""
Unified standard autoregressive baseline evaluation entrypoint.

Supports checkpoint loading across reasoning-task datasets and legacy output
layouts. Builds padded quiz prefixes for strict left-to-right generation and
compares cleaned predictions against padded reference responses.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ar_model import GPT, GPTConfig

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "out"


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate the standard AR baseline on a reasoning task")
    parser.add_argument("--ckpt_iter", type=int, default=300000)
    parser.add_argument("--config", type=str, default="24_16_1024_0-1_1000000")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for stochastic generation")
    parser.add_argument("--dataset", type=str, default="cd5$", help="Dataset name (folder under data/)")
    parser.add_argument("--test_file", type=str, default=None, help="Optional test file path or name")
    parser.add_argument("--meta_file", type=str, default=None, help="Optional metadata file path or name")
    parser.add_argument("--out_dir", type=str, default=None, help="Checkpoint/output directory; if omitted, AR will try to infer it")
    parser.add_argument("--save_suffix", type=str, default="", help="Optional suffix appended to output filenames")
    return parser


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

    # Keep dataset-family defaults in one place so the rest of the script can
    # stay task-agnostic. The main distinction here is which metadata and test
    # file names should be searched first.
    profile = {
        "task_family": "generic",
        "meta_candidates": generic_meta,
        "test_candidates": generic_test,
    }

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


def resolve_checkpoint(dataset, config, ckpt_iter, out_dir):
    ckpt_name = f"{ckpt_iter}_ckpt.pt"
    if out_dir is not None:
        out_dir_path = Path(out_dir).expanduser()
        if not out_dir_path.is_absolute():
            out_dir_path = REPO_ROOT / out_dir_path
        ckpt_path = out_dir_path / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return out_dir_path, ckpt_path

    candidate_dirs = [
        OUT_ROOT / "ar_causal_train" / f"{dataset}_{config}",
        OUT_ROOT / "AR" / f"2train{dataset}_{config}",
        OUT_ROOT / f"AR2train{dataset}_{config}",
    ]
    for candidate_dir in candidate_dirs:
        ckpt_path = candidate_dir / ckpt_name
        if ckpt_path.exists():
            return candidate_dir, ckpt_path

    matches = []
    if OUT_ROOT.exists():
        for ckpt_path in OUT_ROOT.rglob(ckpt_name):
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
    return [stoi[character] for character in text]


def decode_tokens(tokens, itos):
    return "".join(itos[token.item()] for token in tokens)


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
    # If response_size reserves one extra slot beyond the maximum response
    # content length, place <EOS> in that final slot; otherwise pad directly to
    # response_size.
    solution_encoded = encode_text(solution, stoi)
    if max_response_len is not None and len(solution_encoded) > max_response_len:
        raise ValueError(f"Response length {len(solution_encoded)} exceeds max_response_len {max_response_len}")

    if eos_token_id is not None and response_size == max_response_len + 1:
        return solution_encoded + [pad_token_id] * (max_response_len - len(solution_encoded)) + [eos_token_id]
    return solution_encoded + [pad_token_id] * (response_size - len(solution_encoded))


def clean_decoded_text(text):
    # Current metric: exact string match after removing selected special-token
    # markers from decoded text. This is a convenience normalization for
    # fixed-length packed outputs rather than a strict token-by-token metric.
    for token in ["<PAD>", "<EOS>", "<SEP>", "<MASK>"]:
        text = text.replace(token, "")
    return text


def main(argv=None, default_overrides=None):
    parser = build_parser()
    if default_overrides:
        parser.set_defaults(**default_overrides)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    profile = infer_dataset_profile(args.dataset)
    meta_path = resolve_meta_path(args.dataset, args.meta_file, profile)
    test_path = resolve_test_path(args.dataset, args.test_file, profile)
    out_dir, ckpt_path = resolve_checkpoint(args.dataset, args.config, args.ckpt_iter, args.out_dir)

    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)

    # -------------------------------------------------------------------------
    # Evaluation protocol
    #
    # Evaluation uses the same character-level vocabulary and fixed-length
    # packing scheme defined by preprocessing and stored in meta.pkl.
    #
    # Required metadata includes:
    #   - stoi / itos
    #   - quiz_size / response_size
    #   - max_quiz_len / max_response_len
    #
    # For each test example:
    #   1) encode the raw input string at character level
    #   2) pad it to quiz_size and terminate with <SEP>
    #   3) feed that packed quiz segment as the autoregressive prefix
    #   4) generate up to response_size tokens
    #   5) decode and compare against the reference output
    #
    # This file evaluates the plain causal AR baseline. Even if the shared
    # vocabulary contains reserved symbols such as <MASK> or $, this path does
    # not use Tom-CAT masked targets or any Ser-FOX serialization.
    # -------------------------------------------------------------------------
    stoi, itos = meta["stoi"], meta["itos"]
    input_key = meta.get("input_key", "input")
    output_key = meta.get("output_key", "output")
    quiz_size = meta["quiz_size"]
    response_size = meta["response_size"]
    max_quiz_len = meta.get("max_input_len", meta.get("max_quiz_len", quiz_size - 1))
    max_response_len = meta.get("max_output_len", meta.get("max_response_len", response_size - 1))

    pad_token_id = stoi["<PAD>"]
    sep_token_id = stoi.get("<SEP>")
    eos_token_id = stoi.get("<EOS>")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Strip the _orig_mod. prefix introduced by torch.compile so compiled
    # checkpoints can load into the plain GPT module.
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_data = pd.read_json(test_path, lines=True)

    correct = 0
    total = 0

    details_path = out_dir / f"pred_test_{args.ckpt_iter}{args.save_suffix}_details.txt"
    with open(details_path, "w") as handle:
        for idx, row in enumerate(tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")):
            quiz = row[1][input_key]
            solution = row[1][output_key]

            # Evaluation input layout:
            #   [quiz_padded]
            #
            # Standard AR generation then appends the response left-to-right, and
            # we compare that final generated segment against the padded target.
            quiz_tokens = build_quiz_tokens(
                quiz,
                stoi,
                quiz_size,
                max_quiz_len,
                sep_token_id,
                pad_token_id,
            )
            input_seq = torch.tensor(quiz_tokens, dtype=torch.long, device=device).unsqueeze(0)

            with torch.no_grad():
                output = model.generate(
                    input_seq,
                    max_new_tokens=response_size,
                    temperature=args.temperature,
                )

            pred_tokens = output[0, -response_size:]
            pred = decode_tokens(pred_tokens, itos)
            pred_clean = clean_decoded_text(pred)

            target_tokens = build_target_tokens(
                solution,
                stoi,
                response_size,
                max_response_len,
                eos_token_id,
                pad_token_id,
            )
            target_decoded = "".join(itos[token] for token in target_tokens)
            target_clean = clean_decoded_text(target_decoded)

            is_correct = pred_clean == target_clean
            if is_correct:
                correct += 1
            total += 1

            handle.write(
                f"Sample {idx}: Predicted '{pred_clean}', Actual '{target_clean}', Correct: {is_correct}\n"
            )
            handle.write(f"  Raw Predicted: '{pred}'\n")
            handle.write(f"  Raw Actual: '{target_decoded}'\n\n")

    accuracy = correct / total if total > 0 else 0.0
    summary_path = out_dir / f"pred_test_{args.ckpt_iter}{args.save_suffix}.txt"
    with open(summary_path, "w") as handle:
        handle.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
        handle.write(f"Model: {args.config}, Checkpoint: {args.ckpt_iter}\n")
        handle.write(f"Temperature: {args.temperature}\n")
        handle.write(f"Seed: {args.seed}\n")
        handle.write(f"Input key: {input_key}, Output key: {output_key}\n")
        handle.write(f"Quiz size: {quiz_size} (max_len: {max_quiz_len}), Response size: {response_size} (max_len: {max_response_len})\n")

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to {details_path}")


if __name__ == "__main__":
    main()
