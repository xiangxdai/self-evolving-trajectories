"""
Unified Ser-FOX evaluation entrypoint.

Supports:
- serialized autoregressive decoding
- confidence-guided parallel-index decoding

For Ser-FOX, evaluation should explicitly:
1) generate a serialized sequence in the form [prompt][I_i, y_i]...
2) reorder values by index token
3) strip index scaffolding
4) compare the final cleaned output [prompt][response] against test.bin
"""

from pathlib import Path
import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from serfox_model import GPTConfig, GPT

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent


def build_parser(default_mode=None):
    parser = argparse.ArgumentParser(description="Evaluate Ser-FOX")
    parser.add_argument(
        "--mode",
        type=str,
        default=default_mode or "serialized_ar",
        choices=["serialized_ar", "confidence_guided"],
        help="Decoding mode to evaluate.",
    )
    parser.add_argument("--ckpt_iter", type=int, default=1000000)
    parser.add_argument("--config", type=str, default="3_3_180")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Checkpoint/output directory (default: out/serfox_train/{config}_{seed}, with legacy fallback to out/{config}_{seed}).",
    )
    parser.add_argument("--save_suffix", type=str, default="")
    return parser


def resolve_existing_path(candidates):
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve any of these paths: {candidates}")


def resolve_meta_path(meta_file):
    if meta_file is not None:
        return resolve_existing_path([
            meta_file,
            REPO_ROOT / meta_file,
            REPO_ROOT / "data" / meta_file,
        ])
    return resolve_existing_path([
        REPO_ROOT / "meta.pkl",
        REPO_ROOT / "data" / "meta.pkl",
        "meta.pkl",
    ])


def resolve_test_path(test_file):
    if test_file is not None:
        return resolve_existing_path([
            test_file,
            REPO_ROOT / test_file,
            REPO_ROOT / "data" / test_file,
        ])
    return resolve_existing_path([
        REPO_ROOT / "test.bin",
        REPO_ROOT / "data" / "test.bin",
        "test.bin",
    ])


def resolve_out_dir(out_dir, config, seed):
    if out_dir is not None:
        path = Path(out_dir).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    candidate_dirs = [
        REPO_ROOT / "out" / "serfox_train" / f"{config}_{seed}",
        REPO_ROOT / "out" / f"{config}_{seed}",
    ]
    for candidate in candidate_dirs:
        if (candidate / "0_ckpt.pt").exists() or any(candidate.glob("*_ckpt.pt")):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate.resolve()

    candidate_dirs[0].mkdir(parents=True, exist_ok=True)
    return candidate_dirs[0].resolve()


def strip_compiled_prefix(state_dict):
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    return state_dict


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def build_config_from_checkpoint(checkpoint, meta):
    """
    Reconstruct GPTConfig primarily from checkpoint.model_args, while
    backfilling task-layout fields from meta.pkl for older checkpoints.

    The checkpoint is treated as the source of truth when fields are present.
    meta.pkl only provides backward-compatible defaults.
    """
    checkpoint_model_args = checkpoint["model_args"].copy()

    if "quiz_size" in meta and "response_size" in meta:
        quiz_size = meta["quiz_size"]
        response_size = meta["response_size"]
    else:
        # Backward-compatible fallback for older equal-split datasets.
        quiz_size = meta["block_size"] // 2
        response_size = meta["block_size"] - quiz_size

    value_vocab_size = meta["vocab_size"]

    checkpoint_model_args.setdefault("quiz_size", quiz_size)
    checkpoint_model_args.setdefault("response_size", response_size)
    checkpoint_model_args.setdefault("value_vocab_size", value_vocab_size)

    return GPTConfig(**checkpoint_model_args)


def load_model(ckpt_path, meta, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = build_config_from_checkpoint(checkpoint, meta)

    model = GPT(gptconf)
    state_dict = strip_compiled_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, gptconf


def exact_match(a, b):
    return int(list(a) == list(b))


def deserialize_indexed_response(serialized_tokens, cfg):
    """
    Convert a serialized Ser-FOX output

        [prompt][I_?, y_?][I_?, y_?]...

    into the final clean response

        [y_1, y_2, ..., y_L]

    by:
    1) reading each index token
    2) placing its paired value into the indexed position
    3) discarding all index tokens
    """
    expected_len = cfg.quiz_size + 2 * cfg.response_size
    if len(serialized_tokens) < expected_len:
        raise ValueError(f"Generated sequence too short: {len(serialized_tokens)} < {expected_len}")

    response = [-1] * cfg.response_size
    seen = set()

    for i in range(cfg.response_size):
        idx_tok = int(serialized_tokens[cfg.quiz_size + 2 * i])
        val_tok = int(serialized_tokens[cfg.quiz_size + 2 * i + 1])

        # Index tokens occupy a dedicated id range:
        #   index_token_start + 0, ..., index_token_start + response_size - 1
        # Mapping back to the final response position is therefore:
        #   pos = idx_tok - index_token_start
        pos = idx_tok - cfg.index_token_start

        if not (0 <= pos < cfg.response_size):
            raise ValueError(f"Invalid index token {idx_tok} at pair {i}")

        if pos in seen:
            raise ValueError(f"Duplicate index token {idx_tok} at pair {i}")

        if not (0 <= val_tok < cfg.index_token_start):
            raise ValueError(f"Invalid value token {val_tok} at pair {i}")

        seen.add(pos)
        response[pos] = val_tok

    if any(v == -1 for v in response):
        raise ValueError("Missing index positions in serialized output")

    return response


def reconstruct_final_output(serialized_tokens, cfg):
    """
    Return the final cleaned output in the same layout as test.bin:

        [prompt][response]

    Note that evaluation is performed against this cleaned final layout,
    not against the raw serialized generation:

        [prompt][I_i, y_i]...

    This is the explicit "reorder by index and remove index tokens" step.
    """
    prompt = [int(token) for token in serialized_tokens[:cfg.quiz_size]]
    response = deserialize_indexed_response(serialized_tokens, cfg)
    return prompt + response


def generate_serialized_prediction(model, x, cfg, mode, temperature, top_k=None):
    """
    Generate a serialized Ser-FOX prediction.

    Mode-specific length semantics:
    - serialized_ar: generate 2 * response_size raw tokens because the model
      emits an explicit [index, value] pair autoregressively.
    - confidence_guided: generate response_size steps because each step appends
      one selected [index, value] pair back into the serialized prefix.
    """
    with torch.no_grad():
        if mode == "serialized_ar":
            y = model.generate_serialized_ar(
                x,
                max_new_tokens=cfg.response_size * 2,
                temperature=temperature,
                top_k=top_k,
            )
        elif mode == "confidence_guided":
            y = model.generate_parallel_index(
                x,
                max_new_tokens=cfg.response_size,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    return y[0].tolist()


def evaluate(args):
    torch.manual_seed(args.seed)

    meta_path = resolve_meta_path(args.meta_file)
    test_path = resolve_test_path(args.test_file)
    out_dir = resolve_out_dir(args.out_dir, args.config, args.seed)
    ckpt_path = out_dir / f"{args.ckpt_iter}_ckpt.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    meta = load_meta(meta_path)
    model, gptconf = load_model(ckpt_path, meta, args.device)

    # test.bin stores samples in the final task layout:
    #   [prompt][response]
    #
    # During evaluation, we generate a serialized sequence:
    #   [prompt][I_i, y_i]...
    # then deserialize it back to [prompt][response] before comparison.
    test_arr = np.memmap(test_path, dtype=np.uint16, mode="r")
    if len(test_arr) % gptconf.base_seq_len != 0:
        raise ValueError(
            f"test.bin length {len(test_arr)} is not divisible by base_seq_len {gptconf.base_seq_len}"
        )

    num_samples = len(test_arr) // gptconf.base_seq_len

    suffix = args.save_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    details_path = out_dir / f"serfox_eval_{args.mode}_{args.ckpt_iter}_details_{args.temperature}{suffix}.txt"
    summary_path = out_dir / f"serfox_eval_{args.mode}_{args.ckpt_iter}_summary_{args.temperature}{suffix}.txt"

    correct = 0
    total = 0
    malformed = 0

    with open(details_path, "w") as f:
        for i in tqdm(range(num_samples), desc=f"Evaluating {args.mode}"):
            row = test_arr[i * gptconf.base_seq_len : (i + 1) * gptconf.base_seq_len].astype(np.int64)
            target_final = row.tolist()
            prompt = row[:gptconf.quiz_size].tolist()

            x = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)

            raw_serialized_pred = generate_serialized_prediction(
                model=model,
                x=x,
                cfg=gptconf,
                mode=args.mode,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            format_error = ""
            deserialized_response = None
            final_output_pred = None

            try:
                deserialized_response = deserialize_indexed_response(raw_serialized_pred, gptconf)
                final_output_pred = prompt + deserialized_response
                is_correct = bool(exact_match(final_output_pred, target_final))
            except ValueError as e:
                malformed += 1
                is_correct = False
                format_error = str(e)

            correct += int(is_correct)
            total += 1

            # raw_serialized_pred keeps the internal Ser-FOX scaffold:
            #   [prompt][I_i, y_i]...
            # deserialized_response is the reordered response only:
            #   [y_0, ..., y_{L-1}]
            # final_output_pred is the final task output:
            #   [prompt][response]
            f.write(f"Sample {i}\n")
            f.write(f"  Prompt: {prompt}\n")
            f.write(f"  Raw serialized prediction: {raw_serialized_pred}\n")
            f.write(f"  Deserialized response: {deserialized_response}\n")
            f.write(f"  Final output prediction: {final_output_pred}\n")
            f.write(f"  Target final output: {target_final}\n")
            f.write(f"  Correct: {is_correct}\n")
            if format_error:
                f.write(f"  Format error: {format_error}\n")
            f.write("\n")

    accuracy = correct / total if total > 0 else 0.0
    malformed_rate = malformed / total if total > 0 else 0.0

    with open(summary_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.6f} ({correct}/{total})\n")
        f.write(f"Malformed rate: {malformed_rate:.6f} ({malformed}/{total})\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top-k: {args.top_k}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Meta file: {meta_path}\n")
        f.write(f"Test file: {test_path}\n")
        f.write(f"Quiz size: {gptconf.quiz_size}\n")
        f.write(f"Response size: {gptconf.response_size}\n")
        f.write(f"Base seq len: {gptconf.base_seq_len}\n")
        f.write(f"Index token start: {gptconf.index_token_start}\n")
        f.write(f"Value vocab size: {gptconf.value_vocab_size}\n")

    print(f"Accuracy: {accuracy:.6f} ({correct}/{total})")
    print(f"Malformed rate: {malformed_rate:.6f} ({malformed}/{total})")
    print(f"Details saved to {details_path}")
    print(f"Summary saved to {summary_path}")


def main(default_mode=None):
    parser = build_parser(default_mode=default_mode)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
