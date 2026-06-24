"""
Unified Ser-FOX evaluation entrypoint.

Two decoding interfaces only:

- ``--mode serialized_ar``     : serialized autoregressive decoding (AR).
                                 ``--norepeat true`` (DEFAULT, project standard
                                 口径) runs constrained no-repeat decoding that
                                 masks index choices so an index position can
                                 never be emitted twice. ``--norepeat false``
                                 falls back to the plain ``generate_serialized_ar``
                                 path (the model may pick the same index twice).
- ``--mode confidence_guided`` : confidence-guided parallel-index decoding (PI),
                                 unchanged — calls ``model.generate_parallel_index``.

Ser-FOX evaluation works on a serialized sequence:

    1) generate [prompt][I_i, y_i]... autoregressively (or via PI)
    2) reorder values by their index token
    3) strip the index scaffolding
    4) compare the cleaned [prompt][response] against the test target

Test files may be ``.bin`` (flat uint16 memmap, [prompt][response] layout) or
``.jsonl`` / ``.json`` (encoded against meta.pkl on the fly). A single ``--ckpt``
or a whole ``--ckpt_dir`` (sweep) can be evaluated; results go to a CSV at
``--out`` plus a per-checkpoint console line. Field index 2 of each CSV row is
the accuracy (kept stable for downstream parsers such as eval_fox_2round_ar.py).
"""

import argparse
import csv
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from serfox_model import GPT, GPTConfig


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return {
        "ddp": ddp,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def teardown_dist(info):
    if info["ddp"]:
        dist.destroy_process_group()


def log(msg, info):
    if info["rank"] == 0:
        print(msg, flush=True)


def checkpoint_iter(path):
    name = Path(path).name
    try:
        return int(name.split("_ckpt.pt")[0])
    except ValueError:
        return -1


def list_checkpoints(args):
    if args.ckpt:
        return [Path(args.ckpt).expanduser().resolve()]
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    ckpts = sorted(ckpt_dir.glob("*_ckpt.pt"), key=checkpoint_iter)
    out = []
    for ckpt in ckpts:
        it = checkpoint_iter(ckpt)
        if it < 0:
            continue
        if args.min_iter is not None and it < args.min_iter:
            continue
        if args.max_iter is not None and it > args.max_iter:
            continue
        if args.every is not None and it % args.every != 0:
            continue
        out.append(ckpt.resolve())
    if not out:
        raise FileNotFoundError("No checkpoints selected")
    return out


def strip_compiled_prefix(state_dict):
    return {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


def resolve_path(path, checkpoint, args, kind):
    if path:
        p = Path(path).expanduser()
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"{kind} not found: {p}")

    config = checkpoint.get("config", {})
    if kind == "meta":
        dataset = config.get("dataset")
        candidates = []
        if dataset:
            candidates.extend(
                [
                    REPO_ROOT / "data" / dataset / "meta.pkl",
                ]
            )
        candidates.append(Path(checkpoint.get("out_dir", "")) / "meta.pkl")
    else:
        test_file = config.get("test_file")
        candidates = []
        if test_file:
            candidates.extend(
                [
                    Path(test_file).expanduser(),
                    REPO_ROOT / test_file,
                    REPO_ROOT / "data" / test_file,
                ]
            )
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Could not resolve {kind}; pass --{kind}")


def encode_json_eval_file(json_path, meta, cfg):
    stoi = meta["stoi"]
    input_key = meta.get("input_key", "input")
    output_key = meta.get("output_key", "output")
    pad_id = stoi[meta.get("pad_token", "<PAD>")]
    sep_id = stoi[meta.get("sep_token", "<SEP>")]
    eos_id = stoi[meta.get("eos_token", "<EOS>")]
    max_quiz_len = cfg.quiz_size - 1
    max_response_len = cfg.response_size - 1
    rows = []

    with open(json_path, "r", encoding="utf-8") as f:
        if json_path.suffix == ".json":
            loaded = json.load(f)
            samples = loaded if isinstance(loaded, list) else loaded.get("data", [loaded])
        else:
            samples = [json.loads(line) for line in f if line.strip()]

    for sample_idx, sample in enumerate(samples):
        quiz_tokens = [stoi[ch] for ch in str(sample.get(input_key, ""))]
        response_tokens = [stoi[ch] for ch in str(sample.get(output_key, ""))]
        if len(quiz_tokens) > max_quiz_len:
            raise ValueError(f"Quiz too long in sample {sample_idx}")
        if len(response_tokens) > max_response_len:
            raise ValueError(f"Response too long in sample {sample_idx}")
        quiz_padded = quiz_tokens + [pad_id] * (max_quiz_len - len(quiz_tokens)) + [sep_id]
        response_padded = response_tokens + [pad_id] * (max_response_len - len(response_tokens)) + [eos_id]
        rows.append(quiz_padded + response_padded)

    return np.array(rows, dtype=np.uint16).reshape(-1)


def load_test_array(test_path, meta, cfg):
    if test_path.suffix in {".jsonl", ".json"}:
        arr = encode_json_eval_file(test_path, meta, cfg)
    else:
        arr = np.memmap(test_path, dtype=np.uint16, mode="r")
    if len(arr) % cfg.base_seq_len != 0:
        raise ValueError(f"Bad test length {len(arr)} for base_seq_len={cfg.base_seq_len}")
    return arr


def deserialize_indexed_response(tokens, cfg):
    expected_len = cfg.quiz_size + 2 * cfg.response_size
    if len(tokens) < expected_len:
        raise ValueError("generated sequence too short")
    response = [-1] * cfg.response_size
    seen = set()
    for i in range(cfg.response_size):
        idx_tok = int(tokens[cfg.quiz_size + 2 * i])
        val_tok = int(tokens[cfg.quiz_size + 2 * i + 1])
        pos = idx_tok - cfg.index_token_start
        if not (0 <= pos < cfg.response_size):
            raise ValueError(f"invalid index token {idx_tok}")
        if pos in seen:
            raise ValueError(f"duplicate index {pos}")
        if not (0 <= val_tok < cfg.index_token_start):
            raise ValueError(f"invalid value token {val_tok}")
        seen.add(pos)
        response[pos] = val_tok
    if any(v == -1 for v in response):
        raise ValueError("missing index")
    return response


def sample_from_logits(logits, temperature, argmax):
    if argmax:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_ar_norepeat(model, prompts, cfg, temperature=0.01, mask_value_vocab=True, argmax=False, allow_repeat=False):
    """Constrained serialized-AR decode with index dedup (project standard 口径).

    Preserves the serialized AR [index, value] pair order, but masks index-token
    choices via a per-position used-mask so an index can never be emitted twice
    (unless allow_repeat=True, which drops the mask to reproduce plain AR). The
    used-mask dedup semantics here must not be changed.
    """
    idx = prompts
    bsz = idx.size(0)
    device = idx.device
    used = torch.zeros(bsz, cfg.response_size, dtype=torch.bool, device=device)

    hidden, kv_cache = model.prefill_kv_cache(idx)
    next_logits = model.lm_head(hidden[:, -1, :])

    for _ in range(cfg.response_size):
        index_logits = next_logits.float()[:, cfg.index_token_start : cfg.index_token_start + cfg.response_size]
        if not allow_repeat:
            index_logits = index_logits.masked_fill(used, -float("inf"))
        picked_pos = sample_from_logits(index_logits, temperature, argmax)
        picked_index = picked_pos + cfg.index_token_start
        if not allow_repeat:
            used.scatter_(1, picked_pos, True)
        idx = torch.cat([idx, picked_index], dim=1)
        hidden, kv_cache = model.append_to_kv_cache(kv_cache, picked_index, return_hidden=True)
        next_logits = model.lm_head(hidden[:, -1, :])

        if mask_value_vocab:
            value_logits = next_logits.float()[:, : cfg.index_token_start]
            picked_value = sample_from_logits(value_logits, temperature, argmax)
        else:
            picked_value = sample_from_logits(next_logits.float(), temperature, argmax)
        idx = torch.cat([idx, picked_value], dim=1)
        hidden, kv_cache = model.append_to_kv_cache(kv_cache, picked_value, return_hidden=True)
        next_logits = model.lm_head(hidden[:, -1, :])

    return idx


@torch.inference_mode()
def generate_batch(model, prompts, cfg, args):
    """Dispatch a batch of prompts to the requested decoder.

    - serialized_ar + norepeat   -> generate_ar_norepeat (used-mask dedup)
    - serialized_ar + ~norepeat  -> model.generate_serialized_ar (plain AR)
    - confidence_guided          -> model.generate_parallel_index (PI; --argmax => deterministic)

    Returns a list-of-lists of decoded token ids ([prompt][I_i, y_i]...).
    """
    if args.mode == "confidence_guided":
        y = model.generate_parallel_index(
            prompts,
            max_new_tokens=cfg.response_size,
            temperature=(0.0 if args.argmax else args.temperature),
            top_k=args.top_k,
        )
        return y.cpu().tolist()

    # serialized_ar
    if args.norepeat:
        y = generate_ar_norepeat(
            model,
            prompts,
            cfg,
            temperature=args.temperature,
            mask_value_vocab=not args.value_full_vocab,
            argmax=args.argmax,
            allow_repeat=False,
        )
        return y.cpu().tolist()

    # Plain serialized AR (no dedup). argmax is expressed as temperature->0 here
    # so it routes through the same model.generate_serialized_ar path.
    y = model.generate_serialized_ar(
        prompts,
        max_new_tokens=cfg.response_size * 2,
        temperature=args.temperature if not args.argmax else 1e-6,
        top_k=args.top_k,
    )
    return y.cpu().tolist()


@torch.inference_mode()
def evaluate(model, test_arr, cfg, info, args):
    rows = np.asarray(test_arr, dtype=np.uint16).reshape(-1, cfg.base_seq_len)
    if args.limit:
        rows = rows[: args.limit]
    rank = info["rank"]
    world = info["world_size"]
    device = info["device"]
    correct = 0
    total = 0
    malformed = 0

    indices = list(range(rank, len(rows), world))
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        batch_rows = rows[batch_indices]
        prompts = torch.tensor(batch_rows[:, : cfg.quiz_size].astype(np.int64), dtype=torch.long, device=device)
        pred = generate_batch(model, prompts, cfg, args)
        for local_j, raw_pred in enumerate(pred):
            target = batch_rows[local_j].astype(np.int64).tolist()
            prompt = target[: cfg.quiz_size]
            try:
                response = deserialize_indexed_response(raw_pred, cfg)
                ok = (prompt + response) == target
            except ValueError:
                malformed += 1
                ok = False
            correct += int(ok)
            total += 1

    counts = torch.tensor([correct, total, malformed], dtype=torch.long, device=device)
    if info["ddp"]:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    c, t, m = [int(x) for x in counts.tolist()]
    return {"accuracy": c / t if t else 0.0, "correct": c, "total": t, "malformed": m}


def load_model(ckpt_path, device, dtype):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    from serfox_model import GPT as _GPT
    model = _GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compiled_prefix(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return checkpoint, model


def str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Ser-FOX (AR / PI)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", help="Single checkpoint .pt file")
    group.add_argument("--ckpt_dir", help="Directory of *_ckpt.pt files (sweep)")
    parser.add_argument(
        "--mode",
        type=str,
        default="serialized_ar",
        choices=["serialized_ar", "confidence_guided"],
        help="Decoding mode: serialized_ar (AR) or confidence_guided (PI).",
    )
    parser.add_argument(
        "--norepeat",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="serialized_ar only: used-mask index dedup (project standard 口径). "
        "Default true. --norepeat false runs plain generate_serialized_ar.",
    )
    parser.add_argument("--meta")
    parser.add_argument("--test_file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_iter", type=int)
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--every", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--argmax", action="store_true", help="Greedy decode (serialized_ar).")
    parser.add_argument(
        "--value_full_vocab",
        action="store_true",
        help="serialized_ar + norepeat: do not restrict value slots to the value vocab.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    info = setup_dist()
    torch.manual_seed(1337 + info["rank"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ckpts = list_checkpoints(args)
    out_path = Path(args.out)
    if info["rank"] == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "iter",
                    "accuracy",
                    "correct",
                    "total",
                    "malformed",
                    "checkpoint",
                    "mode",
                    "norepeat",
                    "temperature",
                    "argmax",
                    "value_full_vocab",
                ]
            )

    meta_cache = {}
    test_cache = {}
    for ckpt in ckpts:
        it = checkpoint_iter(ckpt)
        try:
            checkpoint, model = load_model(ckpt, info["device"], dtype)
            cfg = model.config
            meta_path = resolve_path(args.meta, checkpoint, args, "meta")
            if meta_path not in meta_cache:
                with open(meta_path, "rb") as f:
                    meta_cache[meta_path] = pickle.load(f)
            test_path = resolve_path(args.test_file, checkpoint, args, "test_file")
            test_key = (test_path, cfg.quiz_size, cfg.response_size)
            if test_key not in test_cache:
                test_cache[test_key] = load_test_array(test_path, meta_cache[meta_path], cfg)

            log(f"Evaluating {it} ({args.mode}, norepeat={args.norepeat}) on rank {info['rank']} world {info['world_size']}", info)
            if args.dtype == "float32":
                metrics = evaluate(model, test_cache[test_key], cfg, info, args)
            else:
                with torch.amp.autocast("cuda", dtype=dtype):
                    metrics = evaluate(model, test_cache[test_key], cfg, info, args)

            if info["rank"] == 0:
                row = [
                    datetime.now().isoformat(timespec="seconds"),
                    it,
                    f"{metrics['accuracy']:.6f}",
                    metrics["correct"],
                    metrics["total"],
                    metrics["malformed"],
                    str(ckpt),
                    args.mode,
                    int(args.norepeat),
                    args.temperature,
                    int(args.argmax),
                    int(args.value_full_vocab),
                ]
                with open(out_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                print(
                    f"iter {it}: {args.mode}{'(norepeat)' if args.norepeat and args.mode == 'serialized_ar' else ''} "
                    f"{metrics['accuracy']:.6f} ({metrics['correct']}/{metrics['total']}), "
                    f"malformed {metrics['malformed']}",
                    flush=True,
                )
        except Exception as exc:
            if info["rank"] == 0:
                print(f"ERROR at {ckpt}: {exc}", flush=True)
        finally:
            if "model" in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    teardown_dist(info)


if __name__ == "__main__":
    main()
