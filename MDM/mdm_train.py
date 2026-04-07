"""
Unified MDM training entrypoint.

This script exposes the vanilla masked diffusion baseline through a single
training interface that supports both scratch runs and checkpoint resume.
It also normalizes dataset metadata resolution and output-directory handling
so the baseline is easier to run as an open-source reference implementation.
"""

import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import get_logger
from mdm_model import GPT, GPTConfig

# Auto-detect the repository root so this entrypoint still works whether it
# lives directly under the repo root or inside a first-level module directory.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "data").exists():
    REPO_ROOT = REPO_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "out"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def infer_training_profile(dataset):
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

    profile = {
        "task_family": "generic",
        "meta_candidates": generic_meta,
        "default_max_iters": 1_000_000,
        "default_batch_size": 512,
        "out_dir_name": "mdm_train",
        "wandb_project": "mdm",
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
        profile["default_max_iters"] = 50_000
        profile["wandb_project"] = "mdm-sat"
    elif dataset_lower.startswith(("cd4",)):
        profile["task_family"] = "countdown"
        profile["meta_candidates"] = merge_candidates(
            ["cd_meta.pkl", "countdown_meta.pkl"],
            generic_meta,
        )
        profile["default_max_iters"] = 100_000
        profile["wandb_project"] = "mdm-cd4"
    elif dataset_lower.startswith(("cd", "countdown")) or "_cd" in dataset_lower:
        profile["task_family"] = "countdown"
        profile["meta_candidates"] = merge_candidates(
            ["cd_meta.pkl", "countdown_meta.pkl"],
            generic_meta,
        )
        profile["default_max_iters"] = 2_000_000
        profile["wandb_project"] = "mdm-cd"
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
        profile["wandb_project"] = "mdm-cipher"
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
        profile["wandb_project"] = "mdm-path"
    elif dataset_lower.startswith(("sudoku", "sdk")) or "_sudoku" in dataset_lower:
        profile["task_family"] = "sudoku"
        profile["meta_candidates"] = merge_candidates(
            ["sudoku_meta.pkl", "sdk_meta.pkl"],
            generic_meta,
        )
        profile["wandb_project"] = "mdm-sudoku"

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


def resolve_resume_checkpoint(checkpoint_path):
    checkpoint = Path(checkpoint_path).expanduser()
    candidates = [checkpoint]
    if not checkpoint.is_absolute():
        candidates.append(REPO_ROOT / checkpoint)

    resolved = resolve_path(candidates)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def default_out_dir(dataset, n_layer, n_head, n_embd, suffix, max_iters, profile):
    return OUT_ROOT / profile["out_dir_name"] / f"{dataset}_{n_layer}_{n_head}_{n_embd}_{suffix}_{max_iters}"


def strip_unwanted_prefix(state_dict):
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    return state_dict


def validate_resume_metadata(meta, checkpoint_model_args):
    checks = {
        "quiz_size": (meta["quiz_size"], checkpoint_model_args["quiz_size"]),
        "response_size": (meta["response_size"], checkpoint_model_args["response_size"]),
        "block_size": (meta["data_size"], checkpoint_model_args["block_size"]),
        "vocab_size": (meta["vocab_size"], checkpoint_model_args["vocab_size"]),
    }
    mismatches = [
        f"{name}: meta={meta_value}, checkpoint={checkpoint_value}"
        for name, (meta_value, checkpoint_value) in checks.items()
        if meta_value != checkpoint_value
    ]
    if mismatches:
        raise ValueError(
            "Resolved dataset metadata does not match the checkpoint model_args: "
            + "; ".join(mismatches)
        )


def resolve_batch_size(args, checkpoint_config, profile):
    if args.batch_size is not None:
        return args.batch_size
    if args.new_batch_size is not None:
        return args.new_batch_size
    return checkpoint_config.get("train_batch_size", profile["default_batch_size"])


def resolve_learning_rate(args, checkpoint_config, train_batch_size):
    if args.learning_rate is not None:
        return args.learning_rate
    if args.new_learning_rate is not None:
        return args.new_learning_rate
    if "learning_rate" in checkpoint_config:
        return checkpoint_config["learning_rate"]
    base_batch_size = 512
    base_learning_rate = 3e-4
    return base_learning_rate * (train_batch_size / base_batch_size)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def build_parser():
    parser = argparse.ArgumentParser(description="Train the vanilla MDM baseline")
    parser.add_argument("--dataset", type=str, default="cd5$", help="Dataset name (folder under data/)")
    parser.add_argument("--meta_file", type=str, default=None, help="Optional metadata file path or name")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Legacy alias for --resume_from")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding size")
    parser.add_argument("--max_iters", type=int, default=None, help="Total training iterations for scratch mode")
    parser.add_argument("--additional_iters", type=int, default=None, help="Additional iterations for resume mode")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--new_batch_size", type=int, default=None, help="Legacy alias for --batch_size during resume")
    parser.add_argument("--learning_rate", type=float, default=None, help="Optional explicit learning rate")
    parser.add_argument("--new_learning_rate", type=float, default=None, help="Legacy alias for --learning_rate during resume")
    parser.add_argument("--compile", type=parse_bool, default=True, help="Use torch.compile")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix appended to the output directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional explicit output directory")
    parser.add_argument("--eval_iters", type=int, default=None, help="Number of batches used for periodic evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Global gradient accumulation steps")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    resume_from = args.resume_from or args.checkpoint_path
    resume_mode = resume_from is not None
    initial_profile = infer_training_profile(args.dataset)

    if resume_mode:
        checkpoint_path = resolve_resume_checkpoint(resume_from)
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        checkpoint_model_args = dict(checkpoint["model_args"])
        checkpoint_config = dict(checkpoint["config"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

        dataset = checkpoint_config.get("dataset", args.dataset)
        profile = infer_training_profile(dataset)
        meta_path = resolve_meta_path(dataset, args.meta_file, profile)

        n_layer = checkpoint_model_args["n_layer"]
        n_head = checkpoint_model_args["n_head"]
        n_embd = checkpoint_model_args["n_embd"]
        bias = checkpoint_model_args["bias"]
        dropout = checkpoint_model_args["dropout"]

        suffix = args.suffix or "resume-1"
        additional_iters = args.additional_iters if args.additional_iters is not None else 200_000
        max_iters = iter_num + additional_iters
        train_batch_size = resolve_batch_size(args, checkpoint_config, profile)
        val_batch_size = max(1, train_batch_size // 2)
        gradient_accumulation_steps = (
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else checkpoint_config.get("gradient_accumulation_steps", 1)
        )
    else:
        checkpoint = None
        checkpoint_config = {}
        checkpoint_model_args = {}
        iter_num = 0
        best_val_loss = 1e9

        dataset = args.dataset
        profile = initial_profile
        meta_path = resolve_meta_path(dataset, args.meta_file, profile)

        n_layer = args.n_layer
        n_head = args.n_head
        n_embd = args.n_embd
        bias = False
        dropout = 0.1

        suffix = args.suffix or "0-1"
        additional_iters = None
        max_iters = args.max_iters if args.max_iters is not None else profile["default_max_iters"]
        train_batch_size = resolve_batch_size(args, checkpoint_config, profile)
        val_batch_size = max(1, train_batch_size // 2)
        gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 1

    data_dir = meta_path.parent

    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)

    if resume_mode:
        validate_resume_metadata(meta, checkpoint_model_args)

    stoi = meta["stoi"]
    data_size = meta["data_size"]
    quiz_size = meta["quiz_size"]
    response_size = meta["response_size"]
    vocab_size = meta["vocab_size"]

    if "<MASK>" not in stoi:
        raise KeyError("The dataset vocabulary does not define <MASK>, which is required for MDM training.")
    mask_token_id = stoi["<MASK>"]

    if resume_mode:
        existing_mask_token_id = checkpoint_model_args.get("mask_token_id")
        if existing_mask_token_id is not None and existing_mask_token_id != mask_token_id:
            raise ValueError(
                f"Resolved mask_token_id {mask_token_id} does not match checkpoint mask_token_id {existing_mask_token_id}."
            )
        checkpoint_model_args.setdefault("mask_token_id", mask_token_id)
        model_args = checkpoint_model_args
    else:
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=data_size,
            bias=bias,
            vocab_size=vocab_size,
            dropout=dropout,
            quiz_size=quiz_size,
            response_size=response_size,
            mask_token_id=mask_token_id,
        )

    learning_rate = resolve_learning_rate(args, checkpoint_config, train_batch_size)
    print(f"Using learning rate = {learning_rate:.6f} for batch size = {train_batch_size}")

    weight_decay = checkpoint_config.get("weight_decay", 1e-1)
    beta1 = checkpoint_config.get("beta1", 0.9)
    beta2 = checkpoint_config.get("beta2", 0.95)
    grad_clip = checkpoint_config.get("grad_clip", 1.0)
    decay_lr = checkpoint_config.get("decay_lr", True)
    backend = checkpoint_config.get("backend", "nccl")
    dtype = checkpoint_config.get("dtype", "bfloat16")
    compile_model = args.compile

    target_interval_span = additional_iters if resume_mode else max_iters
    eval_interval = max(1, target_interval_span // 10)
    log_interval = max(1, target_interval_span // 100)
    eval_iters = args.eval_iters if args.eval_iters is not None else checkpoint_config.get(
        "eval_iters",
        max(1, min(200, target_interval_span)),
    )
    always_save_checkpoint = True
    eval_only = False
    wandb_log = False
    wandb_project = profile["wandb_project"]
    wandb_run_name = f"mdm-{dataset}-{suffix}"

    warmup_iters = checkpoint_config.get("warmup_iters", max(1, target_interval_span // 20))
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = default_out_dir(dataset, n_layer, n_head, n_embd, suffix, max_iters, profile)

    config = dict(
        dataset=dataset,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_iters=max_iters,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        decay_lr=decay_lr,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dtype=dtype,
        backend=backend,
        compile=compile_model,
        eval_iters=eval_iters,
        out_dir=str(out_dir),
    )

    ddp = int(os.environ.get("RANK", -1)) != -1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        ddp_world_size = 1
        master_process = True
        seed_offset = 0

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * data_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    if ddp:
        torch.distributed.barrier()

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        batch_size = train_batch_size if split == "train" else val_batch_size
        max_index = len(data) - data_size
        if max_index < 0:
            raise ValueError(f"Data size {len(data)} is smaller than data_size {data_size}")

        ix = torch.randint(0, (max_index // data_size) + 1, (batch_size,)) * data_size
        z = torch.stack([torch.from_numpy(data[i:i + data_size].astype(np.int64)) for i in ix])
        quiz = z[:, :quiz_size]
        solution = z[:, quiz_size:]

        # -----------------------------------------------------------------------------
        # MDM training objective
        #
        # Each serialized example in train.bin / val.bin is:
        #   [quiz_padded][response_padded]
        #
        # We construct a corrupted input by masking only the response segment:
        #   x = [quiz][corrupted response]
        #   y = [quiz][clean response]
        #
        # For each sample, we draw a mask ratio uniformly from [0, 1) and then
        # mask that fraction of response positions uniformly without replacement,
        # with at least one masked position per sample.
        #
        # The model sees the full bidirectional state x and predicts all positions,
        # but the loss is computed only on positions that are masked in x.
        #
        # Note: unlike the causal AR baseline, MDM uses block_size = data_size
        # because it consumes the full [quiz][response] state at each denoising step.
        # -----------------------------------------------------------------------------
        mask_ratios = torch.rand(batch_size)
        num_to_mask = torch.ceil(response_size * mask_ratios).to(torch.int64)

        x_solution = solution.clone()
        for batch_index in range(batch_size):
            k = min(response_size, max(1, int(num_to_mask[batch_index].item())))
            mask_indices = torch.randperm(response_size)[:k]
            x_solution[batch_index, mask_indices] = mask_token_id

        x = torch.cat([quiz, x_solution], dim=1)
        y = torch.cat([quiz, solution], dim=1)

        zero_mask_rows = (x_solution == mask_token_id).sum(dim=1) == 0
        if zero_mask_rows.any().item():
            raise RuntimeError(
                f"MDM/mdm_train.py get_batch produced {zero_mask_rows.sum().item()} sample(s) without any masked response tokens; loss normalization would divide by zero."
            )

        assert x.max().item() < vocab_size
        assert y.max().item() < vocab_size
        assert x.min().item() >= 0
        assert y.min().item() >= 0

        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    logger = get_logger(os.path.join(out_dir, "train.log")) if master_process else None

    if resume_mode:
        print(f"Resuming training from iteration {iter_num} with best val loss {best_val_loss:.4f}")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(strip_unwanted_prefix(dict(checkpoint["model"])))
    else:
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    last_val_loss = checkpoint.get("last_val_loss", best_val_loss) if resume_mode else None
    if resume_mode:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    checkpoint = None

    if compile_model:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    @torch.no_grad()
    def estimate_loss(eval_model):
        out = {}
        eval_model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for loss_index in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = eval_model(X, Y)
                losses[loss_index] = loss.item()
            out[split] = losses.mean()
        eval_model.train()
        return out

    def get_lr(step):
        if step < warmup_iters:
            return learning_rate * step / warmup_iters
        if step > lr_decay_iters:
            return min_lr
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    def save_checkpoint(step, current_best_val_loss, current_last_val_loss):
        checkpoint_to_save = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": step,
            "best_val_loss": current_best_val_loss,
            "last_val_loss": current_last_val_loss,
            "config": config,
        }
        checkpoint_path = out_dir / f"{step}_ckpt.pt"
        torch.save(checkpoint_to_save, checkpoint_path)
        return checkpoint_path

    if wandb_log and master_process:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while iter_num < max_iters:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0:
            if ddp:
                torch.distributed.barrier()
            if master_process:
                losses = estimate_loss(raw_model)
                last_val_loss = losses["val"].item()
                eval_message = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                print(eval_message)
                logger.info(eval_message)

                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    })

                save_due_to_improvement = last_val_loss < best_val_loss
                if save_due_to_improvement:
                    best_val_loss = last_val_loss
                if (save_due_to_improvement or always_save_checkpoint) and iter_num > 0:
                    checkpoint_path = save_checkpoint(iter_num, best_val_loss, last_val_loss)
                    save_message = f"saving checkpoint to {checkpoint_path}"
                    print(save_message)
                    logger.info(save_message)
            if ddp:
                torch.distributed.barrier()

        if iter_num == 0 and eval_only:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch("train")
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(train_batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            log_message = f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            print(log_message)
            logger.info(log_message)

        iter_num += 1
        local_iter_num += 1

    if master_process:
        final_step = max(0, iter_num)
        if final_step > 0:
            checkpoint_path = out_dir / f"{final_step}_ckpt.pt"
            if not checkpoint_path.exists():
                if last_val_loss is None:
                    last_val_loss = best_val_loss
                save_checkpoint(final_step, best_val_loss, last_val_loss)
        final_message = f"Training completed. Final iteration: {final_step}, Best val loss: {best_val_loss:.4f}"
        print(final_message)
        logger.info(final_message)

    if ddp:
        destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
