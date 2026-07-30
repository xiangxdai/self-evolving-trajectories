"""Loss helpers for the Ser-FOX Siwei/T->0 objective."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ParallelTailBatch:
    input_tokens: torch.Tensor
    next_index_targets: torch.Tensor
    value_targets: torch.Tensor
    step: int
    num_tail: int


def build_parallel_tail_batch(idx, pairs, config, step=None):
    """Build one random-prefix parallel-tail batch."""
    if pairs.ndim != 3 or pairs.size(1) != config.response_size or pairs.size(2) != 2:
        raise ValueError(
            "pairs must have shape [batch, response_size, 2], got "
            f"{tuple(pairs.shape)}"
        )
    if step is None:
        step = int(torch.randint(0, config.response_size, (1,), device=idx.device).item())
    if not 0 <= int(step) < config.response_size:
        raise ValueError(f"step must be in [0, {config.response_size}), got {step}")

    step = int(step)
    num_tail = config.response_size - step
    prompt = idx[:, :config.quiz_size]
    prefix_pairs = pairs[:, :step].reshape(idx.size(0), -1)
    tail_indices = pairs[:, step:, 0]
    input_tokens = torch.cat([prompt, prefix_pairs, tail_indices], dim=1)
    return ParallelTailBatch(
        input_tokens=input_tokens,
        next_index_targets=pairs[:, step, 0],
        value_targets=pairs[:, step:, 1],
        step=step,
        num_tail=num_tail,
    )


def parallel_tail_loss(
    logits,
    batch,
    *,
    index_loss_mode,
    index_token_start,
    response_size,
    soft_index_distribution=None,
    index_supervision_mask=None,
):
    """Balanced next-index and remaining-value loss."""
    expected = (batch.input_tokens.size(0), batch.num_tail + 1)
    if logits.ndim != 3 or tuple(logits.shape[:2]) != expected:
        raise ValueError(
            f"parallel-tail logits must start with shape {expected}, got {tuple(logits.shape)}"
        )

    if index_supervision_mask is None:
        supervised_rows = torch.ones(
            logits.size(0), dtype=torch.bool, device=logits.device
        )
    else:
        supervised_rows = index_supervision_mask.to(
            device=logits.device, dtype=torch.bool
        )
        if tuple(supervised_rows.shape) != (logits.size(0),):
            raise ValueError(
                "index supervision mask must have shape "
                f"{(logits.size(0),)}, got {tuple(supervised_rows.shape)}"
            )

    if index_loss_mode == "soft":
        if soft_index_distribution is None:
            raise ValueError("soft parallel-tail training requires a soft index distribution")
        dist = soft_index_distribution.float()
        if tuple(dist.shape) != (logits.size(0), response_size):
            raise ValueError(
                "soft index distribution must have shape "
                f"{(logits.size(0), response_size)}, got {tuple(dist.shape)}"
            )
        mass = dist.sum(dim=1)
        if torch.any(supervised_rows & (mass <= 0)):
            raise ValueError(
                "every supervised soft parallel-tail row must have positive probability mass"
            )
        dist = dist / mass.clamp_min(torch.finfo(dist.dtype).tiny).unsqueeze(1)
        candidate_ids = torch.arange(
            response_size, device=logits.device, dtype=torch.long
        ) + index_token_start
        index_log_probs = F.log_softmax(logits[:, 0, candidate_ids], dim=-1)
        index_loss_per_row = -(dist * index_log_probs).sum(dim=1)
    elif index_loss_mode == "hard":
        index_loss_per_row = F.cross_entropy(
            logits[:, 0, :], batch.next_index_targets, reduction="none"
        )
    else:
        raise ValueError(f"index_loss_mode must be hard or soft, got {index_loss_mode!r}")

    if supervised_rows.any():
        index_loss = index_loss_per_row[supervised_rows].mean()
    else:
        index_loss = logits[:, 0, :].sum() * 0.0

    value_loss = F.cross_entropy(
        logits[:, 1:, :].reshape(-1, logits.size(-1)),
        batch.value_targets.reshape(-1),
    )
    return 0.5 * index_loss + 0.5 * value_loss
