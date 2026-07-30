"""Trajectory pair ordering helpers for Ser-FOX training."""

import numpy as np
import torch


SHUFFLE_SPECIAL_POLICIES = ("include", "exclude_eos", "exclude_special")


def validate_shuffle_special_policy(policy):
    if policy not in SHUFFLE_SPECIAL_POLICIES:
        raise ValueError(
            f"shuffle_special_policy must be one of {SHUFFLE_SPECIAL_POLICIES}, got {policy!r}"
        )
    return policy


def _ensure_required_special_ids(policy, eos_id=None, pad_id=None):
    if policy == "exclude_eos" and eos_id is None:
        raise ValueError("--shuffle_special_policy exclude_eos requires an EOS token in meta.pkl")
    if policy == "exclude_special" and eos_id is None and pad_id is None:
        raise ValueError(
            "--shuffle_special_policy exclude_special requires PAD and/or EOS token ids in meta.pkl"
        )


def numpy_fixed_special_mask(values, special_policy="include", eos_id=None, pad_id=None):
    """Return positions that must stay out of the shuffled prefix."""
    validate_shuffle_special_policy(special_policy)
    values = np.asarray(values)
    fixed = np.zeros(values.shape, dtype=bool)
    if special_policy == "include":
        return fixed

    _ensure_required_special_ids(special_policy, eos_id=eos_id, pad_id=pad_id)
    if special_policy in {"exclude_eos", "exclude_special"} and eos_id is not None:
        fixed |= values == eos_id
    if special_policy == "exclude_special" and pad_id is not None:
        fixed |= values == pad_id
    return fixed


def torch_fixed_special_mask(values, special_policy="include", eos_id=None, pad_id=None):
    """Torch equivalent of :func:`numpy_fixed_special_mask`."""
    validate_shuffle_special_policy(special_policy)
    fixed = torch.zeros_like(values, dtype=torch.bool)
    if special_policy == "include":
        return fixed

    _ensure_required_special_ids(special_policy, eos_id=eos_id, pad_id=pad_id)
    if special_policy in {"exclude_eos", "exclude_special"} and eos_id is not None:
        fixed |= values == int(eos_id)
    if special_policy == "exclude_special" and pad_id is not None:
        fixed |= values == int(pad_id)
    return fixed


def build_torch_remaining_eligible_mask(
    remaining_mask,
    values,
    special_policy="include",
    eos_id=None,
    pad_id=None,
):
    """Return unresolved positions allowed by the train-time order policy."""
    if remaining_mask.shape != values.shape:
        raise ValueError(
            "remaining_mask and values must have matching shapes, got "
            f"{tuple(remaining_mask.shape)} and {tuple(values.shape)}"
        )
    remaining_mask = remaining_mask.bool()
    fixed = torch_fixed_special_mask(
        values,
        special_policy=special_policy,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    movable_remaining = remaining_mask & ~fixed
    return torch.where(
        movable_remaining.any(dim=1, keepdim=True),
        movable_remaining,
        remaining_mask,
    )


def build_torch_pair_permutation(values, special_policy="include", eos_id=None, pad_id=None):
    """Build per-row pair permutations for per-batch training shuffle."""
    if values.ndim != 2:
        raise ValueError(f"values must have shape [rows, response_size], got {tuple(values.shape)}")
    validate_shuffle_special_policy(special_policy)

    n_rows, response_size = values.shape
    if response_size == 0:
        return torch.empty((n_rows, 0), dtype=torch.long, device=values.device)

    scores = torch.rand(n_rows, response_size, device=values.device)
    fixed = torch_fixed_special_mask(
        values,
        special_policy=special_policy,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    if fixed.any():
        positions = torch.arange(response_size, device=values.device, dtype=scores.dtype)
        fixed_scores = 2.0 + positions.unsqueeze(0) / float(response_size + 1)
        scores = torch.where(fixed, fixed_scores, scores)
    return scores.argsort(dim=1)
