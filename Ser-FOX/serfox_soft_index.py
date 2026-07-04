"""Soft-index supervision helpers for Ser-FOX.

The AR stream still predicts one token at a time. These helpers only replace
hard one-hot supervision on index-token prediction positions with a teacher
distribution over candidate index tokens.
"""

import torch
import torch.nn.functional as F


def _masked_softmax(scores, eligible_mask, temperature):
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not eligible_mask.any(dim=-1).all():
        raise ValueError("Every row must have at least one eligible candidate")
    logits = scores.float() / temperature
    logits = logits.masked_fill(~eligible_mask, -float("inf"))
    return F.softmax(logits, dim=-1)


def soft_index_distribution_from_value_logits(
    value_logits,
    eligible_mask,
    *,
    score_mode="margin",
    temperature=1.0,
    gt_values=None,
):
    """Convert per-index value logits into a soft distribution over indices.

    Args:
        value_logits: Tensor [B, K, V] from parallel candidate value scoring.
        eligible_mask: Bool tensor [B, K] for unresolved candidates.
        score_mode: One of top1_prob, margin, logit_margin, neg_entropy, gt_prob,
            gt_logprob, uniform. See the body for exact definitions. logit_margin
            uses the ground-truth value logit minus the best non-ground-truth
            logit and is the recommended easy-to-hard prior; uniform is an
            order-invariant prior over the remaining positions.
        temperature: Softmax temperature applied to candidate scores.
        gt_values: Optional [B, K] ground-truth value token ids for gt_* and
            logit_margin modes.
    """
    if score_mode in ("top1_prob", "margin", "neg_entropy", "gt_prob", "gt_logprob"):
        probs = F.softmax(value_logits.float(), dim=-1)
        log_probs = F.log_softmax(value_logits.float(), dim=-1)

    if score_mode == "top1_prob":
        scores = probs.max(dim=-1).values
    elif score_mode == "margin":
        top2 = probs.topk(k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            scores = top2[..., 0]
        else:
            scores = top2[..., 0] - top2[..., 1]
    elif score_mode == "logit_margin":
        if gt_values is None:
            raise ValueError(f"gt_values is required for score_mode={score_mode!r}")
        # Ground truth is available when building soft-index targets. Score each
        # position by how far the correct value logit is above the strongest
        # incorrect value logit; plain top1-top2 can reward wrong-but-confident
        # predictions.
        value_logits_f = value_logits.float()
        gt_logits = value_logits_f.gather(dim=-1, index=gt_values.unsqueeze(-1)).squeeze(-1)
        top2 = value_logits_f.topk(k=min(2, value_logits_f.size(-1)), dim=-1)
        if top2.values.size(-1) == 1:
            scores = gt_logits
        else:
            top1_values = top2.values[..., 0]
            top2_values = top2.values[..., 1]
            top1_is_gt = top2.indices[..., 0].eq(gt_values)
            best_other_logits = torch.where(top1_is_gt, top2_values, top1_values)
            scores = gt_logits - best_other_logits
    elif score_mode == "neg_entropy":
        scores = (probs * log_probs).sum(dim=-1)
    elif score_mode in {"gt_prob", "gt_logprob"}:
        if gt_values is None:
            raise ValueError(f"gt_values is required for score_mode={score_mode!r}")
        gathered_probs = probs.gather(dim=-1, index=gt_values.unsqueeze(-1)).squeeze(-1)
        gathered_log_probs = log_probs.gather(dim=-1, index=gt_values.unsqueeze(-1)).squeeze(-1)
        scores = gathered_probs if score_mode == "gt_prob" else gathered_log_probs
    elif score_mode == "uniform":
        # Uniform over remaining (eligible/undecoded) positions: zero scores ->
        # masked softmax = 1/k over the k eligible candidates. No model confidence,
        # just the order-invariance prior that all remaining slots are equal.
        scores = torch.zeros(value_logits.shape[:2], device=value_logits.device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode!r}")

    return _masked_softmax(scores, eligible_mask, temperature)


def mixed_soft_index_ar_loss(
    logits,
    targets,
    index_positions,
    candidate_index_ids,
    soft_index_targets,
):
    """Hard CE on non-index targets, soft CE on index-token targets.

    The returned loss is averaged over hard-supervised tokens plus soft-index
    prediction positions, matching standard AR CE token averaging.
    """
    if index_positions.numel() == 0:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )

    index_positions = index_positions.to(device=targets.device, dtype=torch.long)
    candidate_index_ids = candidate_index_ids.to(device=logits.device, dtype=torch.long)
    soft_index_targets = soft_index_targets.to(device=logits.device, dtype=logits.dtype)

    hard_targets = targets.clone()
    valid_index_mask = torch.zeros_like(targets, dtype=torch.bool)
    valid_index_mask[:, index_positions] = targets[:, index_positions] != -100
    hard_targets[valid_index_mask] = -100

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_hard_targets = hard_targets.reshape(-1)
    hard_loss_sum = F.cross_entropy(
        flat_logits,
        flat_hard_targets,
        ignore_index=-100,
        reduction="sum",
    )
    hard_count = (flat_hard_targets != -100).sum()

    index_logits = logits[:, index_positions, :]
    index_log_probs = F.log_softmax(index_logits.float(), dim=-1)[..., candidate_index_ids]
    soft_loss_per_pos = -(soft_index_targets.float() * index_log_probs).sum(dim=-1)
    soft_valid = targets[:, index_positions] != -100
    soft_loss_sum = soft_loss_per_pos.masked_select(soft_valid).sum()
    soft_count = soft_valid.sum()

    total_count = hard_count + soft_count
    if total_count.item() == 0:
        return logits.sum() * 0.0
    return (hard_loss_sum + soft_loss_sum) / total_count
