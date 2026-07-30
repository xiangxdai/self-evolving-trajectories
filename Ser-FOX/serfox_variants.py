"""Canonical Ser-FOX objective presets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    index_loss_mode: str
    round1_index_target: str
    serialized_ar_weight: float
    parallel_tail_weight: float


TRAINING_VARIANTS = {
    "basic": TrainingVariant("basic", "hard", "hard", 1.0, 0.0),
    "soft": TrainingVariant("soft", "soft", "hard", 1.0, 0.0),
    "siwei": TrainingVariant("siwei", "hard", "hard", 0.0, 1.0),
    "siwei_soft": TrainingVariant("siwei_soft", "soft", "uniform", 0.0, 1.0),
}

TRAINING_VARIANT_ALIASES = {
    "hard": "basic",
    "siwei-soft": "siwei_soft",
    "siwei_tto0": "siwei_soft",
}


def canonical_training_variant(name):
    if name is None:
        return None
    canonical = TRAINING_VARIANT_ALIASES.get(name, name)
    if canonical not in TRAINING_VARIANTS:
        known = ", ".join(sorted((*TRAINING_VARIANTS, *TRAINING_VARIANT_ALIASES)))
        raise ValueError(f"Unknown training variant {name!r}; expected one of: {known}")
    return canonical


def apply_training_variant(args, explicit_options):
    """Apply a named preset and reject contradictory low-level flags."""
    canonical = canonical_training_variant(args.training_variant)
    if canonical is None:
        return "custom"

    spec = TRAINING_VARIANTS[canonical]
    managed = {
        "index_loss_mode": spec.index_loss_mode,
        "round1_index_target": spec.round1_index_target,
        "serialized_ar_weight": spec.serialized_ar_weight,
        "parallel_tail_weight": spec.parallel_tail_weight,
    }
    for field, expected in managed.items():
        current = getattr(args, field)
        if field in explicit_options and current != expected:
            raise ValueError(
                f"--training_variant {canonical} requires --{field} {expected}, "
                f"but the CLI explicitly supplied {current}"
            )
        setattr(args, field, expected)
    args.training_variant = canonical
    return canonical
