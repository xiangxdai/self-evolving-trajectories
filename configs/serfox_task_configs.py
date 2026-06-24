"""Saved source-level run profiles for Ser-FOX tasks.

The training entrypoint imports this file when ``--task`` is used.  Keep only
stable task-selection defaults here: model size, dataset selection, training
rounds, and learning rate.  Explicit command-line flags still win.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


REGIME_MIX_RATIOS = {
    "warm": "1.0,0.0,0.0",
    "warmmix": "0.7,0.2,0.1",
    "warmbest": "1.0,0.0,0.0",
    "warmmixbest": "0.7,0.2,0.1",
}

# A shared "general" backbone available for EVERY task, in addition to each
# task's own original/small backbone. sudoku's own size already equals this,
# so for sudoku small == general.
GENERAL_BACKBONE = (3, 12, 384)


@dataclass(frozen=True)
class SerFoxTaskConfig:
    name: str
    dataset: str
    n_layer: int
    n_head: int
    n_embd: int
    rounds: int
    round_interval: int
    learning_rate: float
    test_file: str
    train_batch_size: int = 256
    eval_batch_size: int = 256
    gradient_accumulation_steps: int = 4
    shuffle_order: bool = True
    first_round_l2r: bool = True
    mix_ratios: str = "1.0,0.0,0.0"

    def max_iters(self, rounds: Optional[int] = None) -> int:
        return self.round_interval * (self.rounds if rounds is None else rounds)

    def eval_interval(self) -> int:
        return max(1, self.round_interval // 10)

    def backbone(self, kind: str = "small") -> Tuple[int, int, int]:
        """(n_layer, n_head, n_embd) for the chosen backbone: 'small'=task's own
        original size, 'general'=the shared GENERAL_BACKBONE (3,12,384)."""
        if kind == "general":
            return GENERAL_BACKBONE
        return (self.n_layer, self.n_head, self.n_embd)

    def as_arg_defaults(
        self,
        *,
        rounds: Optional[int] = None,
        regime: Optional[str] = None,
        backbone: str = "small",
    ) -> Dict[str, object]:
        mix_ratios = self.mix_ratios
        if regime is not None:
            mix_ratios = REGIME_MIX_RATIOS[regime]

        bb_layer, bb_head, bb_embd = self.backbone(backbone)
        return {
            "dataset": self.dataset,
            "n_layer": bb_layer,
            "n_head": bb_head,
            "n_embd": bb_embd,
            "max_iters": self.max_iters(rounds),
            "round_interval": self.round_interval,
            "eval_interval": self.eval_interval(),
            "checkpoint_interval": self.eval_interval(),
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "shuffle_order": self.shuffle_order,
            "first_round_l2r": self.first_round_l2r,
            "mix_ratios": mix_ratios,
            "test_file": self.test_file,
        }

    def as_metadata(
        self,
        *,
        rounds: Optional[int] = None,
        regime: Optional[str] = None,
        backbone: str = "small",
    ) -> Dict[str, object]:
        data = asdict(self)
        data["selected_rounds"] = self.rounds if rounds is None else rounds
        data["selected_max_iters"] = self.max_iters(rounds)
        data["selected_regime"] = regime
        data["selected_backbone"] = backbone
        bb_layer, bb_head, bb_embd = self.backbone(backbone)
        data["selected_n_layer"] = bb_layer
        data["selected_n_head"] = bb_head
        data["selected_n_embd"] = bb_embd
        if regime is not None:
            data["selected_mix_ratios"] = REGIME_MIX_RATIOS[regime]
        else:
            data["selected_mix_ratios"] = self.mix_ratios
        return data


TASK_CONFIGS: Dict[str, SerFoxTaskConfig] = {
    "sat7": SerFoxTaskConfig(
        name="sat7",
        dataset="3sat/3sat7/k1",
        n_layer=3,
        n_head=6,
        n_embd=240,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="3sat7_test.jsonl",
    ),
    "sat9": SerFoxTaskConfig(
        name="sat9",
        dataset="3sat/3sat9/k1",
        n_layer=3,
        n_head=8,
        n_embd=256,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="3sat9_test.jsonl",
    ),
    "cd3": SerFoxTaskConfig(
        name="cd3",
        dataset="cd/cd3/k1",
        n_layer=3,
        n_head=7,
        n_embd=224,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="cd3_test.jsonl",
    ),
    "cd4": SerFoxTaskConfig(
        name="cd4",
        dataset="cd/cd4/k1",
        n_layer=4,
        n_head=8,
        n_embd=512,
        rounds=10,
        round_interval=100_000,
        learning_rate=3e-4,
        test_file="cd4_test.jsonl",
    ),
    "path10": SerFoxTaskConfig(
        name="path10",
        dataset="path/pd9/k1",
        n_layer=3,
        n_head=8,
        n_embd=256,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="path_test-2-10.jsonl",
    ),
    "path14": SerFoxTaskConfig(
        name="path14",
        dataset="path/pd13/k1",
        n_layer=3,
        n_head=8,
        n_embd=256,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="path_test-2-14.jsonl",
    ),
    "sudoku": SerFoxTaskConfig(
        name="sudoku",
        dataset="sudoku_1M_fixed",
        n_layer=3,
        n_head=12,
        n_embd=384,
        rounds=10,
        round_interval=50_000,
        learning_rate=3e-4,
        test_file="sudoku_test.jsonl",
    ),
}


def task_names() -> Tuple[str, ...]:
    return tuple(TASK_CONFIGS)


def get_task_config(name: str) -> SerFoxTaskConfig:
    try:
        return TASK_CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(task_names())
        raise KeyError(f"Unknown Ser-FOX task {name!r}. Known tasks: {known}") from exc
