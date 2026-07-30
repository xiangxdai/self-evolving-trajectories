import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SERFOX = ROOT / "Ser-FOX"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERFOX))

from configs.serfox_task_configs import get_task_config
from serfox_model import GPT, GPTConfig
from serfox_objectives import build_parallel_tail_batch, parallel_tail_loss
from serfox_ordering import (
    build_torch_pair_permutation,
    build_torch_remaining_eligible_mask,
)
from serfox_variants import apply_training_variant


def tiny_config():
    return GPTConfig(
        vocab_size=15,
        block_size=14,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        bias=False,
        quiz_size=2,
        response_size=4,
        value_vocab_size=11,
    )


class Tto0BatchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config()
        self.idx = torch.tensor(
            [
                [1, 2, 11, 4, 13, 6, 12, 5, 14],
                [2, 1, 12, 3, 11, 7, 14, 8, 13],
            ],
            dtype=torch.long,
        )
        self.pairs = torch.tensor(
            [
                [[11, 4], [13, 6], [12, 5], [14, 7]],
                [[12, 3], [11, 7], [14, 8], [13, 6]],
            ],
            dtype=torch.long,
        )

    def test_random_prefix_layout_matches_serfox_tto0(self):
        batch = build_parallel_tail_batch(self.idx, self.pairs, self.cfg, step=2)
        expected = torch.cat(
            [
                self.idx[:, : self.cfg.quiz_size],
                self.pairs[:, :2].reshape(2, -1),
                self.pairs[:, 2:, 0],
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(batch.input_tokens, expected))
        self.assertTrue(torch.equal(batch.next_index_targets, self.pairs[:, 2, 0]))
        self.assertTrue(torch.equal(batch.value_targets, self.pairs[:, 2:, 1]))
        self.assertEqual(batch.num_tail, 2)

    def test_balanced_loss_equals_original_weighted_formula(self):
        batch = build_parallel_tail_batch(self.idx, self.pairs, self.cfg, step=1)
        torch.manual_seed(3)
        logits = torch.randn(2, batch.num_tail + 1, self.cfg.vocab_size)
        actual = parallel_tail_loss(
            logits,
            batch,
            index_loss_mode="hard",
            index_token_start=self.cfg.index_token_start,
            response_size=self.cfg.response_size,
        )
        index = F.cross_entropy(logits[:, 0, :], batch.next_index_targets)
        values = F.cross_entropy(
            logits[:, 1:, :].reshape(-1, logits.size(-1)),
            batch.value_targets.reshape(-1),
        )
        self.assertTrue(torch.allclose(actual, 0.5 * index + 0.5 * values))

    def test_soft_loss_skips_value_only_index_rows(self):
        batch = build_parallel_tail_batch(self.idx, self.pairs, self.cfg, step=1)
        logits = torch.randn(2, batch.num_tail + 1, self.cfg.vocab_size)
        dist = torch.zeros(2, self.cfg.response_size)
        dist[0, 2:] = 0.5
        loss = parallel_tail_loss(
            logits,
            batch,
            index_loss_mode="soft",
            index_token_start=self.cfg.index_token_start,
            response_size=self.cfg.response_size,
            soft_index_distribution=dist,
            index_supervision_mask=torch.tensor([True, False]),
        )
        self.assertTrue(torch.isfinite(loss))


class OrderingAndVariantTests(unittest.TestCase):
    def test_exclude_special_keeps_pad_and_eos_at_tail(self):
        torch.manual_seed(4)
        values = torch.tensor([[1, 9, 2, 10, 3], [10, 4, 9, 5, 6]])
        perm = build_torch_pair_permutation(
            values,
            special_policy="exclude_special",
            eos_id=9,
            pad_id=10,
        )
        shuffled = values.gather(1, perm)
        self.assertTrue(torch.equal(shuffled[:, -2:].sort().values, torch.tensor([[9, 10], [9, 10]])))

    def test_uniform_eligibility_matches_exclude_policy(self):
        remaining = torch.ones(1, 5, dtype=torch.bool)
        values = torch.tensor([[1, 9, 2, 10, 3]])
        eligible = build_torch_remaining_eligible_mask(
            remaining,
            values,
            special_policy="exclude_special",
            eos_id=9,
            pad_id=10,
        )
        self.assertTrue(torch.equal(eligible, torch.tensor([[True, False, True, False, True]])))

    def test_siwei_soft_is_pure_tto0(self):
        args = Namespace(
            training_variant="siwei_soft",
            index_loss_mode="hard",
            round1_index_target="hard",
            serialized_ar_weight=1.0,
            parallel_tail_weight=0.0,
        )
        self.assertEqual(apply_training_variant(args, set()), "siwei_soft")
        self.assertEqual(args.index_loss_mode, "soft")
        self.assertEqual(args.round1_index_target, "uniform")
        self.assertEqual(args.serialized_ar_weight, 0.0)
        self.assertEqual(args.parallel_tail_weight, 1.0)

    def test_task_profiles_default_to_egb512(self):
        cfg = get_task_config("cd4")
        defaults = cfg.as_arg_defaults()
        self.assertEqual(defaults["train_batch_size"], 256)
        self.assertEqual(defaults["gradient_accumulation_steps"], 2)
        self.assertEqual(
            defaults["train_batch_size"] * defaults["gradient_accumulation_steps"],
            512,
        )


class ModelBoundaryTests(unittest.TestCase):
    def test_tail_shape_and_gradients(self):
        cfg = tiny_config()
        model = GPT(cfg)
        pairs = torch.tensor([[[11, 4], [13, 6], [12, 5], [14, 7]]])
        idx = torch.tensor([[1, 2, 11, 4, 13, 6, 12, 5, 14]])
        batch = build_parallel_tail_batch(idx, pairs, cfg, step=2)
        logits = model(batch.input_tokens, parallel_tail=True)
        self.assertEqual(tuple(logits.shape), (1, 3, cfg.vocab_size))
        loss = logits.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_scoped_compile_accepts_two_static_tail_shapes(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        cfg = tiny_config()
        model = GPT(cfg)
        model.enable_parallel_tail_compile(dynamic=False, backend="eager")
        for step in (0, 2):
            pairs = torch.tensor([[[11, 4], [13, 6], [12, 5], [14, 7]]])
            idx = torch.tensor([[1, 2, 11, 4, 13, 6, 12, 5, 14]])
            batch = build_parallel_tail_batch(idx, pairs, cfg, step=step)
            logits = model(batch.input_tokens, parallel_tail=True)
            self.assertEqual(
                tuple(logits.shape),
                (1, batch.num_tail + 1, cfg.vocab_size),
            )
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
