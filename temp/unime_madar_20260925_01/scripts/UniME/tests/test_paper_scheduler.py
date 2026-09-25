"""CPU checks for paper LR values with the released scheduler's clock mechanics."""

import math
from types import SimpleNamespace
import unittest

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

from source.core.lr_scheduler import setup_lldr_scheduler, setup_scheduler


class PaperSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            base_lr=3e-4,
            warmup_lr=1e-5,
            min_lr=1e-6,
            warmup_ratio=0.05,
            num_epochs=600,
            iter_per_epoch=250,
        )

    def _optimizer(self, scales: tuple[float, ...] = (1.0,)) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr_scale": scale}
                for scale in scales
            ],
            lr=self.args.base_lr,
        )

    def _official_rate(self, step: int, total_steps: int = 150000) -> float:
        warmup_steps = int(self.args.warmup_ratio * total_steps)
        if step < warmup_steps:
            return self.args.warmup_lr + step * (
                self.args.base_lr - self.args.warmup_lr
            ) / warmup_steps
        if step >= total_steps:
            return self.args.min_lr
        return self.args.min_lr + 0.5 * (self.args.base_lr - self.args.min_lr) * (
            1.0 + math.cos(math.pi * step / total_steps)
        )

    def test_paper_lr_values_use_official_cosine_clock(self) -> None:
        optimizer = self._optimizer()
        scheduler = setup_scheduler(self.args, optimizer)
        self.assertEqual(scheduler.t_initial, 150000)
        self.assertEqual(scheduler.warmup_t, 7500)
        self.assertFalse(scheduler.warmup_prefix)
        for step in (0, 1, 3750, 7499, 7500, 7501, 75000, 78750, 149999, 150000):
            with self.subTest(step=step):
                scheduler.step_update(step)
                self.assertAlmostEqual(
                    optimizer.param_groups[0]["lr"], self._official_rate(step), places=14
                )
        self.assertLess(self._official_rate(7500), self.args.base_lr)
        self.assertAlmostEqual(self._official_rate(75000), (3e-4 + 1e-6) / 2, places=14)

    def test_matches_released_scheduler_constructor_at_every_update(self) -> None:
        optimizer = self._optimizer()
        reference_optimizer = self._optimizer()
        scheduler = setup_scheduler(self.args, optimizer)
        # Mirror the official constructor, which leaves warmup_prefix at timm's
        # default False, rather than reproducing our wrapper's arguments.
        reference = CosineLRScheduler(
            optimizer=reference_optimizer,
            t_initial=150000,
            lr_min=1e-6,
            warmup_lr_init=1e-5,
            warmup_t=7500,
            cycle_limit=1,
            t_in_epochs=False,
        )
        for step in range(150001):
            scheduler.step_update(step)
            reference.step_update(step)
            self.assertEqual(optimizer.param_groups[0]["lr"], reference_optimizer.param_groups[0]["lr"])

    def test_layer_scales_apply_through_warmup_and_decay(self) -> None:
        scales = (0.75 ** 17, 0.75 ** 15, 0.75, 1.0)
        optimizer = self._optimizer(scales)
        scheduler = setup_lldr_scheduler(self.args, optimizer)
        for step in (0, 3750, 7499, 7500, 75000, 150000):
            base_rate = self._official_rate(step)
            scheduler.step_update(step)
            for group, scale in zip(optimizer.param_groups, scales):
                with self.subTest(step=step, scale=scale):
                    self.assertAlmostEqual(group["lr"], base_rate * scale, places=14)

    def test_no_warmup_starts_at_peak(self) -> None:
        self.args.warmup_ratio = 0.0
        optimizer = self._optimizer()
        scheduler = setup_scheduler(self.args, optimizer)
        scheduler.step_update(0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self.args.base_lr, places=14)
        scheduler.step_update(150000)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self.args.min_lr, places=14)

    def test_short_runs_allow_zero_or_one_warmup_steps(self) -> None:
        for epochs, steps_per_epoch, ratio, warmup_steps in ((1, 1, 0.05, 0), (1, 2, 0.5, 1)):
            with self.subTest(total_steps=epochs * steps_per_epoch):
                self.args.num_epochs = epochs
                self.args.warmup_ratio = ratio
                optimizer = self._optimizer()
                scheduler = setup_scheduler(self.args, optimizer, steps_per_epoch=steps_per_epoch)
                scheduler.step_update(warmup_steps)
                self.assertAlmostEqual(
                    optimizer.param_groups[0]["lr"],
                    self._official_rate(warmup_steps, epochs * steps_per_epoch),
                    places=14,
                )
                scheduler.step_update(epochs * steps_per_epoch)
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], self.args.min_lr, places=14)

    def test_rejects_invalid_warmup_ratios(self) -> None:
        for ratio in (-0.01, 1.0, 2.0, math.nan, math.inf):
            with self.subTest(ratio=ratio):
                self.args.warmup_ratio = ratio
                with self.assertRaisesRegex(ValueError, "warmup_ratio"):
                    setup_scheduler(self.args, self._optimizer())

    def test_rejects_nonpositive_total_steps(self) -> None:
        for epochs, steps_per_epoch in ((0, 250), (600, 0), (-1, 250), (600, -1)):
            with self.subTest(epochs=epochs, steps_per_epoch=steps_per_epoch):
                self.args.num_epochs = epochs
                with self.assertRaisesRegex(ValueError, "positive total"):
                    setup_scheduler(self.args, self._optimizer(), steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    unittest.main()
