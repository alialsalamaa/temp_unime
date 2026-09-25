"""CPU checks for paper block LR decay and the released code's stem exponent."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from source.core.optimizer import _build_uniencoder_layer_scales, setup_lldr_optimizer


class TinyUniME(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder = nn.Module()
        encoder.config = SimpleNamespace(depth=16)
        encoder.tokenizer = nn.Linear(2, 2)
        encoder.register_tokens = nn.Parameter(torch.zeros(1, 4, 2))
        encoder.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(16))
        encoder.after_trans_norm = nn.LayerNorm(2)
        self.uni_encoder = nn.Module()
        self.uni_encoder.tok_uniencoder = encoder
        self.decoder = nn.Linear(2, 2)
        self.layer_decay = 0.75
        self.layerwise_lr_decay_enabled = True


class PaperOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(base_lr=3e-4, weight_decay=1e-4, layer_decay=0.75)

    def _parameter_scales(self, optimizer: torch.optim.Optimizer) -> dict[int, float]:
        scales: dict[int, float] = {}
        for group in optimizer.param_groups:
            self.assertEqual(group["weight_decay"], 1e-4)
            for parameter in group["params"]:
                self.assertNotIn(id(parameter), scales, "Parameter assigned more than once")
                scales[id(parameter)] = group["lr_scale"]
        return scales

    def test_transformer_blocks_follow_paper_equation(self) -> None:
        model = TinyUniME()
        # Frozen parameters must not reappear in the optimizer during grouping.
        model.uni_encoder.tok_uniencoder.blocks[2].bias.requires_grad_(False)
        optimizer = setup_lldr_optimizer(self.args, model)
        scales = self._parameter_scales(optimizer)
        expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(set(scales), expected_ids)

        encoder = model.uni_encoder.tok_uniencoder
        self.assertAlmostEqual(scales[id(encoder.tokenizer.weight)], 0.75 ** 17)
        self.assertAlmostEqual(scales[id(encoder.register_tokens)], 0.75 ** 17)
        for layer_number, block in enumerate(encoder.blocks, start=1):
            self.assertAlmostEqual(scales[id(block.weight)], 0.75 ** (16 - layer_number))
        self.assertEqual(scales[id(encoder.after_trans_norm.weight)], 1.0)
        self.assertEqual(scales[id(model.decoder.weight)], 1.0)
        self.assertEqual(
            optimizer.unime_non_uni_encoder_names,
            ["decoder.bias", "decoder.weight"],
        )

    def test_stem_uses_code_fallback_without_shifting_paper_blocks(self) -> None:
        scales = _build_uniencoder_layer_scales(16, 0.75)
        self.assertEqual(len(scales), 18)
        self.assertEqual(scales[0], 0.75 ** 17)
        self.assertEqual(scales[1:17], [0.75 ** exponent for exponent in range(15, -1, -1)])
        self.assertEqual(scales[17], 1.0)
        multiplied = _build_uniencoder_layer_scales(16, 0.75, backbone_multiplier=0.5)
        self.assertEqual(multiplied[:-1], [scale * 0.5 for scale in scales[:-1]])
        self.assertEqual(multiplied[-1], 1.0)

    def test_compile_preserves_parameter_groups_and_names(self) -> None:
        model = TinyUniME()
        ordinary = setup_lldr_optimizer(self.args, model)
        # Construct only the real compile wrapper; no forward or GPU is needed.
        compiled = torch.compile(model, backend="eager")
        self.assertTrue(next(iter(dict(compiled.named_parameters()))).startswith("_orig_mod."))
        wrapped = setup_lldr_optimizer(self.args, compiled)

        self.assertEqual(self._parameter_scales(wrapped), self._parameter_scales(ordinary))
        self.assertEqual(wrapped.unime_layer_schedule, ordinary.unime_layer_schedule)
        self.assertEqual(
            wrapped.unime_non_uni_encoder_names,
            ordinary.unime_non_uni_encoder_names,
        )
        self.assertEqual(len(wrapped.param_groups), 19)  # stem + 16 blocks + norm + decoder


if __name__ == "__main__":
    unittest.main()
