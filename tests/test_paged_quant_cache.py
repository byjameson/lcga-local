"""GPU test for the resident int8 quant-cache round trip.

The cache must be a pure speed optimisation: loading it has to produce exactly
the same weights (and therefore the same logits) as quantizing from bf16.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from lcsa.paged_qwen import PagedQwen


@unittest.skipUnless(torch.cuda.is_available(), "resident_quant needs CUDA")
class TestResidentQuantCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(3)
        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        model = Qwen3ForCausalLM(cfg).eval().to(torch.bfloat16)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model"
        model.save_pretrained(cls.path, safe_serialization=True)
        cls.cache = Path(cls.tmp.name) / "qwen-test-int8.pt"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _engine(self, use_cache: bool):
        return PagedQwen(
            self.path, device="cuda", dtype=torch.bfloat16, max_seq=64,
            quant="int8", resident_quant=True,
            quant_cache=self.cache if use_cache else None, verbose=False)

    def test_cache_roundtrip_is_exact(self):
        ids = torch.tensor([[1, 5, 9, 23, 7]], dtype=torch.long)
        first = self._engine(use_cache=True)
        self.assertFalse(first.quant_cache_hit)
        self.assertTrue(self.cache.exists())
        with torch.no_grad():
            want = first.forward(ids, start_pos=0, all_logits=True)

        cached = self._engine(use_cache=True)
        self.assertTrue(cached.quant_cache_hit)
        with torch.no_grad():
            got = cached.forward(ids, start_pos=0, all_logits=True)
        torch.testing.assert_close(got, want, atol=0, rtol=0)

        plain = self._engine(use_cache=False)
        with torch.no_grad():
            ref = plain.forward(ids, start_pos=0, all_logits=True)
        torch.testing.assert_close(ref, want, atol=0, rtol=0)

    def test_corrupt_cache_is_rebuilt(self):
        corrupt = Path(self.tmp.name) / "corrupt-int8.pt"
        corrupt.write_text("not a torch cache", encoding="utf-8")
        first = PagedQwen(
            self.path, device="cuda", dtype=torch.bfloat16, max_seq=64,
            quant="int8", resident_quant=True, quant_cache=corrupt,
            verbose=False)
        self.assertFalse(first.quant_cache_hit)
        second = PagedQwen(
            self.path, device="cuda", dtype=torch.bfloat16, max_seq=64,
            quant="int8", resident_quant=True, quant_cache=corrupt,
            verbose=False)
        self.assertTrue(second.quant_cache_hit)


if __name__ == "__main__":
    unittest.main()
