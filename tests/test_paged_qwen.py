"""CPU-only parity check for the paged engine against HuggingFace Qwen3.

The engine started as a Qwen2 implementation. Qwen3 adds a per-head q/k RMSNorm
and decouples ``head_dim`` from ``hidden_size / num_heads``; this test pins both
down on a tiny randomly-initialised model, so the check is fast and needs no GPU
and no downloaded weights.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from lcsa.paged_qwen import PagedQwen


class TestQwen3Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        cls.cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,           # 4 * 32 = 128 != hidden_size (64)
            max_position_embeddings=64,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        cls.ref = Qwen3ForCausalLM(cls.cfg).eval()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.ref.save_pretrained(cls.path, safe_serialization=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_head_dim_is_read_from_config(self):
        eng = PagedQwen(self.path, device="cpu", dtype=torch.float32,
                        max_seq=64, quant="none", verbose=False)
        self.assertEqual(eng.head_dim, 32)
        self.assertEqual(eng.n_heads * eng.head_dim, 128)

    def test_logits_match_huggingface(self):
        eng = PagedQwen(self.path, device="cpu", dtype=torch.float32,
                        max_seq=64, quant="none", verbose=False)
        ids = torch.tensor([[1, 5, 9, 23, 7]], dtype=torch.long)
        with torch.no_grad():
            ref = self.ref(ids).logits
            got = eng.forward(ids, start_pos=0, all_logits=True)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_incremental_decode_matches_full_forward(self):
        eng = PagedQwen(self.path, device="cpu", dtype=torch.float32,
                        max_seq=64, quant="none", verbose=False)
        ids = torch.tensor([[1, 5, 9, 23, 7, 11]], dtype=torch.long)
        with torch.no_grad():
            ref = self.ref(ids).logits[:, -1]
            eng.reset_cache()
            eng.forward(ids[:, :5], start_pos=0, all_logits=False)
            last = eng.forward(ids[:, 5:], start_pos=5, all_logits=False)
        torch.testing.assert_close(last[:, -1], ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
