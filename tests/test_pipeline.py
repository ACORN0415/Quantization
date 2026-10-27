"""Exercise real pipeline RNG branches with tiny deterministic model substitutes.

These are regression tests, not a substitute for the full-model G3 gate.
"""
from types import SimpleNamespace
import unittest
import torch
from pipeline import CausalInferencePipeline


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(local_attn_size=6)

    def get_scheduler(self):
        return SimpleNamespace(add_noise=lambda x, noise, t: x + noise * 0.1)

    def forward(self, noisy_image_or_video, **kwargs):
        return None, noisy_image_or_video * 0.75


class Encoder(torch.nn.Module):
    def forward(self, **kwargs): return {}


class VAE(torch.nn.Module):
    def decode_to_pixel(self, output, **kwargs): return output


class PipelineTests(unittest.TestCase):
    def test_chunk_rng_drain(self):
        config = SimpleNamespace(denoising_step_list=[1000,750,500,250], warp_denoising_step=False,
                                 num_frame_per_block=3, independent_first_frame=False, context_noise=0)
        pipe = CausalInferencePipeline(config, 'cpu', generator=Generator(), text_encoder=Encoder(), vae=VAE())
        pipe.num_transformer_blocks, pipe.frame_seq_length = 1, 1
        noise = torch.randn(1, 6, 1, 2, 2)
        def run(seed, drain):
            torch.manual_seed(123)
            return pipe.inference(noise.clone(), ['test'], return_latents=True,
                                  base_seed=seed, drain_rng=drain)[1]
        self.assertTrue(torch.equal(run(17, False), run(17, True)))
        self.assertFalse(torch.equal(run(None, False), run(None, True)))


if __name__ == '__main__': unittest.main()
