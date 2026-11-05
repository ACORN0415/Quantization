"""Separate fixed offline PSAC screening adaptation, synthetic preparation only.

M_r = stored BF16 U @ diag(stored FP32 lambda) @ U.T. No fit or full moment
is retained. Coefficients, scale, BF16 restoration and packets reuse frozen core.
This is neither official QuantWM code nor a new-method contribution.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch

CORE_SHA256 = '63364f9d4619a231ea5425fcf8e60bf8254bbde738948a952c6604d5432185d9'
CORE_PATH = Path(__file__).resolve().with_name('core.py')
if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
    raise RuntimeError('The reviewed full-M core changed; a new binding review is required')
_spec = importlib.util.spec_from_file_location('_psac_projected_frozen_core_v2', CORE_PATH)
c = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = c
_spec.loader.exec_module(c)


@dataclass
class ProjectedMetric:
    basis: torch.Tensor        # one shared BF16 [H,D,r]
    eigenvalues: torch.Tensor  # one shared FP32 [H,r]

    def validate(self):
        c.finite_cpu(self.basis, 'stored_basis')
        c.finite_cpu(self.eigenvalues, 'stored_eigenvalues')
        c.require(self.basis.dtype == torch.bfloat16 and self.basis.ndim == 3,
                  'stored_basis must be BF16 [H,D,r]')
        h,d,r = self.basis.shape
        c.require(h > 0 and d > 0 and 0 < r <= d, 'invalid basis dimensions')
        c.require(self.eigenvalues.dtype == torch.float32 and self.eigenvalues.shape == (h,r),
                  'stored_eigenvalues must be FP32 [H,r]')
        c.require(bool((self.eigenvalues >= 0).all()), 'negative eigenvalue')
        c.require(self.basis.is_contiguous() and self.eigenvalues.is_contiguous(),
                  'contiguous state required')
        c.require(not self.basis.requires_grad and self.basis.grad_fn is None and
                  not self.eigenvalues.requires_grad and self.eigenvalues.grad_fn is None,
                  'state must not retain an autograd graph; use prepare_metric')
        return h,d,r

    def tensor_payload_bytes(self):
        self.validate()
        return self.basis.numel()*self.basis.element_size() + self.eigenvalues.numel()*self.eigenvalues.element_size()


@torch.no_grad()
def prepare_metric(stored_basis, eigenvalues):
    """Own externally fixed U/lambda, with no selection, covariance fit or clamp.

    Caller must verify the p0 origin/order and common K/error coordinate system.
    BF16 U is not re-orthogonalized: this defines the actual approximate metric.
    """
    c.finite_cpu(stored_basis, 'stored_basis'); c.finite_cpu(eigenvalues, 'eigenvalues')
    state = ProjectedMetric(stored_basis.detach().clone().contiguous(),
                            eigenvalues.detach().clone().contiguous())
    state.validate()
    return state


def _score(projected, state):
    score = (projected.square() * state.eigenvalues.double()[None,:,:]).sum(-1)
    c.require(bool(torch.isfinite(score).all()), 'nonfinite projected screening score')
    return score


@torch.no_grad()
def projected_row_error(error, state):
    """FP64 row scores (E U)^2 dot lambda; no H*D*D tensor is formed."""
    h,d,r = state.validate(); c.finite_cpu(error, 'error')
    c.require(error.ndim == 3 and error.shape[1:] == (h,d), 'error [N,H,D] required')
    projected = torch.einsum('nhd,hdr->nhr', error.double(), state.basis.double())
    return _score(projected, state)


@torch.no_grad()
def psac_encode_projected(key, reconstructed_key, state):
    """Screen actual BF16 reconstruction with the separate projected metric."""
    h,d,r = state.validate()
    c.finite_cpu(key,'key'); c.finite_cpu(reconstructed_key,'reconstructed_key')
    c.require(key.ndim == 3 and key.shape == reconstructed_key.shape and key.shape[1:] == (h,d), 'key shape')
    c.require(key.dtype == reconstructed_key.dtype == torch.bfloat16 and key.shape[0] > 0, 'BF16 K/Khat required')
    n = key.shape[0]
    error = key.float()-reconstructed_key.float()
    coefficients = torch.einsum('nhd,hdr->nhr', error.double(), state.basis.double())
    codes,scales,quant_info = c.quantize_rows(coefficients.reshape(n*h,r))
    chat = (codes.float()*scales.float()).reshape(n,h,r)
    correction = torch.einsum('nhr,hdr->nhd', chat, state.basis.float())
    corrected = (reconstructed_key.float()+correction).to(torch.bfloat16)
    c.require(bool(torch.isfinite(corrected).all()), 'corrected BF16 overflow')
    before = _score(coefficients, state)
    after = projected_row_error(key.float()-corrected.float(), state)
    active = (after < before).reshape(-1)
    packet,storage = c.packet_from_rows(codes,scales,active,n,h,d)
    return packet, {**storage, **quant_info,
                   'basis_bytes_shared': state.basis.numel()*2,
                   'eigenvalue_bytes_shared': state.eigenvalues.numel()*4,
                   'metric_tensor_bytes_shared': state.tensor_payload_bytes(),
                   'retained_full_moment_bytes': 0,
                   'before_query_error': float(before.sum()),
                   'after_screened_query_error': float(torch.where(active.reshape(n,h),after,before).sum()),
                   'metric_definition': 'stored_U_BF16_diag_stored_lambda_FP32_UT'}


@torch.no_grad()
def psac_restore(reconstructed_key, packet, state):
    state.validate()
    return c.psac_restore(reconstructed_key, packet, state.basis)
