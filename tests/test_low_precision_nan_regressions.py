import math
import os
import sys

import numpy
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jammy_flows.main.default as f
from jammy_flows.layers.euclidean.gaussianization_flow import gf_block
from jammy_flows.layers.spheres.sphere_base import sphere_base


def _old_sphere_to_plane(self, x, log_det, sf_extra=None):
    assert self.dimension == 1
    sign = (x > numpy.pi) * -1.0 + (x <= numpy.pi) * 1.0
    new_x = (sign > 0) * x + (sign < 0) * (2 * numpy.pi - x)
    used_eps = 1e-5 if x.dtype == torch.float32 else 1e-8
    new_x = torch.where(new_x <= 0.0, used_eps, new_x)
    new_x = torch.where(new_x >= 2 * numpy.pi, 2 * numpy.pi - used_eps, new_x)
    x = numpy.sqrt(2.0) * torch.erfinv(1.0 - new_x / numpy.pi)
    log_det = log_det - numpy.log(numpy.sqrt(2.0 * numpy.pi)) + (x[:, 0] ** 2) / 2.0
    return self.inplane_spherical_to_euclidean(x, log_det, sign)


def _old_compute_householder_matrix(self, vs):
    Q = torch.eye(self.dimension, device=vs.device).type(vs.dtype).unsqueeze(0).repeat(vs.shape[0], 1, 1)
    for i in range(self.householder_iter):
        v = vs[:, i].reshape(-1, self.dimension, 1).to(vs.device)
        v = v / v.norm(dim=1).unsqueeze(-1)
        Qi = torch.eye(self.dimension, device=vs.device).type(vs.dtype).unsqueeze(0) - 2 * torch.bmm(v, v.permute(0, 2, 1))
        Q = torch.bmm(Q, Qi)
    return Q


S1_ANGLES = [0.0, 1e-12, 1e-8, 5e-8, 2 * math.pi - 1e-9, 1e-6, 1e-3, 1.0, math.pi, 4.0]


def _s1_plane_log_pdf(layer, x, to_plane):
    z, log_det = to_plane(layer, x, torch.zeros(x.shape[0], dtype=x.dtype))
    return log_det + torch.distributions.Normal(0.0, 1.0).log_prob(z[:, 0])


@pytest.mark.parametrize("dtype, tiny_angle", [(torch.float32, 1e-8), (torch.float64, 1e-17)])
def test_s1_sphere_to_plane_is_finite_for_tiny_angles(dtype, tiny_angle):
    layer = f.pdf("s1", "y").layer_list[0][0]
    x = torch.tensor(S1_ANGLES + [tiny_angle], dtype=dtype).unsqueeze(1)

    new = _s1_plane_log_pdf(layer, x, sphere_base.sphere_to_plane)
    old = _s1_plane_log_pdf(layer, x, _old_sphere_to_plane)

    assert torch.isfinite(new).all()
    assert not torch.isfinite(old).all()
    finite = torch.isfinite(old)
    torch.testing.assert_close(new[finite], old[finite], rtol=0, atol=1e-5)
    torch.testing.assert_close(new, torch.full_like(new, -math.log(2 * math.pi)), rtol=0, atol=1e-5)


@pytest.mark.parametrize("flow_def", ["y", "o", "m"])
def test_s1_flow_log_pdf_is_finite_and_matches_old_where_finite(flow_def, monkeypatch):
    torch.manual_seed(1)
    flow = f.pdf("s1", flow_def)
    flow.float()
    x = torch.tensor(S1_ANGLES, dtype=torch.float32).unsqueeze(1)

    new, _, _ = flow(x)
    if new.requires_grad:
        new.sum().backward()
    assert torch.isfinite(new).all()
    for p in flow.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    monkeypatch.setattr(sphere_base, "sphere_to_plane", _old_sphere_to_plane)
    with torch.no_grad():
        old, _, _ = flow(x)
    finite = torch.isfinite(old)
    torch.testing.assert_close(new.detach()[finite], old[finite], rtol=0, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("householder_iter", [1, 2, 3])
def test_1d_householder_matrix_is_finite_and_matches_old(dtype, householder_iter):
    layer = gf_block(1, num_householder_iter=householder_iter, use_permanent_parameters=True)
    v = torch.tensor([0.0, 1e-40, 0.3, -2.0, 1e-3], dtype=dtype)
    vs = v.reshape(-1, 1, 1).repeat(1, householder_iter, 1).requires_grad_(True)

    new = layer.compute_householder_matrix(vs)
    old = _old_compute_householder_matrix(layer, vs)

    assert new.shape == old.shape == (v.shape[0], 1, 1)
    assert new.dtype == old.dtype and new.device == old.device
    assert torch.isfinite(new).all()
    finite = torch.isfinite(old)
    assert finite[2:].all()
    torch.testing.assert_close(new[finite], old.detach()[finite], rtol=0, atol=0)
    torch.testing.assert_close(new, torch.full_like(new, (-1.0) ** householder_iter), rtol=0, atol=0)

    (new_grad,) = torch.autograd.grad(new.sum(), vs)
    (old_grad,) = torch.autograd.grad(old.sum(), vs)
    assert torch.isfinite(new_grad).all()
    old_grad_finite = torch.isfinite(old_grad)
    assert old_grad_finite[2:].all()
    torch.testing.assert_close(new_grad[old_grad_finite], old_grad[old_grad_finite], rtol=0, atol=0)


@pytest.mark.parametrize("v_value", [0.0, 1e-40, 0.5])
def test_1d_gaussianization_flow_with_degenerate_householder_vector(v_value, monkeypatch):
    torch.manual_seed(2)
    flow = f.pdf("e1", "gg")
    flow.float()
    for layer in flow.layer_list[0]:
        layer.vs.data.fill_(v_value)
    x = torch.linspace(-3.0, 3.0, 7).unsqueeze(1)

    new, _, _ = flow(x)
    new.sum().backward()
    assert torch.isfinite(new).all()
    for p in flow.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    monkeypatch.setattr(gf_block, "compute_householder_matrix", _old_compute_householder_matrix)
    with torch.no_grad():
        old, _, _ = flow(x)
    finite = torch.isfinite(old)
    torch.testing.assert_close(new.detach()[finite], old[finite], rtol=0, atol=0)


def test_1d_conditional_gaussianization_flow_is_finite():
    torch.manual_seed(3)
    flow = f.pdf("e1", "gg", conditional_input_dim=8)
    flow.float()
    x = torch.randn(16, 1)
    cond = torch.randn(16, 8)

    log_pdf, _, _ = flow(x, conditional_input=cond)
    log_pdf.sum().backward()
    assert torch.isfinite(log_pdf).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
