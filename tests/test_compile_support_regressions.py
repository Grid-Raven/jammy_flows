import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jammy_flows.main.default as f
from jammy_flows.layers import bisection_n_newton
from jammy_flows.rng_fns import (
    _seeded_standard_normal_impl,
    draw_standard_normal,
)


def test_seeded_sampling_does_not_mutate_global_torch_rng_state():
    flow = f.pdf("e1", "x")
    flow.double()

    torch.manual_seed(12345)
    expected_next = torch.rand(8)

    torch.manual_seed(12345)
    flow.sample(
        samplesize=4,
        seed=999,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    actual_next = torch.rand(8)

    torch.testing.assert_close(actual_next, expected_next, rtol=0.0, atol=0.0)


def test_obtain_flow_param_structure_seed_does_not_mutate_global_torch_rng_state():
    flow = f.pdf("e1", "x")
    flow.double()

    torch.manual_seed(12345)
    expected_next = torch.rand(8)

    torch.manual_seed(12345)
    flow.obtain_flow_param_structure(
        seed=999,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    actual_next = torch.rand(8)

    torch.testing.assert_close(actual_next, expected_next, rtol=0.0, atol=0.0)


def test_seeded_standard_normal_compiles_fullgraph():
    if not hasattr(torch, "compile") or not hasattr(
        getattr(torch, "library", None),
        "custom_op",
    ):
        pytest.skip("fullgraph custom-op support is unavailable")

    def generate():
        return draw_standard_normal(
            8,
            3,
            torch.float32,
            torch.device("cpu"),
            seed=123,
        )

    expected = _seeded_standard_normal_impl(
        8,
        3,
        123,
        torch.float32,
        torch.device("cpu"),
    )
    compiled = torch.compile(generate, fullgraph=True)
    first = compiled()
    second = compiled()

    assert first.shape == (8, 3)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)


def _monotonic_func(z, scale):
    return z + scale * z.pow(3)


def _monotonic_joint_func(z, scale):
    return _monotonic_func(z, scale), 1.0 + 3.0 * scale * z.pow(2)


def _solve_monotonic(target, scale):
    return bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad(
        _monotonic_func,
        _monotonic_joint_func,
        target,
        scale,
        min_boundary=-5.0,
        max_boundary=5.0,
        num_bisection_iter=40,
        num_newton_iter=20,
        newton_tolerance=1e-13,
    )


def _solve_monotonic_compile_probe(target, scale):
    return bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad(
        _monotonic_func,
        _monotonic_joint_func,
        target,
        scale,
        min_boundary=-1.0,
        max_boundary=1.0,
        num_bisection_iter=4,
        num_newton_iter=4,
        newton_tolerance=1e-6,
    )


def test_joint_bisection_newton_inverse_solves_batched_monotonic_equation():
    z_true = torch.tensor(
        [[-3.0], [-0.1], [0.0], [0.2], [4.0]],
        dtype=torch.float64,
    )
    scale = torch.tensor(
        [[0.01], [0.5], [1.0], [0.2], [0.03]],
        dtype=torch.float64,
    )
    target = _monotonic_func(z_true, scale)

    z_hat = _solve_monotonic(target, scale)

    assert torch.isfinite(z_hat).all()
    torch.testing.assert_close(
        _monotonic_func(z_hat, scale),
        target,
        rtol=1e-7,
        atol=1e-9,
    )
    torch.testing.assert_close(z_hat, z_true, rtol=1e-7, atol=1e-9)


def test_joint_bisection_newton_compiles_fullgraph():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    z_true = torch.tensor([[-0.5], [0.25]], dtype=torch.float32)
    scale = torch.tensor([[0.2], [0.5]], dtype=torch.float32)
    target = _monotonic_func(z_true, scale)

    compiled = torch.compile(
        _solve_monotonic_compile_probe,
        backend="eager",
        fullgraph=True,
    )
    z_hat = compiled(target, scale)

    assert z_hat.shape == target.shape
    assert torch.isfinite(z_hat).all()


def test_joint_bisection_newton_falls_back_when_newton_step_leaves_bracket():
    def func(z):
        return z.pow(3)

    def joint_func(z):
        return z.pow(3), 3.0 * z.pow(2)

    target = torch.tensor([[0.0], [0.125], [8.0]], dtype=torch.float32)
    expected = torch.tensor([[0.0], [0.5], [2.0]], dtype=torch.float32)

    result = bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad(
        func,
        joint_func,
        target,
        min_boundary=-3.0,
        max_boundary=3.0,
        num_bisection_iter=20,
        num_newton_iter=20,
    )

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
