import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jammy_flows.main.default as f
from jammy_flows.layers import bisection_n_newton
from jammy_flows.layers.spline_fns import (
    rational_quadratic_spline,
    rational_quadratic_spline_with_linear_extension,
)
from jammy_flows.rng_fns import draw_standard_normal


def _extreme_valid_spline_params(dtype=torch.float64):
    unnormalized_widths = torch.tensor([[50.0, -50.0]], dtype=dtype)
    unnormalized_heights = torch.tensor([[-50.0, 50.0]], dtype=dtype)
    unnormalized_derivatives = torch.full((1, 3), -50.0, dtype=dtype)
    kwargs = dict(
        rel_min_bin_width=1e-4,
        rel_min_bin_height=1e-4,
        min_derivative=1e-4,
    )
    return unnormalized_widths, unnormalized_heights, unnormalized_derivatives, kwargs


def _eval_extreme_valid_spline(x):
    widths, heights, derivatives, kwargs = _extreme_valid_spline_params(x.dtype)
    return rational_quadratic_spline(
        x,
        widths.clone(),
        heights.clone(),
        derivatives.clone(),
        inverse=False,
        **kwargs,
    )


def test_rational_quadratic_spline_logdet_matches_finite_difference_for_tiny_valid_derivative():
    """Checks logdet against a centered finite-difference derivative."""
    x = torch.tensor([[1e-7]], dtype=torch.float64)
    eps = torch.tensor(1e-8, dtype=torch.float64)

    _, logdet = _eval_extreme_valid_spline(x)
    y_plus, _ = _eval_extreme_valid_spline(x + eps)
    y_minus, _ = _eval_extreme_valid_spline(x - eps)
    finite_difference_derivative = (y_plus - y_minus) / (2.0 * eps)

    assert torch.isfinite(logdet).all()
    assert torch.isfinite(finite_difference_derivative).all()
    assert logdet.item() < -8.0
    torch.testing.assert_close(
        logdet.exp(),
        finite_difference_derivative.abs(),
        rtol=1e-4,
        atol=1e-10,
    )


def test_rational_quadratic_spline_forward_inverse_logdet_cancel_for_interior_points():
    dtype = torch.float64
    x = torch.linspace(0.05, 0.95, 13, dtype=dtype).unsqueeze(1)
    unnormalized_widths = torch.tensor([[0.3, -0.8, 1.1, 0.2]], dtype=dtype)
    unnormalized_heights = torch.tensor([[-0.6, 0.7, 0.1, 0.9]], dtype=dtype)
    unnormalized_derivatives = torch.tensor([[0.2, -0.3, 0.8, -0.1, 0.5]], dtype=dtype)

    y, forward_logdet = rational_quadratic_spline(
        x,
        unnormalized_widths.clone(),
        unnormalized_heights.clone(),
        unnormalized_derivatives.clone(),
        inverse=False,
    )
    x_roundtrip, inverse_logdet = rational_quadratic_spline(
        y,
        unnormalized_widths.clone(),
        unnormalized_heights.clone(),
        unnormalized_derivatives.clone(),
        inverse=True,
    )

    torch.testing.assert_close(x_roundtrip, x, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        rtol=1e-7,
        atol=1e-9,
    )


def test_linear_extension_tail_values_and_gradients_are_finite():
    dtype = torch.float32
    inputs = torch.tensor(
        [[[-100.0]], [[0.5]], [[100.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    unnormalized_widths = torch.tensor([[[0.3, -0.8, 1.1, 0.2]]], dtype=dtype)
    unnormalized_heights = torch.tensor([[[-0.6, 0.7, 0.1, 0.9]]], dtype=dtype)
    unnormalized_derivatives = torch.tensor(
        [[[0.2, -0.3, 0.8, -0.1, 0.5]]],
        dtype=dtype,
    )
    left = torch.tensor([[[0.0]]], dtype=dtype)
    right = torch.tensor([[[1.0]]], dtype=dtype)
    bottom = torch.tensor([[[0.0]]], dtype=dtype)
    top = torch.tensor([[[1.0]]], dtype=dtype)

    outputs, forward_logdet = rational_quadratic_spline_with_linear_extension(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )
    (outputs.sum() + forward_logdet.sum()).backward()

    assert torch.isfinite(outputs).all()
    assert torch.isfinite(forward_logdet).all()
    assert torch.isfinite(inputs.grad).all()

    inverse_inputs = outputs.detach().requires_grad_(True)
    recovered, inverse_logdet = rational_quadratic_spline_with_linear_extension(
        inverse_inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=True,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )
    (recovered.sum() + inverse_logdet.sum()).backward()

    assert torch.isfinite(recovered).all()
    assert torch.isfinite(inverse_logdet).all()
    assert torch.isfinite(inverse_inputs.grad).all()
    torch.testing.assert_close(recovered, inputs.detach(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        forward_logdet.detach() + inverse_logdet.detach(),
        torch.zeros_like(forward_logdet),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float64, 1e-10),
        (torch.float32, 1e-6),
        (torch.float16, 2e-3),
    ],
)
def test_s1_embedding_axes_roundtrip_without_moving_valid_endpoints(dtype, atol):
    flow = f.pdf("s1", "y")
    flow.to(dtype=dtype)

    embedding_points = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=dtype,
    )

    intrinsic, _ = flow.transform_target_space(
        embedding_points,
        transform_from="embedding",
        transform_to="intrinsic",
    )
    roundtrip, _ = flow.transform_target_space(
        intrinsic,
        transform_from="intrinsic",
        transform_to="embedding",
    )

    torch.testing.assert_close(roundtrip, embedding_points, rtol=0.0, atol=atol)


def test_seeded_sampling_does_not_mutate_global_torch_rng_state():
    flow = f.pdf("e1", "x")
    flow.double()

    torch.manual_seed(12345)
    expected_next = torch.rand(8)

    torch.manual_seed(12345)
    flow.sample(samplesize=4, seed=999, dtype=torch.float64, device=torch.device("cpu"))
    actual_next = torch.rand(8)

    torch.testing.assert_close(actual_next, expected_next, rtol=0.0, atol=0.0)


def test_obtain_flow_param_structure_seed_does_not_mutate_global_torch_rng_state():
    flow = f.pdf("e1", "x")
    flow.double()

    torch.manual_seed(12345)
    expected_next = torch.rand(8)

    torch.manual_seed(12345)
    flow.obtain_flow_param_structure(seed=999, dtype=torch.float64, device=torch.device("cpu"))
    actual_next = torch.rand(8)

    torch.testing.assert_close(actual_next, expected_next, rtol=0.0, atol=0.0)


def test_seeded_standard_normal_compiles_fullgraph():
    if not hasattr(torch, "compile") or not hasattr(getattr(torch, "library", None), "custom_op"):
        pytest.skip("fullgraph custom-op support is unavailable in this PyTorch version")

    def generate():
        return draw_standard_normal(
            8,
            3,
            torch.float32,
            torch.device("cpu"),
            seed=123,
        )

    compiled = torch.compile(generate, fullgraph=True)
    torch.testing.assert_close(compiled(), generate(), rtol=0.0, atol=0.0)


def test_joint_bisection_newton_inverse_solves_batched_monotonic_equation():
    def func(z, scale):
        return z + scale * z.pow(3)

    def joint_func(z, scale):
        return func(z, scale), 1.0 + 3.0 * scale * z.pow(2)

    z_true = torch.tensor([[-3.0], [-0.1], [0.0], [0.2], [4.0]], dtype=torch.float64)
    scale = torch.tensor([[0.01], [0.5], [1.0], [0.2], [0.03]], dtype=torch.float64)
    target = func(z_true, scale)

    z_hat = bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad(
        func,
        joint_func,
        target,
        scale,
        min_boundary=-5.0,
        max_boundary=5.0,
        num_bisection_iter=40,
        num_newton_iter=20,
        newton_tolerance=1e-13,
    )

    assert torch.isfinite(z_hat).all()
    torch.testing.assert_close(func(z_hat, scale), target, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(z_hat, z_true, rtol=1e-7, atol=1e-9)


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
