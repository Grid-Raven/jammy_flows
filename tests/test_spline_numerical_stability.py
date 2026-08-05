import torch

from jammy_flows.layers.spline_fns import (
    rational_quadratic_spline,
    rational_quadratic_spline_with_linear_extension,
    searchsorted,
)


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


def test_searchsorted_does_not_mutate_bin_locations():
    bin_locations = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
    original = bin_locations.clone()

    searchsorted(bin_locations, torch.tensor([[0.25]], dtype=torch.float64))
    searchsorted(bin_locations, torch.tensor([[0.75]], dtype=torch.float64))

    torch.testing.assert_allclose(bin_locations, original, rtol=0.0, atol=0.0)


def test_rational_quadratic_spline_logdet_matches_finite_difference_for_tiny_valid_derivative():
    x = torch.tensor([[1e-7]], dtype=torch.float64)
    eps = torch.tensor(1e-8, dtype=torch.float64)

    _, logdet = _eval_extreme_valid_spline(x)
    y_plus, _ = _eval_extreme_valid_spline(x + eps)
    y_minus, _ = _eval_extreme_valid_spline(x - eps)
    finite_difference_derivative = (y_plus - y_minus) / (2.0 * eps)

    assert torch.isfinite(logdet).all()
    assert torch.isfinite(finite_difference_derivative).all()
    assert logdet.item() < -8.0
    torch.testing.assert_allclose(
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
    unnormalized_derivatives = torch.tensor(
        [[0.2, -0.3, 0.8, -0.1, 0.5]],
        dtype=dtype,
    )

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

    torch.testing.assert_allclose(x_roundtrip, x, rtol=1e-7, atol=1e-9)
    torch.testing.assert_allclose(
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
    torch.testing.assert_allclose(recovered, inputs.detach(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_allclose(
        forward_logdet.detach() + inverse_logdet.detach(),
        torch.zeros_like(forward_logdet),
        rtol=2e-4,
        atol=2e-4,
    )
