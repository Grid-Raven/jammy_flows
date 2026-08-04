import torch
from torch.nn import functional as F

from . import spline_fns as _spline_fns


_original_rational_quadratic_spline_with_linear_extension = (
    _spline_fns.rational_quadratic_spline_with_linear_extension
)


def rational_quadratic_spline_with_linear_extension(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        left=torch.DoubleTensor([[[0.0]]]),
        right=torch.DoubleTensor([[[1.0]]]),
        bottom=torch.DoubleTensor([[[0.0]]]),
        top=torch.DoubleTensor([[[1.0]]]),
        rel_min_bin_width=1e-3,
        rel_min_bin_height=1e-3,
        min_derivative=1e-3):
    """Evaluate a rational-quadratic spline with safe linear tails.

    Tail inputs are clamped before the interior spline is evaluated. This keeps
    invalid values in the unselected interior branch from contaminating
    gradients through ``torch.where``.
    """
    original_inputs = inputs
    if inverse:
        interior_inputs = torch.maximum(torch.minimum(inputs, top), bottom)
    else:
        interior_inputs = torch.maximum(torch.minimum(inputs, right), left)

    outputs, logabsdet = (
        _original_rational_quadratic_spline_with_linear_extension(
            interior_inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=inverse,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            rel_min_bin_width=rel_min_bin_width,
            rel_min_bin_height=rel_min_bin_height,
            min_derivative=min_derivative,
        )
    )

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    left_derivative = derivatives[..., 0:1]
    right_derivative = derivatives[..., -1:]

    if inverse:
        left_offset = left - bottom / left_derivative
        right_offset = right - top / right_derivative

        outputs = torch.where(
            original_inputs <= bottom,
            original_inputs / left_derivative + left_offset,
            outputs,
        )
        outputs = torch.where(
            original_inputs >= top,
            original_inputs / right_derivative + right_offset,
            outputs,
        )
        logabsdet = torch.where(
            original_inputs <= bottom,
            -torch.log(left_derivative),
            logabsdet,
        )
        logabsdet = torch.where(
            original_inputs >= top,
            -torch.log(right_derivative),
            logabsdet,
        )
    else:
        left_offset = bottom - left * left_derivative
        right_offset = top - right * right_derivative

        outputs = torch.where(
            original_inputs <= left,
            original_inputs * left_derivative + left_offset,
            outputs,
        )
        outputs = torch.where(
            original_inputs >= right,
            original_inputs * right_derivative + right_offset,
            outputs,
        )
        logabsdet = torch.where(
            original_inputs <= left,
            torch.log(left_derivative),
            logabsdet,
        )
        logabsdet = torch.where(
            original_inputs >= right,
            torch.log(right_derivative),
            logabsdet,
        )

    return outputs, logabsdet
