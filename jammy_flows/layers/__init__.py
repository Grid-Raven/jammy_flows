from . import spline_fns
from ._safe_spline_tail import rational_quadratic_spline_with_linear_extension


# Keep the public import path stable while replacing only the unsafe tail
# dispatch. The original implementation is captured by _safe_spline_tail
# before this assignment, so its interior spline calculation is unchanged.
spline_fns.rational_quadratic_spline_with_linear_extension = (
    rational_quadratic_spline_with_linear_extension
)
