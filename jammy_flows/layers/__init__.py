from . import bisection_n_newton
from . import spline_fns
from ._safe_newton import (
    inverse_bisection_n_newton,
    inverse_bisection_n_newton_joint_func_and_grad,
)
from ._safe_spline_tail import rational_quadratic_spline_with_linear_extension


# Keep the public import paths stable while replacing the unsafe numerical
# dispatch functions. Their original implementations remain available in the
# source modules for compatibility and comparison.
bisection_n_newton.inverse_bisection_n_newton = inverse_bisection_n_newton
bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad = (
    inverse_bisection_n_newton_joint_func_and_grad
)
spline_fns.rational_quadratic_spline_with_linear_extension = (
    rational_quadratic_spline_with_linear_extension
)
