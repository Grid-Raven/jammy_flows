from . import bisection_n_newton
from . import spline_fns
from .spheres import moebius_1d
from ._safe_newton import (
    inverse_bisection_n_newton_bracketed,
    inverse_bisection_n_newton_joint_func_and_grad,
)
from ._safe_spline_tail import rational_quadratic_spline_with_linear_extension


# Keep public import paths stable while replacing only the compile-targeted
# joint solver, the numerically sensitive Moebius inverse, and unsafe spline
# tail dispatch. Other users of the legacy split Newton solver remain unchanged.
bisection_n_newton.inverse_bisection_n_newton_joint_func_and_grad = (
    inverse_bisection_n_newton_joint_func_and_grad
)
moebius_1d.inverse_bisection_n_newton = inverse_bisection_n_newton_bracketed
spline_fns.rational_quadratic_spline_with_linear_extension = (
    rational_quadratic_spline_with_linear_extension
)
