import torch


def _inverse_bisection_newton(
        value_func,
        value_and_grad_func,
        target_arg,
        min_boundary,
        max_boundary,
        num_bisection_iter,
        num_newton_iter,
        newton_tolerance):
    """Compiler-friendly safeguarded Newton iteration.

    The full tensor shape is retained for every iteration. Newton candidates
    are accepted only when finite and inside the current bisection bracket;
    otherwise the iteration falls back to the bracket midpoint.
    """
    lower = torch.full_like(target_arg, min_boundary)
    upper = torch.full_like(target_arg, max_boundary)

    for _ in range(num_bisection_iter):
        midpoint = (lower + upper) * 0.5
        midpoint_value = value_func(midpoint)
        move_lower = midpoint_value < target_arg
        lower = torch.where(move_lower, midpoint, lower)
        upper = torch.where(move_lower, upper, midpoint)

    current = (lower + upper) * 0.5
    active = torch.ones_like(target_arg, dtype=torch.bool)

    finfo = torch.finfo(target_arg.dtype)
    effective_tolerance = max(
        float(newton_tolerance),
        10.0 * float(finfo.eps),
    )
    derivative_floor = 10.0 * float(finfo.tiny)

    for _ in range(num_newton_iter):
        value, derivative = value_and_grad_func(current)
        residual = value - target_arg

        finite_value = torch.isfinite(value)
        finite_residual = torch.isfinite(residual)
        valid_derivative = (
            torch.isfinite(derivative)
            & (torch.abs(derivative) > derivative_floor)
        )

        update_bracket = active & finite_value
        move_lower = value < target_arg
        lower = torch.where(update_bracket & move_lower, current, lower)
        upper = torch.where(update_bracket & ~move_lower, current, upper)
        midpoint = (lower + upper) * 0.5

        safe_residual = torch.where(
            finite_residual,
            residual,
            torch.zeros_like(residual),
        )
        safe_derivative = torch.where(
            valid_derivative,
            derivative,
            torch.ones_like(derivative),
        )
        newton_candidate = current - safe_residual / safe_derivative
        newton_step = torch.abs(newton_candidate - current)

        valid_candidate = (
            finite_residual
            & valid_derivative
            & torch.isfinite(newton_candidate)
            & (newton_candidate >= lower)
            & (newton_candidate <= upper)
            & (newton_step > effective_tolerance)
        )
        candidate = torch.where(
            valid_candidate,
            newton_candidate,
            midpoint,
        )

        # A tiny step is not sufficient evidence of convergence: float32 can
        # stop moving while the function residual is still large.
        converged = finite_residual & (
            torch.abs(residual) <= effective_tolerance
        )

        current = torch.where(active, candidate, current)
        active = active & ~converged

    # The last candidate has not yet been evaluated. Choose between it and the
    # final bracket midpoint using the smaller finite residual.
    midpoint = (lower + upper) * 0.5
    current_value = value_func(current)
    midpoint_value = value_func(midpoint)
    current_error = torch.abs(current_value - target_arg)
    midpoint_error = torch.abs(midpoint_value - target_arg)

    current_error = torch.where(
        torch.isfinite(current_error),
        current_error,
        torch.full_like(current_error, float("inf")),
    )
    midpoint_error = torch.where(
        torch.isfinite(midpoint_error),
        midpoint_error,
        torch.full_like(midpoint_error, float("inf")),
    )
    return torch.where(midpoint_error < current_error, midpoint, current)


def inverse_bisection_n_newton_joint_func_and_grad(
        func,
        joint_func,
        target_arg,
        *args,
        min_boundary=-100000.0,
        max_boundary=100000.0,
        num_bisection_iter=25,
        num_newton_iter=30,
        newton_tolerance=1e-14,
        verbose=0):
    del verbose

    def value_func(z):
        return func(z, *args)

    def value_and_grad_func(z):
        return joint_func(z, *args)

    return _inverse_bisection_newton(
        value_func,
        value_and_grad_func,
        target_arg,
        min_boundary,
        max_boundary,
        num_bisection_iter,
        num_newton_iter,
        newton_tolerance,
    )


def inverse_bisection_n_newton(
        func,
        grad_func,
        target_arg,
        *args,
        min_boundary=-100000.0,
        max_boundary=100000.0,
        num_bisection_iter=25,
        num_newton_iter=30,
        newton_tolerance=1e-14,
        verbose=0):
    del verbose

    def value_func(z):
        return func(z, *args)

    def value_and_grad_func(z):
        return func(z, *args), grad_func(z, *args)

    return _inverse_bisection_newton(
        value_func,
        value_and_grad_func,
        target_arg,
        min_boundary,
        max_boundary,
        num_bisection_iter,
        num_newton_iter,
        newton_tolerance,
    )
