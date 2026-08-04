import torch


def _legacy_bisection_bracket(
        func,
        target_arg,
        args,
        min_boundary,
        max_boundary,
        num_bisection_iter):
    """Run the legacy bisection update order and return its final state."""
    upper = torch.full_like(target_arg, max_boundary)
    lower = torch.full_like(target_arg, min_boundary)
    midpoint = (upper + lower) / 2.0

    for _ in range(num_bisection_iter):
        midpoint = (upper + lower) / 2.0
        midpoint_value = func(midpoint, *args)

        right_part = (midpoint_value < target_arg).to(target_arg.dtype)
        left_part = 1.0 - right_part
        correct_part = (
            torch.abs(midpoint_value - target_arg)
            <= 1e-6 * torch.abs(target_arg)
        ).to(target_arg.dtype)

        lower = (
            (1.0 - correct_part)
            * (right_part * midpoint + left_part * lower)
            + correct_part * midpoint
        )
        upper = (
            (1.0 - correct_part)
            * (right_part * upper + left_part * midpoint)
            + correct_part * midpoint
        )

    return lower, upper, midpoint


def _safeguarded_newton_candidate(
        func,
        prev,
        fn_result,
        derivative,
        target_arg,
        args,
        active,
        lower,
        upper,
        residual_tolerance):
    """Return a legacy Newton update unless it fails to improve the residual."""
    residual = fn_result - target_arg
    safe_residual = torch.where(
        active & torch.isfinite(residual),
        residual,
        torch.zeros_like(residual),
    )
    valid_derivative = (
        active
        & torch.isfinite(derivative)
        & (derivative != 0)
    )
    safe_derivative = torch.where(
        valid_derivative,
        derivative,
        torch.ones_like(derivative),
    )

    update = safe_residual / safe_derivative
    newton_candidate = prev - update

    finite_value = active & torch.isfinite(fn_result)
    move_lower = fn_result < target_arg
    lower = torch.where(finite_value & move_lower, prev, lower)
    upper = torch.where(finite_value & ~move_lower, prev, upper)
    fallback = (lower + upper) / 2.0

    finite_candidate = torch.isfinite(newton_candidate)
    safe_candidate = torch.where(
        finite_candidate,
        newton_candidate,
        fallback,
    )
    candidate_value = func(safe_candidate, *args)
    candidate_residual = candidate_value - target_arg

    candidate_moved = newton_candidate != prev
    material_residual = torch.abs(residual) > residual_tolerance
    improves_residual = (
        torch.isfinite(candidate_residual)
        & (torch.abs(candidate_residual) < torch.abs(residual))
    )
    unsafe = active & (
        ~finite_candidate
        | ~valid_derivative
        | (~improves_residual & material_residual)
        | (~candidate_moved & material_residual)
    )

    candidate = torch.where(unsafe, fallback, newton_candidate)
    return candidate, update, unsafe, lower, upper


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
    """Compile-friendly equivalent of the legacy joint Newton solver.

    Normal finite Newton iterations intentionally retain the legacy arithmetic,
    initialization point, row-level convergence mask, and autograd path. A
    bisection fallback is used only when a Newton candidate is non-finite,
    fails to improve a material residual, or cannot move despite one.
    """
    del verbose

    lower, upper, prev = _legacy_bisection_bracket(
        func,
        target_arg,
        args,
        min_boundary,
        max_boundary,
        num_bisection_iter,
    )
    active_row = torch.ones(
        target_arg.shape[0],
        dtype=torch.bool,
        device=target_arg.device,
    )
    residual_tolerance = 1e-7 if target_arg.dtype == torch.float64 else 1e-4

    for _ in range(num_newton_iter):
        fn_result, derivative = joint_func(prev, *args)
        active = active_row.unsqueeze(-1)
        candidate, update, unsafe, lower, upper = (
            _safeguarded_newton_candidate(
                func,
                prev,
                fn_result,
                derivative,
                target_arg,
                args,
                active,
                lower,
                upper,
                residual_tolerance,
            )
        )
        prev = torch.where(active, candidate, prev)

        unsafe_row = unsafe.reshape(unsafe.shape[0], -1).any(dim=1)
        legacy_still_active = (
            torch.abs(update).reshape(update.shape[0], -1).sum(dim=1)
            >= newton_tolerance
        )
        still_active = torch.where(
            unsafe_row,
            torch.ones_like(legacy_still_active),
            legacy_still_active,
        )
        active_row = active_row & still_active

    return prev


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
    """Compile-friendly safeguarded equivalent of the legacy split solver."""
    del verbose

    lower, upper, prev = _legacy_bisection_bracket(
        func,
        target_arg,
        args,
        min_boundary,
        max_boundary,
        num_bisection_iter,
    )
    active_row = torch.ones(
        target_arg.shape[0],
        dtype=torch.bool,
        device=target_arg.device,
    )
    residual_tolerance = 1e-7 if target_arg.dtype == torch.float64 else 1e-4

    for _ in range(num_newton_iter):
        fn_result = func(prev, *args)
        derivative = grad_func(prev, *args)
        active = active_row.unsqueeze(-1)
        candidate, update, unsafe, lower, upper = (
            _safeguarded_newton_candidate(
                func,
                prev,
                fn_result,
                derivative,
                target_arg,
                args,
                active,
                lower,
                upper,
                residual_tolerance,
            )
        )
        prev = torch.where(active, candidate, prev)

        unsafe_row = unsafe.reshape(unsafe.shape[0], -1).any(dim=1)
        legacy_still_active = (
            torch.abs(update).reshape(update.shape[0], -1).sum(dim=1)
            >= newton_tolerance
        )
        still_active = torch.where(
            unsafe_row,
            torch.ones_like(legacy_still_active),
            legacy_still_active,
        )
        active_row = active_row & still_active

    return prev
