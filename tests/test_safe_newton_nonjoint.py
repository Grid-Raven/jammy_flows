import torch

from jammy_flows.layers import bisection_n_newton


def test_nonjoint_newton_uses_bisection_for_unsafe_steps():
    def func(z):
        return z.pow(3)

    def grad_func(z):
        return 3.0 * z.pow(2)

    target = torch.tensor([[0.0], [0.125], [8.0]], dtype=torch.float32)
    expected = torch.tensor([[0.0], [0.5], [2.0]], dtype=torch.float32)

    result = bisection_n_newton.inverse_bisection_n_newton(
        func,
        grad_func,
        target,
        min_boundary=-3.0,
        max_boundary=3.0,
        num_bisection_iter=20,
        num_newton_iter=20,
    )

    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_nonjoint_newton_does_not_accept_tiny_step_with_large_residual():
    def func(z):
        return z

    def deliberately_bad_grad(z):
        return torch.full_like(z, 1e30)

    target = torch.tensor([[0.75]], dtype=torch.float32)
    result = bisection_n_newton.inverse_bisection_n_newton(
        func,
        deliberately_bad_grad,
        target,
        min_boundary=-1.0,
        max_boundary=1.0,
        num_bisection_iter=4,
        num_newton_iter=20,
    )

    torch.testing.assert_close(result, target, rtol=0.0, atol=1e-5)
