import torch


def _seeded_standard_normal_impl(num_samples: int,
                                 dim: int,
                                 seed: int,
                                 dtype: torch.dtype,
                                 device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")

    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    result = torch.randn(
        num_samples,
        dim,
        dtype=dtype,
        device=generator_device,
        generator=generator,
    )

    if generator_device != device:
        result = result.to(device)

    return result


_seeded_standard_normal = _seeded_standard_normal_impl

# torch.library and custom_op are unavailable on older supported PyTorch versions.
_library = getattr(torch, "library", None)
_custom_op = getattr(_library, "custom_op", None) if _library is not None else None

if _custom_op is not None:
    _seeded_standard_normal = _custom_op(
        "jammy_flows::seeded_standard_normal",
        mutates_args=(),
    )(_seeded_standard_normal_impl)

    register_fake = getattr(_seeded_standard_normal, "register_fake", None)
    if register_fake is not None:
        register_fake(
            lambda num_samples, dim, seed, dtype, device: torch.empty(
                num_samples,
                dim,
                dtype=dtype,
                device=device,
            )
        )


def draw_standard_normal(num_samples, dim, dtype, device, seed=None):
    if seed is None:
        return torch.randn(num_samples, dim, dtype=dtype, device=device)

    return _seeded_standard_normal(num_samples, dim, int(seed), dtype, device)
