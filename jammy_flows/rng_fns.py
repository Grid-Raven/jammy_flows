import torch
# custom op so that fullgraph compilation traces this as single node


def _seeded_standard_normal(num_samples: int,
                            dim: int,
                            seed: int,
                            dtype: torch.dtype,
                            device: torch.device) -> torch.Tensor:

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    return torch.randn(num_samples, dim, dtype=dtype, device=device, generator=generator)


# custom_op requires torch>=2.4
if(hasattr(torch.library, "custom_op")):
    _seeded_standard_normal = torch.library.custom_op(
        "jammy_flows::seeded_standard_normal", mutates_args=())(_seeded_standard_normal)

    _seeded_standard_normal.register_fake(
        lambda num_samples, dim, seed, dtype, device: torch.empty(num_samples, dim, dtype=dtype, device=device))


def draw_standard_normal(num_samples, dim, dtype, device, seed=None):
    if(seed is None):
        return torch.randn(num_samples, dim, dtype=dtype, device=device)

    return _seeded_standard_normal(num_samples, dim, int(seed), dtype, device)
