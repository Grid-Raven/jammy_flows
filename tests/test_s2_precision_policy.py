import pytest
import torch

import jammy_flows.main.default as f


@pytest.mark.parametrize("flow_def", ["n", "v"])
def test_sensitive_s2_flows_require_float64(flow_def):
    flow = f.pdf("s2", flow_def)

    floating_parameter_dtypes = {
        parameter.dtype
        for parameter in flow.parameters()
        if parameter.is_floating_point()
    }
    assert floating_parameter_dtypes == {torch.float64}

    with pytest.raises(TypeError, match="require torch.float64"):
        flow.float()

    with pytest.raises(TypeError, match="require torch.float64"):
        flow.to(dtype=torch.float32)

    with pytest.raises(TypeError, match="require torch.float64"):
        flow(torch.zeros(2, 2, dtype=torch.float32))


def test_other_s2_flows_keep_float32_support():
    flow = f.pdf("s2", "f")
    flow.float()

    floating_parameter_dtypes = {
        parameter.dtype
        for parameter in flow.parameters()
        if parameter.is_floating_point()
    }
    assert floating_parameter_dtypes == {torch.float32}
