"""Precision requirements for numerically sensitive manifold flows."""

from functools import wraps

import torch


_FLOAT64_ONLY_S2_FLOW_TYPES = frozenset(("n", "v"))


def _float64_only_s2_flow_defs(instance):
    return tuple(
        flow_def
        for pdf_def, flow_def in zip(
            getattr(instance, "pdf_defs_list", ()),
            getattr(instance, "flow_defs_list", ()),
        )
        if pdf_def == "s2"
        and any(flow_type in flow_def for flow_type in _FLOAT64_ONLY_S2_FLOW_TYPES)
    )


def _requires_float64(instance):
    return bool(getattr(instance, "_float64_only_s2_flow_defs", ()))


def _precision_error(instance, received):
    flow_defs = ", ".join(repr(flow_def) for flow_def in instance._float64_only_s2_flow_defs)
    return TypeError(
        "S2 flow definitions containing 'n' or 'v' require torch.float64 "
        f"(configured flow definitions: {flow_defs}); received {received}."
    )


def _validate_dtype(instance, dtype):
    if not _requires_float64(instance) or dtype is None:
        return

    if getattr(dtype, "is_floating_point", False) and dtype != torch.float64:
        raise _precision_error(instance, dtype)


def _validate_tensors(instance, value):
    if not _requires_float64(instance):
        return

    if isinstance(value, torch.dtype):
        _validate_dtype(instance, value)
        return

    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype != torch.float64:
            raise _precision_error(instance, value.dtype)
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            _validate_tensors(instance, nested_value)
        return

    if isinstance(value, (list, tuple)):
        for nested_value in value:
            _validate_tensors(instance, nested_value)


def _wrap_tensor_entrypoint(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if _requires_float64(self):
            _validate_tensors(self, args)
            _validate_tensors(self, kwargs)
        return method(self, *args, **kwargs)

    return wrapped


def _wrap_forbidden_conversion(method_name, method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if _requires_float64(self):
            raise _precision_error(self, method_name)
        return method(self, *args, **kwargs)

    return wrapped


def apply_s2_float64_policy(pdf_class):
    """Apply the S2 ``n``/``v`` float64 contract to a PDF class once."""
    if getattr(pdf_class, "_s2_float64_policy_applied", False):
        return

    original_init = pdf_class.__init__

    @wraps(original_init)
    def precision_aware_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._float64_only_s2_flow_defs = _float64_only_s2_flow_defs(self)

        # Establish a stable invariant immediately after construction. This is
        # intentionally not an operation-local cast: the model, parameters,
        # inputs, and outputs are expected to remain float64 throughout use.
        if self._float64_only_s2_flow_defs:
            torch.nn.Module.double(self)

    pdf_class.__init__ = precision_aware_init

    for method_name in (
        "forward",
        "all_layer_forward",
        "all_layer_inverse",
        "sample",
        "obtain_flow_param_structure",
        "transform_target_space",
        "init_params",
        "to",
    ):
        method = getattr(pdf_class, method_name, None)
        if method is not None:
            setattr(pdf_class, method_name, _wrap_tensor_entrypoint(method))

    for method_name in ("float", "half", "bfloat16"):
        method = getattr(pdf_class, method_name, None)
        if method is not None:
            setattr(
                pdf_class,
                method_name,
                _wrap_forbidden_conversion(method_name, method),
            )

    pdf_class._s2_float64_policy_applied = True
