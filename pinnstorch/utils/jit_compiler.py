from typing import Tuple, List, Dict

import torch

from torch.utils._python_dispatch import _disable_current_modes
from torch._subclasses.fake_tensor import FakeTensor

import functools

def defake(x):
    """
    Convert FakeTensor instances to real, zero-filled tensors with the same meta-data.

    If the input is not a FakeTensor, it is returned unmodified. Otherwise, the function creates a 
    zero tensor with the same size, stride, dtype, device, and requires_grad properties as the FakeTensor. 
    Symbolic sizes and strides are resolved to concrete values if necessary.

    :param x: The input tensor, which can either be a FakeTensor or a regular tensor.
    :return: A zero-filled tensor with the same properties as the input, or the unmodified input if it's not a FakeTensor.
    """
    
    if not isinstance(x, FakeTensor):
        return x
    if x._has_symbolic_sizes_strides:
        size = [
            s.node.shape_env.size_hint(s.node.expr)
            if isinstance(s, torch.SymInt)
            else s
            for s in x.size()
        ]
        stride = [
            s.node.shape_env.size_hint(s.node.expr)
            if isinstance(s, torch.SymInt)
            else s
            for s in x.stride()
        ]
    else:
        size = x.size()
        stride = x.stride()
    y = torch.empty_strided(
        size,
        stride,
        dtype=x.dtype,
        device=x.device,
        requires_grad=x.requires_grad,
    )
    y.requires_grad_(False).zero_()
    return y

def fake_tensor_unsupported(fn):
    """
    A decorator for functions that are incompatible with FakeTensors. 

    This decorator ensures that any FakeTensors in the inputs are converted to real tensors filled with zeros 
    before passing them to the wrapped function. It disables the current dispatch modes temporarily to perform 
    this conversion.

    :param fn: The function to be wrapped.
    :return: The wrapped function that can handle real tensors only.
    """

    @functools.wraps(fn)
    def wrapper(model, inputs, **kwargs):
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)

    return wrapper        


def strip_overloads(gm):
    """
    Update the targets of graph nodes in the given Fx graph module to remove overloads.

    Iterates through all nodes in the graph, and if the node target is an OpOverload instance, 
    replaces it with the original operator.

    :param gm: The input Fx graph module containing the nodes to be updated.
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

from contextlib import contextmanager

@contextmanager
def _disable_jit_autocast():
    """
    A context manager to temporarily disable JIT autocast mode.

    This is used to turn off autocasting during the JIT compilation process, ensuring that operations are 
    not automatically casted to different data types to improve performance. It saves the current state of the 
    JIT autocast flag, disables autocasting, and then restores the flag to its original state after exiting 
    the context.

    """
    
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
    try:
        yield
    finally:
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)


@fake_tensor_unsupported
def jit_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    Compile the given Fx graph module using JIT, handling fake tensors and overloads appropriately.

    This function, wrapped with the @fake_tensor_unsupported decorator, ensures that the graph module is 
    compatible with real tensors only. It disables JIT autocast, strips overloads, and updates graph nodes 
    with certain conditions. Finally, it uses JIT trace to compile the graph module and performs additional 
    optimizations.

    :param gm: The Fx graph module to be compiled.
    :param example_inputs: A list of example inputs to be used during the tracing process.
    :return: The compiled JIT function.
    """
    
    with _disable_jit_autocast():
        strip_overloads(gm)

        for node in gm.graph.nodes:
            if (
                node.target == torch.ops.aten._to_copy
                and len(node.args) == 1
                and len(node.kwargs) == 1
                and "dtype" in node.kwargs
            ):
                node.target = torch.ops.aten.to

        for node in gm.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs

        gm.graph.lint()
        gm.recompile()

        fn = torch.jit.trace(gm, example_inputs)
        torch._C._te.remove_unused_self_argument(fn.graph)
        torch._C._jit_pass_remove_mutation(fn.graph)

    return fn