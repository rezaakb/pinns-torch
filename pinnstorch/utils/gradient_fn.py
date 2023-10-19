from typing import Dict, List, Union, Optional, Tuple

import torch 
import torch._dynamo

@torch.jit.script
def gradient(dy: torch.Tensor,
             dx: Union[List[torch.Tensor], torch.Tensor],
             ones_like_tensor: Optional[List[Optional[torch.Tensor]]] = None,
             create_graph: bool = True) -> List[torch.Tensor]:
    """Compute the gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the gradient for.
    :param dx: The tensor with respect to which the gradient is computed.
    :param ones_like_tensor: A tensor with the same shape as `dy`, used for creating the gradient (default is None).
    :param create_graph: Whether to create a computational graph for higher-order gradients (default is True).
    :return: The gradient of `dy` with respect to `dx`.
    """
    if ones_like_tensor is None:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(dy)]
    else:
        grad_outputs = ones_like_tensor

    if isinstance(dx, torch.Tensor):
        dx = [dx]

    dy_dx = torch.autograd.grad(
        [dy],
        dx,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )
    
    grads = [grad if grad is not None else torch.zeros_like(dx[i]) for i, grad in enumerate(dy_dx)]
    return grads

@torch.jit.script
def fwd_gradient(dy: torch.Tensor, 
                 dx: Union[List[torch.Tensor], torch.Tensor],
                 create_graph: bool = True) -> List[torch.Tensor]:
    """Compute the forward gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the forward gradient for.
    :param dx: The tensor with respect to which the forward gradient is computed.
    :return: The forward gradient of `dy` with respect to `dx`.
    """
        
    if isinstance(dx, torch.Tensor):
        dx = [dx]
    
    ones_like: List[Optional[torch.Tensor]] = [torch.ones_like(dy).requires_grad_(True)]
        
    grads = gradient(dy, dx, ones_like)

    fwd_grads: List[torch.Tensor] = []
    for i, grad in enumerate(grads):
        
        ones_like_ = ones_like[0]
        assert ones_like_ is not None
        
        fwd_grad = gradient(grad, ones_like_, create_graph=create_graph)[0]
        
        if isinstance(fwd_grad, torch.Tensor):
            fwd_grads.append(fwd_grad)
    
    return fwd_grads