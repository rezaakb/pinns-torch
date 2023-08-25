from torch import autograd, ones_like


def gradient(dy, dx, ones_like_tensor=None, create_graph=True):
    """
    Compute the gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the gradient for.
    :param dx: The tensor with respect to which the gradient is computed.
    :param ones_like_tensor: A tensor with the same shape as `dy`, used for creating the gradient (default is None).
    :param create_graph: Whether to create a computational graph for higher-order gradients (default is True).
    :return: The gradient of `dy` with respect to `dx`.
    """
    if ones_like_tensor is None:
        ones_like_tensor = ones_like(dy, requires_grad=False)
    dy_dx = autograd.grad(
        dy,
        dx,
        grad_outputs=ones_like_tensor,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )
    if len(dy_dx) == 1:
        dy_dx = dy_dx[0]
    return dy_dx


def fwd_gradient(dy, dx):
    """
    Compute the forward gradient of a tensor `dy` with respect to another tensor `dx`.

    :param dy: The tensor to compute the forward gradient for.
    :param dx: The tensor with respect to which the forward gradient is computed.
    :return: The forward gradient of `dy` with respect to `dx`.
    """
    v = ones_like(dy, requires_grad=True)
    g = gradient(dy, dx, v)
    return gradient(g, v)
