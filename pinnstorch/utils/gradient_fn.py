from torch import autograd, ones_like


def gradient(dy, dx, ones_like_tensor=None, create_graph=True):
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
    v = ones_like(dy, requires_grad=True)
    g = gradient(dy, dx, v)
    return gradient(g, v)
