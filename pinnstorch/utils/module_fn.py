import torch


def sse(loss, preds, target=None, keys=None, mid=None):
    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + torch.sum(torch.square(preds[key]))
        elif mid is not None:
            loss = loss + torch.sum(torch.square(preds[key][:mid] - preds[key][mid:]))
        else:
            loss = loss + torch.sum(torch.square(preds[key] - target[key]))

    return loss


def mse(loss, preds, target=None, keys=None, mid=None):
    if keys is None:
        return loss

    for key in keys:
        if target is None:
            loss = loss + torch.mean(torch.square(preds[key]))
        elif mid is not None:
            loss = loss + torch.sum(torch.square(preds[key][:mid] - preds[key][mid:]))
        else:
            loss = loss + torch.mean(torch.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds, target):
    return torch.norm(preds - target, p=2) / torch.norm(target, p=2)


def fix_extra_variables(extra_variables):
    if extra_variables is None:
        return None
    extra_variables_parameters = {}
    for key in extra_variables:
        extra_variables_parameters[key] = torch.tensor(
            [extra_variables[key]], dtype=torch.float32, requires_grad=True
        )
    extra_variables_parameters = torch.nn.ParameterDict(extra_variables_parameters)
    return extra_variables_parameters


def requires_grad(batch, enable_grad=True):
    """Set the requires_grad attribute for tensors in the input list.

    :param batch: A batch containing spatial, time, solutions.
    :param enable_grad: Boolean indicating whether to enable requires_grad or not.
    :return: Modified list of tensors and tensor.
    """
    spatial, time, solutions = batch
    if time is not None:
        time = time.requires_grad_(enable_grad)
    spatial = [x_.requires_grad_(enable_grad) for x_ in spatial]
    solutions = {
        solution_name: solutions[solution_name].requires_grad_(enable_grad)
        for solution_name in solutions.keys()
    }

    return spatial, time, solutions
