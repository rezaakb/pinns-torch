from typing import Dict, List, Union, Optional, Tuple

import torch

def sse(loss: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        target: Union[Dict[str, torch.Tensor], None] = None,
        keys: Union[List[str], None] = None,
        mid: Union[int, None] = None) -> torch.Tensor:
    """Calculate the sum of squared errors (SSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated SSE loss.
    """
    
    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + torch.sum(torch.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + torch.sum(torch.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + torch.sum(torch.square(preds[key] - target[key]))

    return loss

def mse(loss: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        target: Union[Dict[str, torch.Tensor], None] = None,
        keys: Union[List[str], None] = None,
        mid: Union[int, None] = None) -> torch.Tensor:
    """Calculate the mean squared error (MSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated MSE loss.
    """

    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + torch.mean(torch.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + torch.mean(torch.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + torch.mean(torch.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds, target):
    """Calculate the relative L2 error between predictions and target tensors.

    :param preds: Predicted tensors.
    :param target: Target tensors.
    :return: Relative L2 error value.
    """
    l2_norm_error = torch.mean((preds - target)**2)
    l2_norm_target = torch.mean((target)**2)
    #print(l2_norm_error, l2_norm_target, relative_l2)
    if l2_norm_target == 0:
        return l2_norm_error
    return torch.sqrt(l2_norm_error/l2_norm_target)


def fix_extra_variables(extra_variables):
    """Convert extra variables to torch tensors with gradient tracking. These variables are
    trainables in inverse problems.

    :param extra_variables: Dictionary of extra variables to be converted.
    :return: Dictionary of converted extra variables as torch tensors with gradients.
    """
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


def set_requires_grad(x: List[torch.Tensor],
                      t: torch.Tensor,
                      enable_grad: bool=True) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Set the requires_grad attribute for tensors in the input list.

    :param x: List of tensors to modify requires_grad attribute.
    :param t: Tensor to modify requires_grad attribute.
    :param enable_grad: Boolean indicating whether to enable requires_grad or not.
    :return: Modified list of tensors and tensor.
    """
    if t is not None:
        t = t.requires_grad_(enable_grad)
    x = [x_.requires_grad_(enable_grad) for x_ in x]

    return x, t

def fix_predictions(preds_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """It will concatenate predictions for each solution 
    into a tensor and return a dictionary including them.

    :param preds_list: List of dictionary of predictions.
    :return: A dictionary containing the predictions.
    """
    preds_dict = {}
    for preds in preds_list:
        for sol_key, pred in preds.items():
            if sol_key in preds_dict.keys():
                preds_dict[sol_key] = torch.cat((preds_dict[sol_key], pred.detach()), 0)
            else:
                preds_dict[sol_key] = pred.detach()
    return preds_dict


