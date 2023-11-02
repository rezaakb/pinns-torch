from pinnstorch.utils.gradient_fn import fwd_gradient, gradient
from pinnstorch.utils.instantiators import instantiate_callbacks, instantiate_loggers
from pinnstorch.utils.logging_utils import log_hyperparameters
from pinnstorch.utils.module_fn import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    sse,
    set_requires_grad,
    fix_predictions
)
from pinnstorch.utils.plotting import (
    plot_ac,
    plot_burgers_continuous_forward,
    plot_burgers_continuous_inverse,
    plot_burgers_discrete_forward,
    plot_burgers_discrete_inverse,
    plot_kdv,
    plot_navier_stokes,
    plot_schrodinger,
)
from pinnstorch.utils.pylogger import get_pylogger
from pinnstorch.utils.rich_utils import enforce_tags, print_config_tree
from pinnstorch.utils.utils import (
    extras,
    get_metric_value,
    load_data,
    load_data_txt,
    task_wrapper,
    set_mode
)

from pinnstorch.utils.jit_compiler import jit_compiler