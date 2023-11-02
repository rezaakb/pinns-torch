import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Tuple

import numpy as np
import requests
import scipy
from omegaconf import DictConfig, open_dict

from pinnstorch.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(
        cfg: DictConfig, read_data_fn: Callable, pde_fn: Callable, output_fn: Callable
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(
                cfg=cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=output_fn
            )

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_names: list) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: The name of the metric to retrieve.
    :return: The value of the metric.
    """
    for type_metric, list_metrics in metric_names.items():
        if type_metric == "extra_variables":
            prefix = ""
        elif type_metric == "error":
            prefix = "val/error_"

        for metric_name in list_metrics:
            metric_name = f"{prefix}{metric_name}"

            if not metric_name:
                log.info("Metric name is None! Skipping metric value retrieval...")
                continue

            if metric_name not in metric_dict:
                log.info(
                    f"Metric value not found! <metric_name={metric_name}>\n"
                    "Make sure metric name logged in LightningModule is correct!\n"
                    "Make sure `optimized_metric` name in `hparams_search` config is correct!"
                )
            else:
                metric_value = metric_dict[metric_name].item()
                log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def download_file(path, folder_name, filename):
    """Download a file from a given URL and save it to the specified path.

    :param path: Path where the file should be saved.
    :param folder_name: Name of the folder containing the file on the server.
    :param filename: Name of the file to be downloaded.
    """

    url = f"https://storage.googleapis.com/pinns/{folder_name}/{filename}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(response.content)
        log.info("File downloaded successfully.")
    else:
        FileNotFoundError("File download failed.")


def load_data_txt(root_path, file_name):
    """Load text data from a file, downloading it if not already present.

    :param root_path: The root directory where the data file should be located.
    :param file_name: Name of the data file.
    :return: Loaded data as a numpy array.
    """
    path = os.path.join(root_path, file_name)
    if os.path.exists(path):
        log.info("Weights are available.")
    else:
        download_file(path, "irk_weights", file_name)

    return np.float32(np.loadtxt(path, ndmin=2))


def load_data(root_path, file_name):
    """Load data from a MATLAB .mat file, downloading it if not already present.

    :param root_path: The root directory where the data file should be located.
    :param file_name: Name of the data file.
    :return: Loaded data using scipy.io.loadmat function.
    """

    path = os.path.join(root_path, file_name)
    if os.path.exists(path):
        log.info("Data is available.")
    else:
        download_file(path, "data", file_name)

    return scipy.io.loadmat(path)


def set_mode(cfg):
    import torch
    
    if not torch.cuda.is_available():
        log.info("GPU is not found. Using CPU.")
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    if cfg.model.lazy:
        log.info("Using LazyTensor as backend.")
    
    if cfg.model.cudagraph_compile and cfg.trainer.accelerator != "cpu":
        log.info("Model will be compiled.")
        log.info("Setting optimizer capturable attribute to True.")
        cfg.model.optimizer.capturable = True
        log.info("Disabling automatic optimization.")
        if cfg.trainer.devices is not None and isinstance(cfg.trainer.devices, list):
            if len(cfg.trainer.devices) > 1:
                log.info(
                    f"DDP is not supported for compiled model. Using device {cfg.trainer.devices[0]}"
                )
                cfg.trainer.devices = [cfg.trainer.devices[0]]
        
    elif not cfg.model.cudagraph_compile and cfg.trainer.accelerator != "cpu": 
        log.info("Model will not be compiled.")
        log.info("Setting optimizer capturable attribute to False.")
        cfg.model.optimizer.capturable = False
        log.info("Enabling automatic optimization.")
        if cfg.model.amp:
            with open_dict(cfg):
                cfg.trainer.precision = 16

    elif cfg.trainer.accelerator == "cpu":
        log.info("Model will not be compiled with CUDA Graph.")
        cfg.model.cudagraph_compile = False
        log.info("Setting optimizer capturable attribute to False.")
        cfg.model.optimizer.capturable = False
        log.info("Enabling automatic optimization.")
    
    return cfg