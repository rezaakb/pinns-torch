from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    if cfg.compile:
        log.info("Model will be compiled. Setting optimizer capturable attribute to True.")
        cfg.model.optimizer.capturable = True
        log.info("Model will be compiled. Disabling automatic optimization.")
        cfg.model.automatic_optimization = False
        if len(cfg.trainer.devices) > 1:
            log.info(
                f"DDP is not supported for compiled model. Using device {cfg.trainer.devices[0]}"
            )
            cfg.trainer.devices = cfg.trainer.devices[0]
    else:
        log.info("Model will not be compiled. Setting optimizer capturable attribute to False.")
        cfg.model.optimizer.capturable = False
        log.info("Model will not be compiled. Enabling automatic optimization.")
        cfg.model.automatic_optimization = True

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.get("time_domain"):
        log.info(f"Instantiating time domain <{cfg.time_domain._target_}>")
        td: TimeDomain = hydra.utils.instantiate(cfg.time_domain)

    if cfg.get("spatial_domain"):
        log.info(f"Instantiating spatial domain <{cfg.spatial_domain._target_}>")
        sd: SpatialDomain = hydra.utils.instantiate(cfg.spatial_domain)

    log.info(f"Instantiating mesh <{cfg.mesh._target_}>")
    if cfg.mesh._target_ == "src.data.mesh.mesh.Mesh":
        mesh: Mesh = hydra.utils.instantiate(
            cfg.mesh, time_domain=td, spatial_domain=sd, read_data_fn=read_data_fn
        )
    elif cfg.mesh._target_ == "src.data.mesh.mesh.PointCloud":
        mesh: PointCloud = hydra.utils.instantiate(cfg.mesh, read_data_fn=read_data_fn)
    else:
        raise "Mesh should be defined in config file."

    train_datasets = []
    for dataset_dic in cfg.train_datasets:
        for i, (key, dataset) in enumerate(dataset_dic.items()):
            log.info(f"Instantiating training dataset number {i+1}: <{dataset._target_}>")
            train_datasets.append(hydra.utils.instantiate(dataset)(mesh=mesh))

    val_dataset = None
    if cfg.get("val_dataset"):
        for dataset_dic in cfg.val_dataset:
            for i, (key, dataset) in enumerate(dataset_dic.items()):
                log.info(f"Instantiating validation dataset number {i+1}: <{dataset._target_}>")
                val_dataset = hydra.utils.instantiate(dataset)(mesh=mesh)

    test_dataset = None
    if cfg.get("test_dataset"):
        for dataset_dic in cfg.test_dataset:
            for i, (key, dataset) in enumerate(dataset_dic.items()):
                log.info(f"Instantiating test dataset number {i+1}: <{dataset._target_}>")
                test_dataset = hydra.utils.instantiate(dataset)(mesh=mesh)

    pred_dataset = None
    if cfg.get("pred_dataset"):
        for dataset_dic in cfg.pred_dataset:
            for i, (key, dataset) in enumerate(dataset_dic.items()):
                log.info(f"Instantiating prediction dataset number {i+1}: <{dataset._target_}>")
                pred_dataset = hydra.utils.instantiate(dataset)(mesh=mesh)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_datasets=train_datasets,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pred_dataset=pred_dataset,
        batch_size=cfg.get("batch_size"),
    )

    if cfg.net._target_ == "src.models.net.neural_net.FCN":
        log.info(f"Instantiating neural net <{cfg.net._target_}>")
        net: torch.nn.Module = hydra.utils.instantiate(cfg.net)(lb=mesh.lb, ub=mesh.ub)
    elif cfg.net._target_ == "src.models.net.neural_net.NetHFM":
        # TODO
        log.info(f"Instantiating neural net <{cfg.net._target_}>")
        net: torch.nn.Module = hydra.utils.instantiate(cfg.net)(
            mean=train_datasets[0].mean, std=train_datasets[0].std
        )

    log.info(f"Instantiating model <{cfg.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model)(
        net=net, pde_fn=pde_fn, output_fn=output_fn
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
