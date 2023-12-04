from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os
import time
import hydra
import lightning as L
import rootutils
import pickle 
import torch
import numpy as np

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pinnstorch import utils
from pinnstorch.data import (
    Interval,
    Mesh,
    PointCloud,
    Rectangle,
    RectangularPrism,
    TimeDomain,
)

from torch.profiler import profile, record_function, ProfilerActivity


log = utils.get_pylogger(__name__)

OmegaConf.register_new_resolver("eval", eval)

@utils.task_wrapper
def train(
    cfg: DictConfig, read_data_fn: Callable, pde_fn: Callable, output_fn: Callable = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    cfg = utils.set_mode(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    if cfg.get("time_domain"):
        log.info(f"Instantiating time domain <{cfg.time_domain._target_}>")
        td: TimeDomain = hydra.utils.instantiate(cfg.time_domain)

    if cfg.get("spatial_domain"):
        log.info(f"Instantiating spatial domain <{cfg.spatial_domain._target_}>")
        sd: Union[Interval, Rectangle, RectangularPrism] = hydra.utils.instantiate(
            cfg.spatial_domain
        )

    log.info(f"Instantiating mesh <{cfg.mesh._target_}>")
    if cfg.mesh._target_ == "pinnstorch.data.Mesh":
        mesh: Mesh = hydra.utils.instantiate(
            cfg.mesh, time_domain=td, spatial_domain=sd, read_data_fn=read_data_fn
        )
    elif cfg.mesh._target_ == "pinnstorch.data.PointCloud":
        mesh: PointCloud = hydra.utils.instantiate(cfg.mesh, read_data_fn=read_data_fn)
    else:
        raise "Mesh should be defined in config file."

    train_datasets = []
    for i, (dataset_dic) in enumerate(cfg.train_datasets):
        for key, dataset in dataset_dic.items():
            log.info(f"Instantiating training dataset number {i+1}: <{dataset._target_}>")
            train_datasets.append(hydra.utils.instantiate(dataset)(mesh=mesh))

    val_dataset = None
    if cfg.get("val_dataset"):
        for i, dataset_dic in enumerate(cfg.val_dataset):
            for key, dataset in dataset_dic.items():
                log.info(f"Instantiating validation dataset number {i+1}: <{dataset._target_}>")
                val_dataset = hydra.utils.instantiate(dataset)(mesh=mesh)

    test_dataset = None
    if cfg.get("test_dataset"):
        for i, dataset_dic in enumerate(cfg.test_dataset):
            for key, dataset in dataset_dic.items():
                log.info(f"Instantiating test dataset number {i+1}: <{dataset._target_}>")
                test_dataset = hydra.utils.instantiate(dataset)(mesh=mesh)

    pred_dataset = None
    if cfg.get("pred_dataset"):
        for i, dataset_dic in enumerate(cfg.pred_dataset):
            for key, dataset in dataset_dic.items():
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

    if cfg.net._target_ == "pinnstorch.models.FCN":
        log.info(f"Instantiating neural net <{cfg.net._target_}>")
        net: torch.nn.Module = hydra.utils.instantiate(cfg.net)(lb=mesh.lb, ub=mesh.ub)
    elif cfg.net._target_ == "pinnstorch.models.NetHFM":
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

    if cfg.get("train"):
        log.info("Starting training!")
        start_time = time.time()
        trainer.fit(model=model, datamodule=datamodule)
        log.info(f"Elapsed time: {time.time() - start_time}")

    log.info(f"Median time for each batch: {np.median(model.times)}")
    
    train_metrics = trainer.callback_metrics
    
    if cfg.get("val"):
        log.info("Starting validation!")
        model.amp = False
        trainer.validate(model=model, datamodule=datamodule)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = None  # trainer.checkpoint_callback.best_model_path
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    if cfg.get("pred_dataset"):
        preds_list = trainer.predict(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
        preds_dict = utils.fix_predictions(preds_list)
        if cfg.get("save_pred"):
            pred_path = f'{cfg.paths.output_dir}/predictions.pkl'
            with open(pred_path, 'wb') as f:
                pickle.dump(preds_dict, f)
            log.info(f"Predictions saved at: {pred_path}")
        
    if cfg.get("plotting"):
        log.info("Plotting the results")
        hydra.utils.instantiate(
            cfg.plotting,
            mesh=mesh,
            preds=preds_dict,
            train_datasets=train_datasets,
            val_dataset=val_dataset,
            file_name=cfg.paths.output_dir,
        )()

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
