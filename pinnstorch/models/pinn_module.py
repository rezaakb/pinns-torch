from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import move_data_to_device
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

from pinnstorch.utils.gradient_fn import fwd_gradient, gradient
from pinnstorch.utils.module_fn import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    requires_grad,
    sse,
)


class PINNModule(LightningModule):
    """Example of a `LightningModule` for PDE equations.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        pde_fn: Callable[[Any, ...], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        loss_fn: str = "sse",
        extra_variables: Dict[str, Any] = None,
        output_fn: Callable[[Any, ...], torch.Tensor] = None,
        runge_kutta=None,
        automatic_optimization: bool = False,
    ) -> None:
        """Initialize a `PINNModule`.

        :param net: The model to train.
        :param pde_fn: PDE function.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param extra_variables: Extra variables should be in a dictionary.
        :param loss_fn: PDE function will apply on the collection points.
        :param extra_variables: Dictionary
        :param output_fn: Output function will apply on the output of the net.
        :param runge_kutta: Runge–Kutta method will be used in discrete problems.
        :param automatic_optimization: Whether to use automatic optimization.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "runge_kutta"])

        self.net = net
        self.capture_end = False
        self.extra_variables = fix_extra_variables(extra_variables)
        self.output_fn = output_fn
        self.pde_fn = pde_fn
        self.automatic_optimization = False
        self.preds_dict = {}

        self.rk = runge_kutta

        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_error = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_error = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        # for using in each loss function of each condition.
        self.functions = {
            "runge_kutta": self.rk,
            "forward": self.forward,
            "pde_fn": self.pde_fn,
            "output_fn": self.output_fn,
            "extra_variables": self.extra_variables,
            "loss_fn": self.loss_fn,
        }

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """
        outputs = self.net(spatial, time)
        if self.output_fn:
            outputs = self.output_fn(outputs, *spatial, time)
        return outputs

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
            ],
        ],
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the
        input tensor of different conditions and data.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dictionary of predictions.
        """
        loss = 0.0
        for loss_fn, data in batch.items():
            loss, preds = loss_fn(data, loss, **self.functions)

        return loss, preds

    def capture_graph(self, batch):
        """Fills the graph's input memory with new data to compute on. If the batch_size is not
        specified, the model uses all the available data, and there is no need to copy the data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """

        self.opt = self.optimizers()

        self.static_batch = batch
        self.batch_idx = 0

        self.s = torch.cuda.Stream()
        self.s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.s):
            for i in range(3):
                self.opt.zero_grad(set_to_none=True)
                loss, pred = self.model_step(self.static_batch, self.batch_idx)
                self.manual_backward(loss)
                self.opt.step()

        torch.cuda.current_stream().wait_stream(self.s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.opt.zero_grad(set_to_none=True)
            self.static_loss, self.static_pred = self.model_step(self.static_batch, self.batch_idx)
            self.manual_backward(self.static_loss)
            self.opt.step()

        self.batch_idx = 1
        self.capture_end = True

    def copy_batch(self, batch) -> None:
        """Fills the graph's input memory with new data to compute on. If the batch_size is not
        specified, the model uses all the available data, and there is no need to copy the data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """

        for key in batch:
            spatial, time, solution = self.static_batch[key]
            spatial_new, time_new, solution_new = batch[key]
            time = time.requires_grad_(False).copy_(time_new)
            x = [
                spatial_.requires_grad_(False).copy_(spatial_new[i])
                for i, spatial_ in enumerate(spatial)
            ]
            if solution_new is not None:
                solution = {
                    key_sol: solution[key_sol].copy_(solution_new[key_sol])
                    for key_sol in solution_new
                }

    def training_step(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
            ],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set. self.g.replay()
        includes forward, backward, and step.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        if self.automatic_optimization or self.trainer.accelerator == 'cpu':
            self.static_loss, pred = self.model_step(batch, batch_idx)

        else:
            if batch_idx == 0 and not self.capture_end:
                self.capture_graph(batch)
            else:
                # If we utilize all the data in a single batch,
                # there is no need to copy data into the static batch.
                if self.trainer.datamodule.batch_size:
                    self.copy_batch(batch)
                self.g.replay()

        # update and log metrics
        self.train_loss(self.static_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return self.static_loss

    def on_train_epoch_end(self):
        if self.extra_variables:
            for key in self.extra_variables:
                self.log(key, self.extra_variables[key], prog_bar=True)

    def eval_step(self,
                  batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single evaluation step on a batch of data.

        :param batch: A batch of data containing input tensors and conditions.
        :return: A tuple containing loss, error dictionary, and predictions.
        """

        # `loss_fn` and `solution` are passed from PINNDataModule class.
        loss_fn = self.trainer.datamodule.loss_fn
        solution = self.trainer.datamodule.solution_names

        with torch.set_grad_enabled(True):
            x, t, u = batch
            loss, preds = self.model_step({loss_fn: batch})

            if self.rk:
                error_dict = {
                    solution_name: relative_l2_error(
                        preds[solution_name][:, -1][:, None], u[solution_name]
                    )
                    for solution_name in solution
                }
                
            else:
                error_dict = {
                    solution_name: relative_l2_error(preds[solution_name], u[solution_name])
                    for solution_name in solution
                }

        return loss, error_dict, preds

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        set_grad_enabled=True,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, error_dict, preds = self.eval_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for solution_name, error in error_dict.items():
            self.val_error(error)
            self.log(f"val/error_{solution_name}", error, prog_bar=True, sync_dist=False)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        set_grad_enabled=True,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, error_dict, preds = self.eval_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for solution_name, error in error_dict.items():
            self.test_error(error)
            self.log(f"test/error_{solution_name}", error, prog_bar=True, sync_dist=False)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        set_grad_enabled=True,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss, error, preds = self.eval_step(batch)

        return preds

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            self.scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PINNModule(None, None, None)
