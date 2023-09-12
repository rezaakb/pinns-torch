from typing import Any, Callable, Dict, List, Tuple, Union

import os
import time
import torch
import torch._lazy
import torch._lazy.ts_backend
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch.cuda.amp as amp
import numpy as np

from lightning import LightningModule
from lightning.pytorch.utilities import move_data_to_device
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin

from pinnstorch.utils.gradient_fn import fwd_gradient, gradient
from pinnstorch.utils.module_fn import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    set_requires_grad,
    sse,
)

from pinnstorch.utils import get_pylogger

def traceable(f):
    f = torch._dynamo.allow_in_graph(f)
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

log = get_pylogger(__name__)

torch.backends.cudnn.benchmark = True

class PINNModule(LightningModule):
    """Example of a `LightningModule` for PDE equations.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def capture_graph(self, batch):
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

    function_mapping: Dict[str, Callable]
    net: torch.nn.Module
    
    def __init__(
        self,
        net: torch.nn.Module,
        pde_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        scaler: torch.cuda.amp.GradScaler = None,
        loss_fn: str = "sse",
        extra_variables: Dict[str, Any] = None,
        output_fn = None,
        runge_kutta=None,
        cudagraph_compile: bool = False,
        jit_compile: bool = True,
        amp: bool = True,
        lazy: bool = False,
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
        :param jit_compile: Whether to use TorchScripts compiler.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "runge_kutta", "scaler"])

        self.net = net
        self.capture_end = False
        self.extra_variables: torch.nn.ParameterDict = fix_extra_variables(extra_variables)
        self.pde_fn = pde_fn
        self.output_fn = output_fn
        self.rk = runge_kutta
        self.cudagraph_compile = cudagraph_compile
        self.jit_compile = jit_compile
        self.preds_dict = {}
        self.lazy = lazy
        self.amp = amp
        self.opt = None
        self.scaler = scaler
        self.times = []

        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse

        if self.amp:
            torch._C._jit_set_autocast_mode(True)

        if self.cudagraph_compile:
            self.automatic_optimization = False
            
        if self.jit_compile:
            s=1
            self.pde_fn = torch.compile(self.pde_fn, backend='aot_eager')
            #self.output_fn = torch.compile(traceable(self.output_fn), backend='aot_eager')
            #torch._C._jit_set_profiling_mode(False)
            #torch._C._jit_set_profiling_executor(False)
            #torch._C._jit_set_nvfuser_enabled(True)
            #self.pde_fn = torch.jit.script(self.pde_fn)
            #self.pde_fn = torch.jit.script(self.output_fn)
            #self.net = torch.jit.script(self.net)
            #self.loss_fn = torch.jit.script(self.loss_fn)
            #self.pde_fn.__name__ = 'pde_fn'
            #self.pde_fn = torch.compile(self.pde_fn)
     

        if self.lazy:
            torch._lazy.ts_backend.init()
        
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
        if self.output_fn is not None:
            outputs = self.output_fn(outputs, *spatial, time)
        return outputs

    def on_fit_start(self) -> None:
        self.function_mapping = self.trainer.datamodule.function_mapping
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        device = xm.xla_device()
        if self.lazy: 
            device = 'lazy'
        self.net = self.net.to(device)
        self.train_loss = self.train_loss.to(device)
        if self.extra_variables:
            self.extra_variables = self.extra_variables.to(device)
        if self.rk:
            self.rk = self.rk.to(device)

    def on_train_batch_start(self):
        if self.lazy: 
            self.train_loss = self.train_loss.to('lazy')
            
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Perform moving the data to the target device.
        If we do not use CUDA graphs we need to move the batch to self.device for
        every batch, but for capturing CUDA graphs, we have to do it one time.

        :param batch: A batch of data (a tuple) containing the
        :param device: The target device
        :param dataloader_idx: The index of dataloader 

        :return: The batch in target device.
        """
        if self.lazy:
            device = 'lazy'
        device = xm.xla_device()
        if self.automatic_optimization:
            # If we do not use CUDA graphs we need to move the batch to self.device.
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            # If we use CUDA graphs we need to move the batch to self.device just for the first time.
            if self.capture_end is False:
                batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            else:
                if self.trainer.datamodule.batch_size:
                    self.copy_batch(batch)
            
        return batch

   
    def on_before_backward(self, loss):
        """In the case that we use AMP, it scales loss value before backward.
        """ 
        
        if self.amp and self.cudagraph_compile:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def capture_graph(self, batch):
        """Performs 11 iterations for warm-up. Then, it will capture the graph.
        """
        self.capture_time = time.time()
        self.opt = self.optimizers()
        
        self.static_batch = batch
        self.batch_idx = 0
        
        self.s = torch.cuda.Stream()
        self.s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.s):
            for i in range(11):
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled = self.amp):
                    loss, pred = self.model_step(self.static_batch)
                    
                self.manual_backward(loss)
                if self.amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                    if self.lazy:
                        torch._lazy.mark_step()

        torch.cuda.current_stream().wait_stream(self.s)
        
        self.g = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.g):
            with torch.autocast(device_type='cuda', enabled = self.amp):
                self.static_loss, self.static_pred = self.model_step(self.static_batch)
            self.manual_backward(self.static_loss)
            if self.amp is False:
                self.opt.step()
                if self.lazy:
                    torch._lazy.mark_step()

        self.batch_idx = 1
        self.capture_end = True
        print('Capture Time', time.time() - self.capture_time)
    
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
    
    def model_step(
        self,
        batch: Dict[
            str,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
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
            x, t, u = data
            x, t = set_requires_grad(x, t, True)
            loss, preds = self.function_mapping[loss_fn](data, loss, self.functions)
        
        return loss, preds


    def training_step(
        self,
        batch: Dict[
            str,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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
        self.start_time = time.time()
        
        if not self.cudagraph_compile:
            loss, pred = self.model_step(batch)
            
            # update and log metrics    
            self.train_loss(loss)
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            
            return loss

        else:
            if batch_idx == 0 and not self.capture_end:
                self.capture_graph(batch)
            else:
                
                self.g.replay()
                
                if self.amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()

            # update and log metrics
            self.train_loss(self.static_loss)
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)




    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.start_time_500 = time.time()

    def on_train_batch_start(self, batch, batch_idx):
        
        if self.global_step == 80:
            self.start_time_100 = time.time()

    def on_train_batch_end(self, batch_output, batch, batch_idx):
        if self.global_step == 81:
            log.info(f"step {self.global_step}: {time.time() - self.start_time_100}")
        self.times.append(time.time() - self.start_time)
        if self.lazy:
            torch._lazy.mark_step()
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch.

        If extra_variables are provided, logs each extra variable's value to the progress bar.
        """
        if self.current_epoch == 0:
            print('=======')
            if self.capture_end:
                log.info('CUDAGraph')
            if self.jit_compile: 
                log.info('JIT')
            if self.amp:
                log.info('AMP')
            if self.automatic_optimization:
                log.info('Auto')
            log.info(f"epoch {self.current_epoch}: {time.time() - self.start_time_500}")
        if len(self.times) == 11:
            log.info(f"Compile Time - SUM = {np.sum(self.times)}, - MEAN = {np.mean(self.times)}")
        if self.extra_variables:
            for key in self.extra_variables:
                self.log(key, self.extra_variables[key], prog_bar=True, sync_dist=False)
        
        
            

    def eval_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single evaluation step on a batch of data.

        :param batch: A batch of data containing input tensors and conditions.
        :return: A tuple containing loss, error dictionary, and predictions.
        """

        # `loss_fn` and `solution` are passed from PINNDataModule class.
        loss_fn = self.trainer.datamodule.loss_fn
        solution = self.trainer.datamodule.solution_names
        
        # In evalutation, because we do not use captured graph,
        # we need to transfer batch manually.
        batch = super().transfer_batch_to_device(batch, self.device, 0)
        _, _, u = batch

        with torch.set_grad_enabled(True):
            loss, preds = self.model_step({str(loss_fn): batch})

        if self.rk:
            error_dict = {
                solution_name: relative_l2_error(
                    preds[solution_name][:, -1][:, None], u[solution_name]
                )
                for solution_name in solution
            }
            for solution_name in solution:
                print(preds[solution_name][:, -1][:, None].shape, u[solution_name].shape)

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
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """
        
        loss, error_dict, preds = self.eval_step(batch)
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        for solution_name, error in error_dict.items():
            self.val_error(error)
            self.log(f"val/error_{solution_name}", error, prog_bar=True, sync_dist=False)

    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=False, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """

        loss, error_dict, preds = self.eval_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        for solution_name, error in error_dict.items():
            self.test_error(error)
            self.log(f"test/error_{solution_name}", error, prog_bar=True, sync_dist=False)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Perform a single predict step on a batch of data from the prediction set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """

        loss, error, preds = self.eval_step(batch)

        return preds

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
