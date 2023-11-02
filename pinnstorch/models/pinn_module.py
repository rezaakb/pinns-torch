from typing import Any, Callable, Dict, List, Tuple, Union

import os
import time
import torch
import torch._lazy
import torch._lazy.ts_backend
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

log = get_pylogger(__name__)

class PINNModule(LightningModule):
    """
    A LightningModule for training Physics-Informed Neural Networks (PINNs) to solve partial 
    differential equations (PDEs).
    """

    function_mapping: Dict[str, Callable]
    net: torch.nn.Module
    
    def __init__(
        self,
        net: torch.nn.Module,
        pde_fn,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
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
        inline: bool = False
    ) -> None:
        """
        Initialize the PINNModule.

        Sets up the model, loss function, optional features like AMP, JIT compilation, and CUDA Graphs,
        and other utilities for training and evaluating a Physics-Informed Neural Network.

        :param net: The neural network model.
        :param pde_fn: The function representing the PDE to solve.
        :param optimizer: The optimizer for training.
        :param scheduler: Optional learning rate scheduler.
        :param scaler: Optional gradient scaler for AMP.
        :param loss_fn: The loss function to use, either "sse" or "mse".
        :param extra_variables: Optional extra variables in inverse problems.
        :param output_fn: Optional function to process the model's output.
        :param runge_kutta: Optional Runge-Kutta method for solving PDEs.
        :param cudagraph_compile: Flag to enable CUDA Graph compilation. It works only with a single GPU.
        :param jit_compile: Flag to enable JIT compilation.
        :param amp: Flag to enable Automatic Mixed Precision (AMP).
        :param lazy: Flag to enable the use of LazyTensors.
        :param inline: Flag to enable inline mode in JIT compilation.
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
        self.times_batch = []
        self.xla = False

        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse

        if self.amp:
            torch._C._jit_set_autocast_mode(True)

        if self.cudagraph_compile:
            self.automatic_optimization = False
            
        if self.jit_compile and self.cudagraph_compile:
            if inline:
                torch._C._debug_set_autodiff_subgraph_inlining(False)
                torch._C._jit_set_nvfuser_single_node_mode(True)
            else:
                torch._C._jit_set_nvfuser_enabled(True)
        
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
            "net": self.net,
            "jit_compile": self.jit_compile,
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
        """
        Initialize the function mapping at the start of the fitting process.

        :return: None
        """
        
        self.function_mapping = self.trainer.datamodule.function_mapping
        
    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.

        By default lightning executes validation step sanity checks before training starts,
        so it's worth to make sure validation metrics don't store results from these checks
        
        :return: None
        """
        
        self.val_loss.reset()
        self.val_error.reset()
        self.val_loss_best.reset()

        self.functions["batch_size"] = True if self.trainer.datamodule.batch_size else False
        
        if self.lazy: 
            device = 'lazy'
        elif self.xla:
            device = 'xla'
        else:
            device = self.device
        self.net = self.net.to(device)
        self.train_loss = self.train_loss.to(device)
        if self.extra_variables:
            self.extra_variables = self.extra_variables.to(device)
        if self.rk:
            self.rk = self.rk.to(device)

    def on_train_batch_start(self):
        """
        Prepare the training loss tensor for the upcoming training batch.
        This method is called at the beginning of each training batch
        to ensure the `train_loss` tensor is on the appropriate device.
        
        :return: None
        """
        
        if self.lazy: 
            device = 'lazy'
        elif self.xla:
            device = 'xla'
        else:
            device = self.device
            
        self.train_loss = self.train_loss.to(device)
            
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Perform moving the data to the target device.
        If we do not use CUDA graphs we need to move the batch to self.device for
        every batch, but for capturing CUDA graphs, we have to do it one time.

        :param batch: A batch of data (a tuple) containing the
        :param device: The target device
        :param dataloader_idx: The index of dataloader 

        :return: The batch in target device.
        """
        
        if self.lazy:
            device = 'lazy'

        if self.xla:
            device = 'xla'

        if self.automatic_optimization:
            # If we do not use CUDA graphs we need to move the batch to self.device.
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            # If we use CUDA graphs we need to move the batch to self.device just for the first time.
            if self.capture_end is False:
                batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            else:
                if self.trainer.datamodule.batch_size and self.val_stage is False:
                    self.copy_batch(batch)
                elif self.trainer.datamodule.batch_size and self.val_stage is True:
                    batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            
        return batch

   
    def on_before_backward(self, loss):
        """    
        This method is invoked before the backward pass in the training process.
        If both AMP and CUDA Graph compilation are enabled,
        it scales the loss value using the model's scaler to facilitate a more numerically stable 
        and efficient backward pass. Otherwise, the loss remains unmodified.
    
        :param loss: The original loss value calculated from the forward pass.
        :return: The potentially scaled loss value, depending on the AMP and CUDA Graph compilation settings.
        """
        
        if self.amp and self.cudagraph_compile:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def capture_graph(self, batch):
        """
        Capture the computation graph after warming up with 11 iterations.
    
        This method initiates a warm-up phase of 11 training iterations and then captures the CUDA computation graph for 
        the subsequent iterations. This is particularly useful for optimizing GPU computations by reusing the 
        precompiled computation graph.
    
        The warm-up and graph capturing are performed on a separate CUDA stream to avoid interfering with the ongoing 
        computations on the default stream. The captured graph encompasses the forward and backward passes, as well 
        as the optimizer step, making the entire training iteration highly efficient.
    
        :param batch: The input batch of data to be used for warm-up and graph capturing.
        :return: None
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

        torch.cuda.current_stream().wait_stream(self.s)
        
        self.g = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.g):
            with torch.autocast(device_type='cuda', enabled = self.amp):
                self.static_loss, self.static_pred = self.model_step(self.static_batch)
            self.manual_backward(self.static_loss)
            if self.amp is False:
                self.opt.step()

        self.batch_idx = 1
        self.capture_end = True
        print('Capture Time', time.time() - self.capture_time)
    
    def copy_batch(self, batch) -> None:
        """
        Fills the graph's input memory with new data to compute on. If the batch_size is not
        specified, the model uses all the available data, and there is no need to copy the data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        """
        
        import time as tm
        st = tm.time()
        
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
        
        self.times_batch.append(tm.time() - st)
        
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
        
        if self.cudagraph_compile:
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

        else:
                
            loss, pred = self.model_step(batch)            
            
            # update and log metrics    
            self.train_loss(loss)
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
            
            return loss


    def on_train_batch_start(self, batch, batch_idx):
        """
        Set validation flag to False and record start time at a specific training step.
    
        :param batch: The current batch of data being processed.
        :param batch_idx: The index of the current batch.
        :return: None
        """
        self.functions['val'] = False

    def on_train_batch_end(self, batch_output, batch, batch_idx):
        """
        Log the time taken for a specific training step and append the time to the times list.

        :param batch_output: The output of the training batch.
        :param batch: The current batch of data being processed.
        :param batch_idx: The index of the current batch.
        :return: None
        """
        self.times.append(time.time() - self.start_time)
        if self.lazy:
            torch._lazy.mark_step()
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch.

        If extra_variables are provided, logs each extra variable's value to the progress bar.
        """
        if self.extra_variables:
            for key in self.extra_variables:
                self.log(key, self.extra_variables[key], prog_bar=True, sync_dist=False)
        
        
    def on_validation_start(self):
        """
        Set the validation stage flag to True at the start of the validation phase.

        :return: None
        """
        
        self.val_stage = True

    def on_validation_end(self):
        """
        Reset the validation stage flag to False at the end of the validation phase.

        :return: None
        """
        
        self.val_stage = False

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
        
        
        self.functions['val'] = True
        
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
        """
        Perform a single validation step on a batch of data from the validation set.

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
        """
        Lightning hook that is called when a validation epoch ends.

        :return: None
        """
        
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=False, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Perform a single test step on a batch of data from the test set.

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
        """
        Perform a single predict step on a batch of data from the prediction set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """

        loss, error, preds = self.eval_step(batch)

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimizers and learning-rate schedulers to be used for training.

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
