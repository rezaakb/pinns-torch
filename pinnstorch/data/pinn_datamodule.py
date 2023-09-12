from typing import Any, Dict, Optional, Tuple, Callable

import torch
from lightning import LightningDataModule

from .dataloader.dataloader import PINNDataLoader


class PINNDataModule(LightningDataModule):
    """`LightningDataModule` for the PINN dataset.

    This Data Module will hold datasets that should be used for training and Validation.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    function_mapping: Dict[str, Callable]
    
    def __init__(
        self,
        train_datasets,
        val_dataset,
        test_dataset=None,
        pred_dataset=None,
        batch_size: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `PINNDataModule`.

        :param train_datasets: train datasets.
        :param val_dataset: validation dataset.
        """
        super().__init__()

        self.train_datasets = train_datasets
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

        if isinstance(val_dataset, list):
            raise "Validation dataset cannot be a list."

        self.batch_size = batch_size

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        #self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = self.val_dataset
        self.data_test = (
            self.test_dataset if self.test_dataset else self.val_dataset
        )
        self.data_pred = (
            self.pred_dataset if self.pred_dataset else self.val_dataset
        )

        self.function_mapping = torch.jit.annotate(Dict[str, Callable], {})

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = {}
            self.set_mode_for_discrete_mesh()

            for train_dataset in self.train_datasets:
                self.data_train[str(train_dataset.loss_fn)] = PINNDataLoader(
                    train_dataset, batch_size=self.batch_size, ignore=True, shuffle=True
                )
                self.function_mapping[str(train_dataset.loss_fn)] = train_dataset.loss_fn

    def set_mode_for_discrete_mesh(self):
        """This function will figuere out which training datasets are for discrete.

        Then set the mode value that will be used for Runge–Kutta methods
        """

        mesh_idx = [
            (train_dataset.idx_t, train_dataset)
            for train_dataset in self.train_datasets
            if type(train_dataset).__name__ == "DiscreteMeshSampler"
        ]

        mesh_idx = sorted(mesh_idx, key=lambda x: x[0])

        if len(mesh_idx) == 1:
            mesh_idx[0][1].mode = "forward_discrete"
        elif len(mesh_idx) == 2:
            mesh_idx[0][1].mode = "inverse_discrete_1"
            mesh_idx[1][1].mode = "inverse_discrete_2"

        mesh_idx.clear()

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.data_train

    def val_dataloader(self):
        """Create and return the validation dataloader. `self.loss_fn` and `self.solution` will be
        used in `PINNModule`. Lightning does not allow multiple val datasets.

        :return: The validation dataloader.
        """

        self.loss_fn = self.data_val.loss_fn
        self.function_mapping[str(self.loss_fn)] = self.loss_fn
        self.solution_names = self.data_val.solution_names

        return PINNDataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Create and return the test dataloader. `self.loss_fn` and `self.solution` will be used
        in `PINNModule`. Lightning does not allow multiple val datasets.

        :return: The test dataloader.
        """
        self.loss_fn = self.test_dataset.loss_fn if self.test_dataset else self.val_dataset.loss_fn
        self.function_mapping[str(self.loss_fn)] = self.loss_fn
        self.solution_names = (
            self.test_dataset.solution_names
            if self.test_dataset
            else self.val_dataset.solution_names
        )

        return PINNDataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        """Create and return the test dataloader. `self.loss_fn` and `self.solution` will be used
        in `PINNModule`. Lightning does not allow multiple val datasets.

        :return: The test dataloader.
        """
        self.loss_fn = self.pred_dataset.loss_fn if self.pred_dataset else self.val_dataset.loss_fn
        self.function_mapping[str(self.loss_fn)] = self.loss_fn
        self.solution_names = (
            self.pred_dataset.solution_names
            if self.pred_dataset
            else self.val_dataset.solution_names
        )

        return PINNDataLoader(self.data_pred, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
        


if __name__ == "__main__":
    _ = PINNDataModule(None, None)
