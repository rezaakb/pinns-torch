from typing import Tuple, List, Dict

import numpy as np
import torch

from pinnstorch.utils import jit_compiler

class SamplerBase:
    """Other samplers will be used this helper class."""

    def __init__(self):
        """Base class for sampling mesh data for training.

        Initializes instance variables for storing sampled data and solutions.
        """
        self.time_domain_sampled = None
        self.spatial_domain_sampled = None
        self.solution_sampled = None
        self.solution_names = None
        self.first_batch = True
        self.zeros_tensor = torch.tensor(0.0, dtype=torch.float32)
        

    def concatenate_solutions(self, flatten_mesh):
        """Concatenate dictionary of sampled solution data.

        :param flatten_mesh: Flattened mesh data.
        :return: Flattened mesh data with concatenated solutions.
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name] for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        return flatten_mesh

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

        flatten_mesh = self.concatenate_solutions(flatten_mesh)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor(
                (flatten_mesh[0][idx, :], flatten_mesh[1][idx, :], flatten_mesh[2][idx, :])
            )

    @staticmethod
    def convert_to_tensor(arrays):
        """Convert NumPy arrays to PyTorch tensors.

        :param arrays: List of NumPy arrays to convert.
        :return: List of converted PyTorch tensors.
        """

        return [torch.from_numpy(array.astype(np.float32)) for array in arrays]

    def loss_fn(self, inputs, loss, functions) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """
        
        self.functions = functions
        if self.first_batch and self.functions["jit_compile"] and not self.functions["val"]:
            if self.functions["batch_size"]:
                self.functions["pde_fn"] = torch.jit.script(self.functions["pde_fn"])
            else:
                self._loss_fn = torch.compile(self._loss_fn, backend=jit_compiler)
            self.first_batch = False

        loss, outputs = self._loss_fn(inputs, loss)

        return loss, outputs

    @property
    def mean(self):
        """Calculate the mean of the concatenated input data along each column.

        :return: A numpy array containing the mean values along each column.
        """

        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1)

        return inputs.mean(0, keepdims=True)

    @property
    def std(self):
        """Calculate the standard deviation of the concatenated input data along each column.

        :return: A numpy array containing the standard deviation values along each column.
        """

        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1)

        return inputs.std(0, keepdims=True)

    def __len__(self):
        """Get the number of sampled data points.

        :return: The number of sampled data points.
        """

        return len(self.spatial_domain_sampled[0])

    def __getitem__(self, idx):
        """Get a specific sampled data point using indexing. In some cases, we may not have
        `time_domain` and `solution_domain`. For example, in periodic boundary condition, there is
        not `solution_domain`.

        :param idx: Index of the desired data point.
        :return: Tuple containing spatial, time, and solution data for the indexed point.
        """

        spatial_domain = [spatial_domain[idx] for spatial_domain in self.spatial_domain_sampled]

        time_domain = self.zeros_tensor
        if self.time_domain_sampled is not None:
            time_domain = self.time_domain_sampled[idx]

        solution_domain = {'///': self.zeros_tensor}
        if self.solution_sampled is not None:
            solution_domain = {
                solution_name: self.solution_sampled[i][idx]
                for i, solution_name in enumerate(self.solution_names)
            }

        return (spatial_domain, time_domain, solution_domain)
