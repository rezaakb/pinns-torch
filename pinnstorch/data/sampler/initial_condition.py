import numpy as np
import torch

from .sampler_base import SamplerBase
from pinnstorch.utils import set_requires_grad


class InitialCondition(SamplerBase):
    """Initialize initial boundary condition."""

    def __init__(self, mesh, num_sample=None, solution=None, initial_fun=None):
        """Initialize an InitialCondition object for sampling initial condition data.

        :param mesh: Mesh object containing spatial and time domain information.
        :param num_sample: Number of samples.
        :param solution: List of solution variable names.
        :param initial_fun: Function to generate initial conditions (optional).
        """
        super().__init__()

        self.solution_names = solution

        (self.spatial_domain, self.time_domain, self.solution) = mesh.on_initial_boundary(
            self.solution_names
        )

        if initial_fun:
            self.solution = initial_fun(self.spatial_domain)

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(num_sample, (self.spatial_domain, self.time_domain, self.solution))

        self.spatial_domain_sampled = list(torch.split(self.spatial_domain_sampled, (1), dim=1))
        self.solution_sampled = list(torch.split(self.solution_sampled, (1), dim=1))

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training. If num_sample is not defined the whole points will be
        selected.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name] for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor(
                (flatten_mesh[0][idx, :], flatten_mesh[1][idx, :], flatten_mesh[2][idx, :])
            )

    def _loss_fn(self, inputs, loss):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        outputs = self.functions["forward"](x, t)
        loss = self.functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs
