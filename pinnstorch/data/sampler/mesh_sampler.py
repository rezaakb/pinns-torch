from typing import Dict, List

import numpy as np
import torch

from pinnstorch import utils

from .sampler_base import SamplerBase


class MeshSampler(SamplerBase):
    """Sample from Mesh for continuous mode."""

    def __init__(
        self,
        mesh,
        idx_t: int = None,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        use_lhs: bool = True,
    ):
        """Initialize a mesh sampler for collecting training data.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        :param use_lhs: Whether use lhs or not for generating collection points.
        """

        super().__init__()

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t

        # On a time step.
        if self.idx_t:
            flatten_mesh = mesh.on_initial_boundary(self.solution_names, self.idx_t)

        # All time steps.
        elif self.solution_names is not None:
            flatten_mesh = mesh.flatten_mesh(self.solution_names)

        if self.solution_names:
            (
                self.spatial_domain_sampled,
                self.time_domain_sampled,
                self.solution_sampled,
            ) = self.sample_mesh(num_sample, flatten_mesh)

            self.solution_sampled = list(torch.split(self.solution_sampled, (1), dim=1))

        # Collection Points only.
        else:
            (self.spatial_domain_sampled, self.time_domain_sampled) = self.convert_to_tensor(
                mesh.collection_points(num_sample, use_lhs)
            )

            self.solution_sampled = None

        self.spatial_domain_sampled = list(torch.split(self.spatial_domain_sampled, (1), dim=1))

    def loss_fn(self, inputs, loss, **functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs
        x, t = self.requires_grad(x, t, True)

        outputs = functions["forward"](x, t)

        if self.collection_points_names:
            outputs = functions["pde_fn"](outputs, *x, t, functions["extra_variables"])

        loss = functions["loss_fn"](loss, outputs, keys=self.collection_points_names) + functions[
            "loss_fn"
        ](loss, outputs, u, keys=self.solution_names)

        return loss, outputs


class DiscreteMeshSampler(SamplerBase):
    """Sample from Mesh for discrete mode."""

    def __init__(
        self,
        mesh,
        idx_t: int,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
    ):
        """Initialize a mesh sampler for collecting training data in discrete mode.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step for discrete mode.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        """
        super().__init__()

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t
        self._mode = None

        flatten_mesh = mesh.on_initial_boundary(self.solution_names, self.idx_t)

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(num_sample, flatten_mesh)

        self.spatial_domain_sampled = list(torch.split(self.spatial_domain_sampled, (1), dim=1))
        self.time_domain_sampled = None
        self.solution_sampled = list(torch.split(self.solution_sampled, (1), dim=1))

    @property
    def mode(self):
        """Get the current mode for RungeKutta class.

        :return: The current mode value.
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """Set the mode value by PINNDataModule for RungeKutta class.

        :param value: The mode value to be set.
        """
        self._mode = value

    def loss_fn(self, inputs, loss, **functions):
        """Compute the loss function based on inputs and functions. _mode is assigned in
        PINNDataModule class. It can be `inverse_discrete_1`, `inverse_discrete_2`, or
        `forward_discrete`

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs
        x, t = self.requires_grad(x, t, True)

        outputs = functions["forward"](x, t)

        if functions["output_fn"]:
            outputs = functions["output_fn"](outputs, *x, t)

        if self._mode:
            outputs = functions["pde_fn"](outputs, *x, functions["extra_variables"])
            outputs = functions["runge_kutta"](
                outputs,
                mode=self._mode,
                solution_names=self.solution_names,
                collection_points_names=self.collection_points_names,
            )
        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs
