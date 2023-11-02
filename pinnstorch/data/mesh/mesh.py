from typing import Callable, List, Union

import numpy as np
import torch
from pyDOE import lhs

from pinnstorch.data import Interval, Rectangle, RectangularPrism, TimeDomain


class MeshBase:
    """This helper class is utilized by the Mesh and PointCloud classes."""

    def __init__():
        """Base class for generating mesh data and boundary conditions."""

        pass

    def domain_bounds(self):
        """Calculate the domain bounds based on the generated spatial and time domain mesh.

        :return: Lower and upper bounds of the domain.
        """
        mesh = np.hstack(
            (
                self.spatial_domain_mesh.reshape(-1, self.spatial_dim),
                self.time_domain_mesh.reshape(-1, 1),
            )
        )

        ub = mesh.max(0)
        lb = mesh.min(0)
        return lb, ub

    def on_lower_boundary(self, solution_names: List):
        """Generate data for points on the lower boundary.

        :param solution_names: Names of the solution outputs.
        :return: Spatial, time, and solution data on the lower boundary.
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim)) * self.lb[0:-1]
        )
        time_domain = self.time_domain_mesh[0, :]
        solution_domain = {
            solution_name: self.solution[solution_name][0, :][:, None]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def on_upper_boundary(self, solution_names: List):
        """Generate data for points on the upper boundary.

        :param solution_names: Names of the solution outputs.
        :return: Spatial, time, and solution data on the upper boundary.
        """
        spatial_domain = (
            np.ones((self.time_domain_mesh.shape[1], self.spatial_dim)) * self.ub[0:-1]
        )
        time_domain = self.time_domain_mesh[-1, :]
        solution_domain = {
            solution_name: self.solution[solution_name][-1, :][:, None]
            for solution_name in solution_names
        }
        return spatial_domain, time_domain, solution_domain

    def on_initial_boundary(self, solution_names: List, idx: int = 0):
        """Generate data for points on the initial boundary.

        :param solution_names: Names of the solution outputs.
        :param idx: Index of the time step.
        :return: Spatial, time, and solution data on the initial boundary.
        """

        spatial_domain = np.squeeze(self.spatial_domain_mesh[:, idx : idx + 1, :], axis=-2)
        time_domain = self.time_domain_mesh[:, idx]
        solution_domain = {
            solution_name: self.solution[solution_name][:, idx : idx + 1]
            for solution_name in solution_names
        }

        return spatial_domain, time_domain, solution_domain

    def collection_points(self, N_f: int, use_lhs: bool = True):
        """Generate a collection of points for data collection.

        :param N_f: Number of points to collect.
        :return: Collection of points in the spatial domain.
        """
        if use_lhs:
            f = self.lb + (self.ub - self.lb) * lhs(self.spatial_dim + 1, N_f)
            spatial_domain = f[:, 0 : self.spatial_dim]
            time_domain = f[:, self.spatial_dim : self.spatial_dim + 1]
        else:
            spatial_domain, time_domain, _ = self.flatten_mesh(None)
        return spatial_domain, time_domain

    def flatten_mesh(self, solution_names: List):
        """Flatten the mesh data for training.

        :param solution_names: Names of the solution outputs.
        :return: Flattened spatial, time, and solution data.
        """
        time_domain = self.time_domain_mesh.flatten()[:, None]
        spatial_domain = np.zeros((len(time_domain), self.spatial_domain_mesh.shape[-1]))
        for i in range(self.spatial_domain_mesh.shape[-1]):
            spatial_domain[:, i] = self.spatial_domain_mesh[:, :, i].flatten()

        solution_domain = {}
        if solution_names is not None:
            for solution_name in solution_names:
                solution_domain[solution_name] = self.solution[solution_name][:, :].flatten()[
                    :, None
                ]

        return spatial_domain, time_domain, solution_domain


class Mesh(MeshBase):
    """For using this class you should define a SpatialDomain and TimeDomain classes.

    If dimensions of mesh is not determined, it is better to use PointCloud.
    """

    def __init__(
        self,
        spatial_domain: Union[Interval, Rectangle, RectangularPrism],
        time_domain: TimeDomain,
        root_dir: str,
        read_data_fn: Callable,
        ub: List = None,
        lb: List = None,
    ):
        """Generate a mesh based on spatial and time domains, and load solution data.

        :param spatial_domain: Instance of a SpatialDomain class.
        :param time_domain: Instance of a TimeDomain class.
        :param root_dir: Root directory for solution data.
        :param read_data_fn: Function to read solution data.
        :param ub: Upper bounds for domain.
        :param lb: Lower bounds for domain.
        """

        self.solution = read_data_fn(root_dir)
        spatial_points, t_points = list(self.solution.values())[0].shape

        self.spatial_domain, self.time_domain = spatial_domain, time_domain

        # Generate Mesh for both spatial and time domain
        self.spatial_domain_mesh = spatial_domain.generate_mesh(t_points)
        self.time_domain_mesh = time_domain.generate_mesh(spatial_points)

        self.spatial_dim = self.spatial_domain_mesh.shape[-1]

        # We utilize the user-defined lower and upper bounds if they are specified.
        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)


class PointCloud(MeshBase):
    """For using this class you should define a mesh of spatial domain, time domain, and
    solutions."""

    def __init__(self, root_dir: str, read_data_fn: Callable, ub: List = None, lb: List = None):
        """Generate a point cloud mesh and load data from files.

        :param root_dir: Root directory for data.
        :param read_data_fn: Function to read spatial, time, and solution data.
        :param ub: Upper bounds for domain.
        :param lb: Lower bounds for domain.
        """

        data = read_data_fn(root_dir)
        self.spatial_domain, self.time_domain, self.solution = (
            data.spatial,
            data.time,
            data.solution,
        )

        if not isinstance(self.solution, dict):
            raise "Solution outputs of read_data_fn function is not dictionary."

        if isinstance(self.time_domain, list):
            if len(self.time_domain) == 1:
                self.time_domain = self.time_domain[0]

        if not isinstance(self.spatial_domain, list):
            self.spatial_domain = [self.spatial_domain]

        spatial_num_points, time_num_points = list(self.solution.values())[0].shape

        self.spatial_dim = len(self.spatial_domain)
        self.time_dim = 1
        self.solution_dim = len(self.solution.keys())

        # Generate Mesh for both spatial and time domain
        self.spatial_domain_mesh = np.zeros(
            (spatial_num_points, time_num_points, self.spatial_dim)
        )

        for i, interval in enumerate(self.spatial_domain):
            self.spatial_domain_mesh[:, :, i] = np.tile(interval, (1, time_num_points))

        self.time_domain_mesh = np.tile(self.time_domain, (1, spatial_num_points)).T[:, :, None]

        # We utilize the user-defined lower and upper bounds if they are specified.
        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)
