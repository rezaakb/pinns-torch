from typing import Callable, List, Union

import numpy as np
import torch
from pyDOE import lhs

from pinnstorch.data import TimeSeriesDomain


class VectorBase:
    """This helper class is utilized by the Timeseries class."""

    def __init__():
        """Base class for generating vector data. No support for boundaries and inital points  """

        pass

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
        time_domain = self.timeseries_domain_mesh.flatten()[:, None]
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


class Vector(VectorBase):
    """For using this class you should define a SpatialDomain and TimeDomain classes.

    If dimensions of mesh is not determined, it is better to use PointCloud.
    """

    def __init__(
        self,
        time_series_domain: TimeSeriesDomain,
        root_dir: str,
        read_data_fn: Callable,
        ub: List = None,
        lb: List = None,
    ):
        """Generate a mesh based on spatial and time domains, and load solution data.

        :param time_series_domain: Instance of a TimeDomain class.
        :param root_dir: Root directory for solution data.
        :param read_data_fn: Function to read solution data.
        :param ub: Upper bounds for domain.
        :param lb: Lower bounds for domain.
        """

        self.solution = read_data_fn(root_dir)
        t_points = list(self.solution.values()).shape   # single 

        self.time_series_domain = time_series_domain

        # Generate Mesh for both spatial and time domain
        self.time_series_domain_mesh = time_series_domain.generate_mesh()

        # We utilize the user-defined lower and upper bounds if they are specified.
        if ub is None and lb is None:
            self.lb, self.ub = self.domain_bounds()
        else:
            self.lb, self.ub = np.array(lb), np.array(ub)