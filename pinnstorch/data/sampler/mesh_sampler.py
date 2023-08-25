from typing import List, Dict

import torch
import numpy as np

from .sampler_base import SamplerBase
from pinnstorch import utils

class MeshSampler(SamplerBase):
    def __init__(self,
                 mesh,
                 idx_t: int = None,
                 num_sample: int = None,
                 solution: List = None,
                 collection_points: List = None,
                 discrete: bool = False):
        """
        Initialize a mesh sampler for collecting training data.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        """
        
        super().__init__()
        
        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t
        
        # On a time step.
        if self.idx_t:
            flatten_mesh = mesh.on_initial_boundary(self.solution_names,
                                                    self.idx_t)
            
        # All time steps.
        elif self.solution_names is not None:
            flatten_mesh = mesh.flatten_mesh(self.solution_names)

        if self.solution_names:
            (self.spatial_domain_sampled,
             self.time_domain_sampled,
             self.solution_sampled) = self.sample_mesh(num_sample,
                                                       flatten_mesh)
            
            self.solution_sampled = list(torch.split(self.solution_sampled, (1), dim=1))
        
        # Collection Points only.
        else:
            (self.spatial_domain_sampled,
             self.time_domain_sampled) = self.convert_to_tensor(mesh.collection_points(num_sample))
            
            self.solution_sampled = None

        self.spatial_domain_sampled = list(torch.split(self.spatial_domain_sampled, (1), dim=1))
        
    
    def sample_mesh(self, num_sample, flatten_mesh):
        """
        Sample the mesh data for training.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [flatten_mesh[2][solution_name] for solution_name in self.solution_names]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)
        
        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor((flatten_mesh[0][idx,:],
                                           flatten_mesh[1][idx,:],
                                           flatten_mesh[2][idx,:]))
    
    def loss_fn(self, inputs, loss, **functions):
        """
        Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """
                                              
        x, t, u = inputs
        x, t = self.requires_grad(x, t, True)
        
        outputs = functions['forward'](x, t)
        
        if self.collection_points_names:
            outputs = functions['pde_fn'](outputs, *x, t, functions['extra_variables'])

        loss = functions['loss_fn'](loss, outputs, keys=self.collection_points_names) + \
               functions['loss_fn'](loss, outputs, u, keys=self.solution_names)
        
        return loss, outputs


class DiscreteMeshSampler(SamplerBase):
    def __init__(self,
                 mesh,
                 idx_t: int,
                 num_sample: int = None,
                 solution: List = None,
                 collection_points: List = None):
        """
        Initialize a mesh sampler for collecting training data in discrete mode.

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
        
        flatten_mesh = mesh.on_initial_boundary(self.solution_names,
                                                self.idx_t)

        (self.spatial_domain_sampled,
         self.time_domain_sampled,
         self.solution_sampled) = self.sample_mesh(num_sample,
                                                   flatten_mesh)

        self.spatial_domain_sampled = list(torch.split(self.spatial_domain_sampled, (1), dim=1))
        self.time_domain_sampled = None
        self.solution_sampled = list(torch.split(self.solution_sampled, (1), dim=1))
        
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        self._mode = value
        
    def sample_mesh(self, num_sample, flatten_mesh):
        """
        Sample the mesh data for training.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [flatten_mesh[2][solution_name] for solution_name in self.solution_names]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)
        
        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor((flatten_mesh[0][idx,:],
                                           flatten_mesh[1][idx,:],
                                           flatten_mesh[2][idx,:]))
    
    def loss_fn(self, inputs, loss, **functions):
        """
        Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs
        x, t = self.requires_grad(x, t, True)
        
        outputs = functions['forward'](x, t)
        
        if functions['output_fn']:
            outputs = functions['output_fn'](outputs, *x, t)

        if self._mode:
            outputs = functions['pde_fn'](outputs, *x, functions['extra_variables'])
            outputs = functions['runge_kutta'](outputs,
                                               mode = self._mode,
                                               solution_names = self.solution_names,
                                               collection_points_names = self.collection_points_names)
        
        loss = functions['loss_fn'](loss, outputs, u, keys=self.solution_names)
   
        return loss, outputs