import numpy as np
import torch 

from torch.utils.data import Dataset

class SamplerBase(Dataset):
    def __init__(self):
        self.time_domain_sampled = None
        self.spatial_domain_sampled = None
        self.solution_sampled = None
        self.solution_names = None
    
    def sample_mesh(self,
                    num_sample,
                    flatten_mesh):
        """
        Sample the mesh data for training. This class is base class for different samplers and conditions.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: flatten data.
        """
        
        pass
    
    @staticmethod
    def convert_to_tensor(arrays):
        """
        Convert NumPy arrays to PyTorch tensors.

        :param arrays: List of NumPy arrays to convert.
        :return: List of converted PyTorch tensors.
        """
        
        return [torch.from_numpy(array.astype(np.float32)) for array in arrays]

        
    def loss_fn(self, inputs, loss, **functions):
        """
        Compute the loss function based on given inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        """
        
        pass

    def requires_grad(self, x, t, enable_grad=True):
        """
        Set the requires_grad attribute for tensors in the input list.
        
        :param x: List of tensors to modify requires_grad attribute.
        :param t: Tensor to modify requires_grad attribute.
        :param enable_grad: Boolean indicating whether to enable requires_grad or not.
        :return: Modified list of tensors and tensor.
        """
        if t is not None:
            t = t.requires_grad_(enable_grad)
        x = [x_.requires_grad_(enable_grad) for x_ in x]
        
        return x, t

    @property
    def mean(self):
        """
        Calculate the mean of the concatenated input data along each column.

        :return: A numpy array containing the mean values along each column.
        """
        
        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1)
        
        return inputs.mean(0, keepdims=True)

    @property
    def std(self):
        """
        Calculate the standard deviation of the concatenated input data along each column.

        :return: A numpy array containing the standard deviation values along each column.
        """
        
        x, t, _ = self[:]
        inputs = np.concatenate((*x, t), 1)
        
        return inputs.std(0, keepdims=True)
    
    def __len__(self):
        """
        Get the number of sampled data points.

        :return: The number of sampled data points.
        """
        
        return len(self.spatial_domain_sampled[0])
    
    def __getitem__(self, idx):
        """
        Get a specific sampled data point using indexing.

        :param idx: Index of the desired data point.
        :return: Tuple containing spatial, time, and solution data for the indexed point.
        """
        
        spatial_domain = [spatial_domain[idx] for spatial_domain in self.spatial_domain_sampled]

        time_domain = None
        if self.time_domain_sampled is not None:
            time_domain = self.time_domain_sampled[idx]

        solution_domain = None
        if self.solution_sampled is not None:
            solution_domain = {solution_name: self.solution_sampled[i][idx]
                               for i, solution_name in enumerate(self.solution_names)}

        return (spatial_domain, time_domain, solution_domain)
        
