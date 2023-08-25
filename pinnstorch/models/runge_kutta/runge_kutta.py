import logging
import os

import numpy as np
import requests
import torch

from pinnstorch.utils import load_data_txt

log = logging.getLogger(__name__)


class RungeKutta(torch.nn.Module):
    def __init__(self, root_dir, t1: int, t2: int, time_domain, q: int = None):
        """Initialize a RungeKutta object for solving differential equations using Implicit Runge-
        Kutta methods.

        :param root_dir: Root directory where the weights data is stored.
        :param t1: Start index of the time domain.
        :param t2: End index of the time domain.
        :param time_domain: TimeDomain class representing the time domain.
        :param q: Order of the Implicit Runge-Kutta method. If not provided, it is automatically
            calculated.
        """
        super().__init__()

        dt = np.array(time_domain[t2] - time_domain[t1]).astype(np.float32)

        # self.irk = {}
        # self.irk['dt'] = torch.tensor(dt)
        self.register_buffer("dt", torch.from_numpy(dt))

        if q is None:
            q = int(np.ceil(0.5 * np.log(np.finfo(float).eps) / np.log(dt)))

        self.load_irk_weights(root_dir, q)

    def load_irk_weights(self, root_dir, q: int) -> None:
        """Load the weights and coefficients for the Implicit Runge-Kutta method and save in the
        dictionary.

        :param root_dir: Root directory where the weights data is stored.
        :param q: Order of the Implicit Runge-Kutta method.
        """
        file_name = "Butcher_IRK%d.txt" % q
        tmp = load_data_txt(root_dir, file_name)

        weights = np.reshape(tmp[0 : q**2 + q], (q + 1, q)).astype(np.float32)
        # self.irk['alpha'] = torch.tensor(weights[0:-1, :].T)
        self.register_buffer("alpha", torch.from_numpy(weights[0:-1, :].T))
        # self.irk['beta'] = torch.tensor(weights[-1:, :].T)
        self.register_buffer("beta", torch.from_numpy(weights[-1:, :].T))
        # self.irk['weights'] = torch.tensor(weights.T)
        self.register_buffer("weights", torch.from_numpy(weights.T))
        self.IRK_times = tmp[q**2 + q :]

    def to(self, device) -> None:
        """Move the internal tensors of the RungeKutta object to the specified device.

        :param device: The target device.
        """
        self.irk = {key: tensor.to(device) for key, tensor in self.irk.items()}

    def forward(self, outputs, mode, solution_names, collection_points_names):
        """Perform a forward step using the Runge-Kutta method for solving differential equations.

        :param outputs: Dictionary containing solution tensors and other variables.
        :param mode: The mode of the forward step, e.g., "inverse_discrete_1",
            "inverse_discrete_2", "forward_discrete".
        :param solution_names: List of keys for solution variables.
        :param collection_points_names: List of keys for collection point variables.
        :return: Dictionary with updated solution tensors after the forward step.
        """

        for solution_name, collection_points_name in zip(solution_names, collection_points_names):
            if mode == "inverse_discrete_1":
                outputs[solution_name] = outputs[solution_name] - self.dt * torch.matmul(
                    outputs[collection_points_name], self.alpha
                )

            elif mode == "inverse_discrete_2":
                outputs[solution_name] = outputs[solution_name] + self.dt * torch.matmul(
                    outputs[collection_points_name], (self.beta - self.alpha)
                )

            elif mode == "forward_discrete":
                outputs[solution_name] = outputs[solution_name] - self.dt * torch.matmul(
                    outputs[collection_points_name], self.weights
                )

        return outputs


if __name__ == "__main__":
    _ = RungeKutta(None, None, None, None)