from typing import Dict, List

import numpy as np
import torch
from torch import nn


class FCN(nn.Module):
    """A simple fully-connected neural net for solving equations.

    In this model, lower and upper bound will be used for normalization of input data
    """
    output_names: List[str]

    def __init__(self, layers, lb, ub, output_names, discrete: bool = False, 
                dropout_rate: float = 0.0, enable_mc_dropout: bool = False) -> None:
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.enable_mc_dropout = enable_mc_dropout
        
        self.model = self.initalize_net(layers)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32, requires_grad=False))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32, requires_grad=False))
        self.output_names = output_names
        self.discrete = discrete

    def initalize_net(self, layers: List):
        initializer = nn.init.xavier_uniform_
        net = nn.Sequential()

        input_layer = nn.Linear(layers[0], layers[1])
        initializer(input_layer.weight)
        net.add_module("input", input_layer)
        net.add_module("activation_1", nn.Tanh())
        
        if self.dropout_rate > 0:
            net.add_module("dropout_1", nn.Dropout(self.dropout_rate))

        for i in range(1, len(layers) - 2):
            hidden_layer = nn.Linear(layers[i], layers[i + 1])
            initializer(hidden_layer.weight)
            net.add_module(f"hidden_{i+1}", hidden_layer)
            net.add_module(f"activation_{i+1}", nn.Tanh())
            
            if self.dropout_rate > 0:
                net.add_module(f"dropout_{i+1}", nn.Dropout(self.dropout_rate))

        output_layer = nn.Linear(layers[-2], layers[-1])
        initializer(output_layer.weight)
        net.add_module("output", output_layer)
        
        return net

    def mc_predict(self, spatial: List[torch.Tensor], time: torch.Tensor, 
                num_samples: int = 100) -> Dict[str, torch.Tensor]:
        if not self.enable_mc_dropout:
            raise ValueError("MC-Dropout is not enabled for this model")
        
        was_training = self.training
        self.train()  # Force training mode to keep dropout active
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(spatial, time)
                predictions.append(pred)
        
        self.train(was_training)
        
        results = {}
        for key in predictions[0].keys():
            stacked_preds = torch.stack([pred[key] for pred in predictions])
            results[f"{key}_mean"] = torch.mean(stacked_preds, dim=0)
            results[f"{key}_std"] = torch.std(stacked_preds, dim=0)
            results[f"{key}_samples"] = stacked_preds
        
        return results

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """

        # Discrete Mode
        if self.discrete:
            if len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y), 1)
            elif len(spatial) == 3:
                x, y, z = spatial
                z = torch.cat((x, y, z), 1)
            else:
                z = spatial[0]
            z = 2.0 * (z - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0

        # Continuous Mode
        else:
            if len(spatial) == 1:
                x = spatial[0]
                z = torch.cat((x, time), 1)
            elif len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y, time), 1)
            else:
                x, y, z = spatial
                z = torch.cat((x, y, z, time), 1)
            z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

        z = self.model(z)

        # Discrete Mode
        if self.discrete:
            outputs_dict = {name: z for i, name in enumerate(self.output_names)}

        # Continuous Mode
        else:
            outputs_dict = {name: z[:, i : i + 1] for i, name in enumerate(self.output_names)}
        return outputs_dict


class NetHFM(nn.Module):
    """A simple fully-connected neural net for solving equations.

    In this model, mean and std will be used for normalization of input data. Also, weight
    normalization will be done.
    """
    output_names: List[str]
    
    def __init__(self, mean, std, layers: List, output_names: List):
        super().__init__()
        """Initialize a `NetHFM` module.

        :param mesh: The number of layers.
        :param layers: The list indicating number of neurons in each layer.
        :param output_names: Names of outputs of net.
        """
        self.num_layers = len(layers)
        self.output_names = output_names

        self.register_buffer("X_mean", torch.from_numpy(mean))
        self.register_buffer("X_std", torch.from_numpy(std))

        self.initalize_net(layers)
        self.sigmoid = nn.Sigmoid()

    def initalize_net(self, layers: List) -> None:
        """Initialize the neural network weights, biases, and gammas.

        :param layers: The list indicating number of neurons in each layer.
        """

        self.weights = []
        self.biases = []
        self.gammas = []

        for i in range(0, self.num_layers - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(torch.tensor(W, dtype=torch.float32))
            self.biases.append(torch.tensor(b, dtype=torch.float32))
            self.gammas.append(torch.tensor(g, dtype=torch.float32))

        self.weights = nn.ParameterList(self.weights)
        self.biases = nn.ParameterList(self.biases)
        self.gammas = nn.ParameterList(self.gammas)

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A dictionary with output names as keys and corresponding output tensors as values.
        """
        if len(spatial) == 1:
            x = spatial[0]
            H = torch.cat((x, time), 1)
        elif len(spatial) == 2:
            x, y = spatial
            H = torch.cat((x, y, time), 1)
        else:
            x, y, z = spatial
            H = torch.cat((x, y, z, time), 1)
        
        H = (H - self.X_mean) / self.X_std

        for i, (W, b, g) in enumerate(zip(self.weights, self.biases, self.gammas)):
            # weight normalization
            V = W / torch.norm(W, dim=0)
            # matrix multiplication
            H = torch.matmul(H, V)
            # add bias
            H = g * H + b
            # activation
            if i < self.num_layers - 2:
                H = H * self.sigmoid(H)
        
        outputs_dict = {name: H[:, i : i + 1] for i, name in enumerate(self.output_names)}

        return outputs_dict


if __name__ == "__main__":
    _ = FCN()
    _ = NetHFM()
