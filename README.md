<div align="center">

<img src="http://drive.google.com/uc?export=view&id=1Sqz8yYnij-7Vjl-4laOxBBCdhE0eDCDe" width="400">
</br>
</br>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rezaakb/pinns-torch/blob/main/tutorials/0-Schrodinger.ipynb)

<a href="https://openreview.net/forum?id=nl1ZzdHpab">[Paper]</a> - <a href="https://github.com/rezaakb/pinns-tf2">[TensorFlow v2]</a> - <a href="https://github.com/maziarraissi/PINNs">[TensorFlow v1]</a>
</div>

## Description

Our package introduces Physics-Informed Neural Networks (PINNs) implemented using PyTorch. The standout feature is the incorporation of CUDA Graphs and JIT Compilers (TorchScript) for compiling models, resulting in significant performance gains up to 9x compared to the original TensorFlow v1 implementation.

<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1WVZSSQwFAyNAkSqNgvZqok2vkPhpoERy" width="1000">
</br>
<em>Each subplot corresponds to a problem, with its iteration count displayed at the
top. The logarithmic x-axis shows the speed-up factor w.r.t the original code in TensorFlow v1, and the y-axis illustrates the mean relative error.</em>
</div>
</br>


For more information, please refer to our paper:

<a href="https://openreview.net/forum?id=nl1ZzdHpab">PINNs-Torch: Enhancing Speed and Usability of Physics-Informed Neural Networks with PyTorch.</a> Reza Akbarian Bafghi, and Maziar Raissi. DLDE III, NeurIPS, 2023.

## Installation

PINNs-Torch requires following dependencies to be installed:

- [PyTorch](https://pytorch.org) >=2.0.0
- [PyTorch Lightning](https://lightning.ai/) >= 2.0.0
- [Hydra](https://hydra.cc/docs/intro/) >= 1.3

Then, you can install PINNs-Torch itself via \[pip\]:

```bash
pip install pinnstorch
```

If you intend to introduce new functionalities or make code modifications, we suggest duplicating the repository and setting up a local installation:

```bash
git clone https://github.com/rezaakb/pinns-torch
cd pinns-torch

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install package
pip install -e .
```

## Quick start

Explore a variety of implemented examples within the [examples](examples) folder. To run a specific code, such as the one for the Navier-Stokes PDE, you can use:

```bash
python examples/navier_stokes/train.py
```

You can train the model using a specified configuration, like the one found in [examples/navier_stokes/configs/config.yaml](examples/navier_stokes/configs/config.yaml). Parameters can be overridden directly from the command line. For instance:

```bash
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

To utilize our package, there are two primary options:

- Implement your training structures using Hydra, as illustrated in our provided examples.
- Directly incorporate our package to solve your custom problem.

For a practical guide on directly using our package to solve the Schrödinger PDE in a continuous forward problem, refer to our tutorial here: [tutorials/0-Schrodinger.ipynb](tutorials/0-Schrodinger.ipynb).

## Data

The data located on the server and will be downloaded automatically upon running each example.

## Contributing

As this is the first version of our package, there might be scope for enhancements and bug fixes. We highly value community contributions. If you find any issues, missing features, or unusual behavior during your usage of this library, please feel free to open an issue or submit a pull request on GitHub. For any queries, suggestions, or feedback, please send them to [Reza Akbarian Bafghi](https://www.linkedin.com/in/rezaakbarian/) at [reza.akbarianbafghi@colorado.edu](mailto:reza.akbarianbafghi@colorado.edu).

## License

Distributed under the terms of the \[BSD-3\] license, "pinnstorch" is free and open source software.

## Resources

We employed [this template](https://github.com/ashleve/lightning-hydra-template) to develop the package, drawing from its structure and design principles. For a deeper understanding, we recommend visiting their GitHub repository. We also recommend consulting the official documentation of [Hydra](https://hydra.cc/docs/intro/) and [PyTorch Lightning](https://lightning.ai/) for additional insights.


## Citation
If you find this useful in your research, please consider citing:
```
@inproceedings{
bafghi2023pinnstorch,
title={{PINN}s-Torch: Enhancing Speed and Usability of Physics-Informed Neural Networks with PyTorch},
author={Reza Akbarian Bafghi and Maziar Raissi},
booktitle={The Symbiosis of Deep Learning and Differential Equations III},
year={2023},
url={https://openreview.net/forum?id=nl1ZzdHpab}
}
```
