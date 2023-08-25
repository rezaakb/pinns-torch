<div align="center">

<img src="http://drive.google.com/uc?export=view&id=1JO83M12_y2F8h7QYZZSK5NXkRSdWnSqy" width="400">
</br>
</br>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<br>

</div>

## Description

Our package introduces Physics-Informed Neural Networks (PINNs) implemented using PyTorch. The standout feature is the incorporation of CUDAGraphs for compiling models, resulting in significant performance gains of 3x to 15x compared to traditional PyTorch implementations.

<div align="center">
<img src="http://drive.google.com/uc?export=view&id=1qbDpnSZiDRm5CQKjAUkNsfYcDqLEShQA" width="500">
</br>
<em>Comparing elapsed time for a single epoch in solving Navier-Stokes, Allen-Cahn, and Burgers partial differential equations using naive models and models compiled with CUDAGraphs. </em>
</div>

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
git clone https://github.com/rezaakb/pinnstorch
cd pinnstorch

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install package
pip install -e .
```

## Quick start

There are several implemented examples on [examples](examples) folder. For example, you can run the code corresponding to navier stokes pde:

```bash
python examples/navier_stokes/train.py
```

Train model with chosen experiment configuration; for example, from [examples/navier_stokes/configs/config.yaml](examples/navier_stokes/configs/config.yaml). You can override any parameter from command line like this

```bash
python examples/navier_stokes/train.py trainer.max_epochs=20 n_train=3000
```

## Data

The data located on the server and will be downloaded automatically upon running each example.

## Contributing

We greatly value contributions from the community. If you identify any missing features, encounter bugs, or notice unexpected behavior while utilizing this library, we kindly invite you to open an issue or submit a pull request on GitHub. Alternatively, please feel free to reach out to the authors directly. Your input is highly appreciated and will help enhance the quality of our project.

## License

Distributed under the terms of the \[BSD-3\] license, "pinnstorch" is free and open source software.

## Resources

We employed [this template](https://github.com/ashleve/lightning-hydra-template) to develop the package, drawing from its structure and design principles. For a deeper understanding, we recommend visiting their GitHub repository.

## Citation

```
...
```
