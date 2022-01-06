# Proto ToxTwentyCast with TrochText

# Conda enviroment

`conda create --name torchText python=3.7`

`conda activate torchText `

## Pytorch with CUDA support

`conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch`

# Required packages


## PyTorch and Torchtext
`pip install -U torch==1.8.0 torchtext==0.9.0`

## PyTorchLightning
`pip install git+https://github.com/PyTorchLightning/pytorch-lightning fsspec --no-deps --target=$nb_path`

## Weights and bias:
`pip install wandb`

## Transformers
`pip install transformers`