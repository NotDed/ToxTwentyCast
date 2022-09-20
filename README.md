# Fine Tuning Roberta for Molecular Toxicity Analysis

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

## Model training
para un entrenamiento con el dataset toxtwentycast usar el comando:
`python TTC3.py`

## Model prediction
Para usar el predictor `predictions.py` usamos el comando:
`python predictions.py <modo> <mol> <save>`

donde:
`modo` es el tipo de prediccion que queremos realizar ya sea para un selfie escrito directamente en la linea de comandos o un archivo que contenga una o mas selfies
`i`: individual para pruebas con una sola cadena selfie
`f`: archivo para pruebas con archivos que contengan multiples cadenas selfies

`mol` si se esta en modo individual se refiere a una cadena selfie si se esta en modo archivo se refiere a el path de el archivo con las diferentes cadenas de selfies

`save` se refiere a la direccion de archivo .bin de el modelo que se va a usar para la prediccion


## Api prediction

