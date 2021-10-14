# ToxTwentyCast

# Conda enviroment

`conda create --name ToxTwentyCast python=3.7`

`conda activate ToxTwentyCast `

# Required packages

`conda install pandas`

`conda install openpyxl`

`pip install sklearn`

`pip install selfies`

## Weights and bias:
`pip install wandb`


## Pytorch with CUDA support

`conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch`

## Simple transformers (0.48.6):

`pip install 'simpletransformers==0.48.6'`

## Transformers (3.3.1):

`pip install 'transformers==3.3.1'`

# Execution format for training

`python TTCBin.py <modelo seyonec> <epochs>`

## ex:

`python TTCBin.py s160k 15`

## Available models

`'s50k' == 'seyonec/BPE_SELFIES_PubChem_shard00_50k'`

`'s70k' == 'seyonec/BPE_SELFIES_PubChem_shard00_70k'`

`'s120k' == 'seyonec/BPE_SELFIES_PubChem_shard00_120k'`

`'s150k' == 'seyonec/BPE_SELFIES_PubChem_shard00_150k'`

`'s160k' == 'seyonec/BPE_SELFIES_PubChem_shard00_160k'`

# Execution format for predictions

`python Predicciones.py <mode> <selfie mol> <model path>`

## Modes
`f` for file testing (.xslx)
`i` for individual testing
## ex:

`python Predicciones.py [C][C][O][C][=C][C][=C][N] /outputs/best_model`
