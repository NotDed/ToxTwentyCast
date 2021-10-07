# ToxTwentyCast

## Crear entorno
`conda create --name ToxTwentyCast python=3.7`

## Activar entorno

`conda activate ToxTwentyCast `

# Paquetes necesarios
## Pandas:

`conda install pandas`

## Sklearn:
`conda install sklearn`

## Selfies:
`conda install selfies`

## Weights and bias:
`conda install -c conda-forge wandb`


## Pytorch with CUDA support

`conda install pytorch>=1.6 cudatoolkit=10.2 -c pytorch`

## Simple transformers (0.48.6):

`pip install 'simpletransformers==0.48.6'`

## Transformers (3.3.1):

`pip install 'transformers==3.3.1'`

# Formato de ejecución para entrenamiento

`python TTCBin.py <modelo seyonec> <epochs>`

ex:

`python TTCBin.py s160k 15`

## Modelos disponibles

`'s120k' == 'seyonec/BPE_SELFIES_PubChem_shard00_120k'`

`'s150k' == 'seyonec/BPE_SELFIES_PubChem_shard00_150k'`

`'s160k' == 'seyonec/BPE_SELFIES_PubChem_shard00_160k'`
