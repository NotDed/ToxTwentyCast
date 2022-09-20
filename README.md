# Fine Tuning Roberta for Molecular Toxicity Analysis

## Model training
para un entrenamiento con el dataset toxtwentycast usar el comando:
´python TTC3.py´

## Model prediction
Para usar el predictor ´predictions.py´ usamos el comando:
´python predictions.py <modo> <mol> <save>´

donde:
´modo´ es el tipo de prediccion que queremos realizar ya sea para un selfie escrito directamente en la linea de comandos o un archivo que contenga una o mas selfies
´i´: individual para pruebas con una sola cadena selfie
´f´: archivo para pruebas con archivos que contengan multiples cadenas selfies

´mol´ si se esta en modo individual se refiere a una cadena selfie si se esta en modo archivo se refiere a el path de el archivo con las diferentes cadenas de selfies

´save´ se refiere a la direccion de archivo .bin de el modelo que se va a usar para la prediccion


## Api prediction

