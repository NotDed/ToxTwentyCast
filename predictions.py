import pandas as pd
import sys
import torch
import json

from transformers import AutoTokenizer
from modelFunctions import predict, multiPredict

seyonecModel = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'
tokenizer = AutoTokenizer.from_pretrained(seyonecModel, padding=True)

PATH = input('ingrese la direccion del modelo: ')
model = torch.load(PATH)

# Optional model configuration

#i: individual f: archivo
modo = sys.argv[1]

'''
si esta en modo individual mol se refiere a una cadena selfie
si esta en modo archivo mol se refiere a el path de el archivo con
las diferentes cadenas de selfies
'''
mol = sys.argv[2]


'''
save se refiere a la direccion de archivo .bin de el modelo
que se va a usar para la prediccion
'''
save = sys.argv[3]

if modo == 'F':
    resultados = multiPredict(model, tokenizer, mol)
else:
    resultados = predict(model, tokenizer, mol)
    
print('los resultado de esta prediccion son los siguientes: ')
print(resultados)
    