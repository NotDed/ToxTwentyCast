from operator import index
import pandas as pd
import sys
import torch
import json

from transformers import AutoTokenizer
from modelFunctions import predict, multiPredict



# Optional model configuration
'''
modos
i: individual para pruebas con una sola cadena selfie

f: archivo para pruebas con archivos que contengan multiples cadenas selfies
'''
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

seyonecModel = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'
tokenizer = AutoTokenizer.from_pretrained(seyonecModel, padding=True)

model = torch.load(save).cuda()

resultados = {}

resultados = predict(model, tokenizer, mol)

# if modo == 'f':
#     resultados = multiPredict(model, tokenizer, mol)
# else:
#     resultados[mol] = predict(model, tokenizer, mol)
    
print('los resultado de esta prediccion son los siguientes: ')
print(json.dumps(resultados, indent = 2))

# df = {
#     "mol" : list(resultados.keys()),
#     "result" : list(resultados.values())
# }

# df = pd.DataFrame({'Mol': resultados.keys(), 'Result': resultados.values()})

# positivos = len(df[ df['Result'] == 1 ])
# negativos = len(df[ df['Result'] == 0 ])

# print("Resultados positivos: {}".format(positivos))
# print("{}%".format(round(positivos*100/len(df), 2)))
# print("Resultados negativos: {}".format(negativos))
# print("{}%".format(round(negativos*100/len(df), 2)))