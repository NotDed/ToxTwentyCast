import pandas as pd
import sys
import torch

from simpletransformers.classification import ClassificationModel, ClassificationArgs


# Optional model configuration
modo = sys.argv[1] #i: individual f: archivo

mol = sys.argv[2]

save = sys.argv[3]

if modo == 'f':
    test_df = pd.read_excel(mol, header=(0))
    test = list(test_df['selfies'])
else:
    test_df = pd.DataFrame.from_dict({'selfies': [mol], 'Toxicidad': [0]})
    test = [mol]

# Create a ClassificationModel
model_args = ClassificationArgs()
model_args.wandb_project = 'toxTwentyCastBin'

model = ClassificationModel(
    "roberta", save, args=model_args, use_cuda = False
)

predictions, raw_outputs = model.predict(test)

if modo == 'i':
    print('predictions: ')
    print(predictions, raw_outputs)

out_df = {
    'selfie':test,
    'real_tox': list(test_df['Toxicidad']),
    'predictions':predictions
    # 'raw_outputs':raw_outputs[]
}

out_df = pd.DataFrame(out_df)

predictionName = input('Indique el nombre de el archivo de predicciones: ')

out_df.to_csv('predictions/'+predictionName+'.csv', index=False)
