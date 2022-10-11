#Fine Tuning Roberta for Sentiment Analysis

# Importing the libraries needed
import json
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

import wandb

# Setting up the device for GPU usage

from torch import cuda


from modelClasses import SentimentData, RobertaClass
from modelFunctions import train, valid

new_df = pd.read_csv('~/ToxTwentyCast/dataset/toxTwentyCast.csv')
    

# Defining some key variables that will be used later on in the training
def mainTrain():
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    MODEL_NAME = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True)
            
    #data split
    train_size = 0.8
    train_data=new_df.sample(frac=train_size,random_state=200)
    test_data=new_df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    training_set = SentimentData(train_data, tokenizer, MAX_LEN)
    testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

    #trining params
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    device = torch.device('cuda')
    model = RobertaClass()
    model = torch.nn.DataParallel(model)
    model.to(device)

    #Fine Tuning the Model

    # Creating the loss function and optimizer
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

    EPOCHS = 100
    output_model_name = input('Ingrese el nombre de el modelo de salida sin usar espacios: ')

    predicciones = {}
    for epoch in range(EPOCHS):
        model = train(epoch, model, training_loader, loss_function, optimizer)
        
        with torch.set_grad_enabled(False):
            y_pred, y_target, avg_loss = valid(model, testing_loader, loss_function)
            predicciones[str(epoch+1)] = [y_pred, y_target]
            print(len(y_pred))
    
    print('predictions by epoch')
    print(json.dumps(predicciones, indent=4))
    
    predictionsFileName = 'predictions_{}.json'.format(output_model_name)
    with open(predictionsFileName, "w") as outfile:
        json.dump(predicciones, outfile)
        


    output_model_file = '{}.bin'.format(output_model_name) #'pytorch_roberta_sentiment.bin'
    output_vocab_file = './'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('All files saved')
    

if __name__ == '__main__': 
    mainTrain()    