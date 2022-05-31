#Fine Tuning Roberta for Sentiment Analysis

# Importing the libraries needed
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
#new_df = pd.read_csv('~/ToxTwentyCast/dataset/Datasets NR/NR-AhR.csv')
#new_df = pd.read_csv('~/ToxTwentyCast/dataset/Tox21.csv')

    

# Defining some key variables that will be used later on in the training
def mainTrain():
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16 
    VALID_BATCH_SIZE = 8
    LEARNING_RATE = 3e-05
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

    EPOCHS = 1
    #-------------------------------------Wandb login-------------------------------
    wandb.login()
    run = wandb.init(project="FineT-Roberta")
    output_model_name = input('Ingrese el nombre de el modelo de salida sin usar espacios: ')

    for epoch in range(EPOCHS):
        model = train(epoch, model, training_loader, testing_loader, loss_function, optimizer)
        
    #Validating the Model
    
    # auc, auprc, f1, predictions, loss = valid(model, testing_loader, loss_function)
    
    with torch.set_grad_enabled(False):
        auc, auprc, f1, predictions, loss = valid(model, testing_loader, loss_function)
            
        print('Validation at Epoch {}, AUROC: {}, AUPRC: {}, F1: {}, LOSS: {}'.format(epoch + 1, auc, auprc, f1, loss))

    output_model_file = '{}.bin'.format(output_model_name) #'pytorch_roberta_sentiment.bin'
    output_vocab_file = './'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('All files saved')


if __name__ == '__main__': 
    mainTrain()
    #  study = optuna.create_study(direction="maximize")
    #  study.optimize(objective, n_trials = 4)

    #  print("best trial: ")
    #  trial_ = study.best_trial

    #  print(trial_.values)
    #  print(trial_.params)
    
# def objective(trial):
#     params = {
#         "MAX_SEQ_LEN": trial.suggest_int ("MAX_SEQ_LEN", 180, 250),
#         "BATCH_SIZE": trial.suggest_int ("BATCH_SIZE", 64, 84),
#         "lr": trial.suggest_loguniform("lr", 2e-6, 3e-5)
#     }
    
#     MODEL_NAME = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True)
    
#     #data split
#     train_size = 0.8
#     train_data = new_df.sample(frac=train_size,random_state=200)
#     test_data = new_df.drop(train_data.index).reset_index(drop=True)
#     train_data = train_data.reset_index(drop=True)
    
#     training_set = SentimentData(train_data, tokenizer, params['MAX_SEQ_LEN'])
#     testing_set = SentimentData(test_data, tokenizer, params['MAX_SEQ_LEN'])
    
#     #trining params
#     train_params = {'batch_size': params['BATCH_SIZE'],
#                     'shuffle': True,
#                     'num_workers': 0
#                     }

#     test_params = {'batch_size': params['BATCH_SIZE'],
#                     'shuffle': True,
#                     'num_workers': 0
#                     }
    
#     training_loader = DataLoader(training_set, **train_params)
#     testing_loader = DataLoader(testing_set, **test_params)
    
#     device = torch.device('cuda')
#     model = RobertaClass()
#     model = torch.nn.DataParallel(model)
#     model.to(device)
#     loss_function = torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(params =  model.parameters(), lr=params['lr'])

#     EPOCHS = 1

    
#     wandb.login()
#     run = wandb.init(project="FineT-Roberta")
    
#     for epoch in range(EPOCHS):
#         model, _ = train(epoch, model, training_loader, loss_function, optimizer)
#         wandb.log({'EPOCH': epoch})
        
#     #Validating the Model
#     acc = valid(model, testing_loader, loss_function)
    
#    # run.finish()
    
#     return acc
        
        