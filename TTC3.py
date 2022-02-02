#Fine Tuning Roberta for Sentiment Analysis

# Importing the libraries needed
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

import wandb

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from modelClasses import SentimentData, RobertaClass
from modelFunctions import train, valid

new_df = pd.read_csv('~/ToxTwentyCast/dataset/toxTwentyCast.csv')

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE =64 
VALID_BATCH_SIZE = 32
# EPOCHS = 1
LEARNING_RATE = 3e-05
MODEL_NAME = 'seyonec/BPE_SELFIES_PubChem_shard00_166_5k'
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

model = RobertaClass()
model.to(device)

#Fine Tuning the Model

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

EPOCHS = 15

#-------------------------------------Wandb login-------------------------------
wandb.login()
run = wandb.init(project="FineT-Roberta")


for epoch in range(EPOCHS):
    model, _ = train(epoch, model, training_loader, loss_function, optimizer)
    wandb.log({'EPOCH': epoch})
    
#Validating the Model
acc = valid(model, testing_loader, loss_function)
print("Accuracy on test data = %0.2f%%" % acc)

run.finish()

#Saving the Trained Model Artifacts for inference

output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')