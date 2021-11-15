#pip install -U torch==1.8.0 torchtext==0.9.0
#pip install transformers
#pip install git+https://github.com/PyTorchLightning/pytorch-lightning fsspec --no-deps --target=$nb_path
#pip install wandb --quiet

#------------------------------------Imports------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torchtext
from torchtext import data
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM

import wandb

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import utilityFunctions
import modelFunctions

#-------------------------------------Paths-------------------------------------

data_path = '/dataset/toxTwentyCast.csv'
output_path = '/outputs/'

#-------------------------------------Dataset Load------------------------------

df = pd.read_csv(data_path)

#-------------------------------------Wandb login-------------------------------

wandb.login()

#-------------------------------------Tokenizer definition----------------------

BERT_MODEL_NAME = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

MAX_SEQ_LEN = 256
BATCH_SIZE = 16

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

#-------------------------------------Dataloaders-------------------------------

label_field = torchtext.legacy.data.Field(sequential=False, use_vocab=False, batch_first=True)
text_field = torchtext.legacy.data.Field(use_vocab=False,
                   tokenize=tokenizer.encode,
                   include_lengths=False,
                   batch_first=True,
                   fix_length=MAX_SEQ_LEN,
                   pad_token=PAD_INDEX,
                   )

fields = {'text' : ('text', text_field), 'labels' : ('labels', label_field)}

train_data, valid_data, test_data = torchtext.legacy.data.TabularDataset(data_path,
                                                   format='CSV',
                                                   fields=fields,
                                                   skip_header=False).split(split_ratio=[0.70, 0.2, 0.1],
                                                                            stratified=True,
                                                                            strata_field='labels')

train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               sort_key=lambda x: len(x.text),
                                               sort=True,
                                               sort_within_batch=False)

test_iter = Iterator(test_data, batch_size=BATCH_SIZE, train=False, shuffle=False, sort=False)

#-------------------------------------Main training loop------------------------
NUM_EPOCHS = 3
steps_per_epoch = len(train_iter)

model = ROBERTAClassifier()


optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=steps_per_epoch*1,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

print("======================= Start pretraining ==============================")

pretrain(model=model,
         train_iter=train_iter,
         valid_iter=valid_iter,
         optimizer=optimizer,
         scheduler=scheduler,
         num_epochs=NUM_EPOCHS,
         valid_period=927)

NUM_EPOCHS = 6
print("======================= Start training =================================")
optimizer = AdamW(model.parameters(), lr=2e-6)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=steps_per_epoch*2,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

train(model=model,
      train_iter=train_iter,
      valid_iter=valid_iter,
      optimizer=optimizer,
      scheduler=scheduler,
      num_epochs=NUM_EPOCHS,
         valid_period=927)

model = ROBERTAClassifier()
model = model.to(device)

load_checkpoint(output_path + '/model.pkl', model)

evaluate(model, test_iter)

print(len(train_data))
print(len(valid_data))
print(len(test_data))