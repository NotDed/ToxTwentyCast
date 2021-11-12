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

from utiliyFunctions import *
#-------------------------------------ROBERTA Classifier------------------------

class ROBERTAClassifier(torch.nn.Module):
    def __init__(self):
        super(ROBERTAClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained(BERT_MODEL_NAME, return_dict=False)
        self.d1 = torch.nn.Dropout(p = 0.3, inplace=False)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(p=0.3, inplace=False)
        self.l2 = torch.nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x
