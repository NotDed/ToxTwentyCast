#------------------------------------Imports------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torchtext
from torchtext import data
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification

import wandb

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import utilityFunctions
#-------------------------------------ROBERTA Classifier------------------------

class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, BERT_MODEL_NAME):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(BERT_MODEL_NAME, return_dict=False)
        self.d1 = torch.nn.Dropout(p = 0.15, inplace=False)
        self.l1 = torch.nn.Linear(768, 768)
        self.d2 = torch.nn.Dropout(p=0.2, inplace=False)
        self.l2 = torch.nn.Linear(768, 384)
        self.d3 = torch.nn.Dropout(p=0.2, inplace=False)
        self.l3 = torch.nn.Linear(384, 192)
        self.d4 = torch.nn.Dropout(p=0.2, inplace=False)
        self.l4 = torch.nn.Linear(192, 2)
        self.act3 = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.d2(x)
        x = self.l2(x)
        x = torch.nn.ReLU()(x)
        x = self.d3(x)
        x = self.l3(x)     
        x = torch.nn.ReLU()(x)
        x = self.d4(x)
        x = self.l4(x)
        x = self.act3(x)
        return x