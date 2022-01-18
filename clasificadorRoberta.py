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
        #self.d1 = torch.nn.Dropout(p = 0.3, inplace=False)
        self.l1 = torch.nn.Linear(768, 768)
        #self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(p=0.3, inplace=False)
        self.l2 = torch.nn.Linear(64, 2)
        #self.act3 = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:,0]
        pooler = self.l1(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.d2(pooler)
        output = self.l2(pooler)
        #x = self.act3(x)
        return output