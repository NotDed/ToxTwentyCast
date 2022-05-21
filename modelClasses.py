import torch
import transformers
from torch.utils.data import Dataset
from transformers import RobertaModel

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0]* padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
        
#Creating the Neural Network for Fine Tuning

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k")
        #self.drop1 = torch.nn.Dropout(0.2)
        self.pre_classifier = torch.nn.Linear(768, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(64, 1)
        #self.act = torch.nn.Softmax(dim=1)
        
        # self.pre_classifier = torch.nn.Linear(768, 768)
        # self.norm1 = torch.nn.LayerNorm(768)
        # self.dropout = torch.nn.Dropout(0.2)
        # self.classifier = torch.nn.Linear(768, 384)
        # self.norm2 = torch.nn.LayerNorm(384)
        # self.classifier1 = torch.nn.Linear(384, 192)
        # self.norm3 = torch.nn.LayerNorm(192)
        # self.classifier2 = torch.nn.Linear(192, 1)
        # # self.norm4 = torch.nn.LayerNorm(1)
        # self.threshold = torch.nn.Threshold(0.8, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state =output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        pooler = self.pre_classifier(pooler)
        pooler = torch.tanh(pooler)
        pooler = self.dropout(pooler)
        pooler = self.classifier(pooler)
        #output = self.act(pooler)
        #output = self.threshold(output)
        # hidden_state = output_1[0]

        # pooler = hidden_state[:, 0]
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.norm1(pooler)
        # pooler = self.dropout(pooler)
        # pooler = self.classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.norm2(pooler)
        # pooler = self.dropout(pooler)
        # pooler = self.classifier1(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.norm3(pooler)
        # pooler = self.dropout(pooler)
        # output = self.classifier2(pooler)
        # output = self.threshold(pooler)
        # pooler = self.classifier2(pooler)
        return output



        