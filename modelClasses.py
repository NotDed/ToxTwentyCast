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
        super (RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained ("seyonec/BPE_SELFIES_PubChem_sharde0_166_5k")
        self.relu = torch.nn.ReLU()
        self.pre_classifier = torch.nn. Linear (768, 768)
        #self.dropout = torch.nn. Dropout (0.3)
        self.classifier1 = torch.nn. Linear (768, 500) #1
        self.classifier2 = torch.nn. Linear(500, 300) #2
        self.classifier3 = torch.nn. Linear(300, 100) #3
        self.classifier4 = torch.nn. Linear (100, 50) #4
        self.classifiers = torch.nn. Linear(50, 2) #5
        self.act = torch.nn.Softmax(dim=1)
        # super(RobertaClass, self).__init__()
        # self.l1 = RobertaModel.from_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k")
        # self.d1 = torch.nn.Dropout(p = 0.2, inplace=False) 
        # self.pre_classifier = torch.nn.Linear(768, 64)
        # self.batchnorm2 = torch.nn.BatchNorm1d(64)
        # self.d2 = torch.nn.Dropout(p = 0.2, inplace=False) 
        # self.classifier2 = torch.nn.Linear(64, 32)
        # self.batchnorm3 = torch.nn.BatchNorm1d(32)
        # self.d3 = torch.nn.Dropout(p = 0.2, inplace=False) 
        # self.classifier3 = torch.nn.Linear(32, 1)
        # self.activation = torch.nn.Sigmoid()


    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state= output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier (pooler)
        pooler = self.relu(pooler)
        #pooler = self.dropout (pooler)
        pooler = self.classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.classifier2(pooler)
        pooler = self.relu (pooler)
        pooler = self.classifier3(pooler)
        pooler = self.relu (pooler)
        pooler = self.classifier4(pooler)
        pooler = self.relu(pooler)
        output = self.classifiers(pooler)
        output = self.act(output)
        print("this output: ", output)
        return output
        
        # output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # hidden_state =output_1[0]
        # pooler = hidden_state[:, 0]
        # pooler = self.d1(pooler)
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.functional.relu(pooler)
        # pooler = self.batchnorm2(pooler)
        # pooler = self.d2(pooler)
        # pooler = self.classifier2(pooler)
        # pooler = torch.nn.functional.relu(pooler)
        # pooler = self.batchnorm3(pooler)
        # pooler = self.d3(pooler)
        # pooler = self.classifier3(pooler)
        # pooler = self.activation(pooler)
        # output = torch.squeeze(pooler)
        # return output



        