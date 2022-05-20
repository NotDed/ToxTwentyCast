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
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("seyonec/BPE_SELFIES_PubChem_shard00_160k")
        self.d1 = torch.nn.Dropout( 0.25)
        self.l1 = torch.nn.Linear(768, 768)
        self.d2 = torch.nn.Dropout(0.25)
        self.l2 = torch.nn.Linear(768, 2)
        self.act = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = torch.nn.ReLU()(x)
        x = self.d2(x)
        x = self.l2(x)
        x = self.act(x)
        return x


        
  
        
        