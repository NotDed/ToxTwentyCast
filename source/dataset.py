import config
from config import MAX_LEN
import torch

class BERTDataset:
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self,item):
        text = str(self.text)
        text = " ".join(text.split())

        inputs = self.tokenizer.encode(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len
        )
        print(inputs)
        print(type(inputs))
        input()
        
        ids=inputs["inputs_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]


        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0]* padding_length)

        return{
            'ids': torch.tensor(ids, dytype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)

        }
