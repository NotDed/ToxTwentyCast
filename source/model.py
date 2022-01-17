import config
import transformers
import torch.nn as nn
class ROBERTAClassifier(nn.Module):
    def __init__(self):
        super(ROBERTAClassifier, self).__init__()

        self.roberta = transformers.RobertaModel.from_pretrained(config.BERTH_PATH)
        self.roberta_drop = nn.Dropout(0.2)
        self.l1 = torch.nn.Linear(768, 1)
        
    def forward(self, ids, attention_mask, token_type_ids):
        #_ = lasthiddenstate,o2pooler output
        _, o2 = self.roberta(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        #robertoutput
        bo =  self.roberta_drop(o2)
        output = self.l1(bo)
        return output