import config
import dataset
import torch
import torch.nn as nn
import pandas as pd
import engine
import numpy as np


from model import ROBERTAClassifier
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna('none')
    # dfx.labels = dfx['labels']

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.labels#.values
    ) 

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    df_train = df_train.dropna()

    train_dataset = dataset.BERTDataset(
        text = df_train.text.values,
        labels = df_train.labels.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE, 
        num_workers = 4
    )

    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values,
        labels=df_valid.labels.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config.VALID_BATCH_SIZE, 
        num_workers = 1
    )


    device = torch.device("cuda")
    model = ROBERTAClassifier()
    # model.to_device()

    param_optimizer = list (model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters  =[
        {'params': [p for n, p in param_optimizer if not any (nd in n for nd in no_decay)], 'weight_decay':0.001},
        {'params': [p for n, p in param_optimizer if any (nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    num_train_steps  =int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr = 3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps = num_train_steps
    )

    model = nn.DataParallel(model)

    best_accuracy = 0 
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Acuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ =="__main__":
    run()