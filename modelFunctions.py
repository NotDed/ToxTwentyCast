import torch
import wandb
import json
import copy

from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score
from torch import cuda
from tqdm import tqdm
import numpy as np

device = 'cuda' if cuda.is_available() else 'cpu'


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def avg(l):
    return sum(l)/len(l)

# Defining the training function on the 80% of the dataset for tuning the distilbert model
def getDataFromLoader(loaderData):
    #model inputs
    ids = loaderData['ids'].to(device, dtype = torch.long)
    mask = loaderData['mask'].to(device, dtype = torch.long)
    token_type_ids = loaderData['token_type_ids'].to(device, dtype = torch.long)
    
    #targets
    targets = loaderData['targets'].to(device, dtype = torch.float32)
    
    return ids, mask, token_type_ids, targets
  

def valid(model, loader, loss_function):
    y_pred = []
    y_target = []
    avg_loss = []
    
    torch.cuda.empty_cache()
    
    model.eval()
    for step, data in tqdm(enumerate(loader, 0)):

        ids, mask, token_type_ids, targets = getDataFromLoader(data)
        
        outputs = model(ids, mask, token_type_ids)
        outputs = outputs.to(device, dtype = torch.float32)
        print(outputs)
        loss = loss_function(outputs, targets)
        outputs = outputs.to(device, dtype = torch.float32)
        outputs = outputs.flatten()
        
        avg_loss.append(loss.item())
        
        outputs = outputs.detach().cpu().numpy()
        targets = targets.to('cpu').numpy()
        
        y_pred.extend(outputs.flatten().tolist())
        y_target.extend(targets.flatten().tolist())
      
    #Metrics
    avg_loss = sum(avg_loss)/len(avg_loss)
    
    return y_pred, y_target, avg_loss

def train(epoch, model, loader, loss_function, optimizer):
    y_pred = []
    y_target = []
    
    model.train()
    for step, data in tqdm(enumerate(loader, 0)):
        ids, mask, token_type_ids, targets = getDataFromLoader(data)
        
        #model ouputs
        outputs = model(ids, mask, token_type_ids)
        outputs = outputs.to(device, dtype = torch.float32)
        outputs = outputs.flatten()
        
        #loss meassure
        loss = loss_function(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #printing loss every n steps
        if (step % 100 == 0):
            print('Training at Epoch {} iteration {} with loss {}'.format(epoch + 1, step, loss.cpu().detach().numpy()))

    return model

def predict(model, tokenizer, text, threshold = 0.26):
  model.eval()
  inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_token_type_ids=True)
  
  ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
  mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
  token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
  
  outputs = model(ids, mask, token_type_ids)
  outputs = outputs.to(device, dtype = torch.float32)
  outputs = outputs.flatten().tolist()[0]
  predValue = outputs
#   predValue = 1 if outputs >= threshold else 0

  return {text: {'pred' : predValue, 'linear value': outputs, 'with threshold' : threshold}}

def multiPredict(model, tokenizer, selfies, threshold = 0.26):
    predictions = {}
        
    for T in selfies:
        out = predict(model, tokenizer, T, threshold)
        predictions.update(out)
    
    return predictions