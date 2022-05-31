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
    loss_accumulate = 0.0
    count = 0.0
    
    torch.cuda.empty_cache()
    
    model.eval()
    
    for step, data in tqdm(enumerate(loader, 0)):
      ids, mask, token_type_ids, targets = getDataFromLoader(data)
      
      #model ouputs
      outputs = model(ids, mask, token_type_ids)
      outputs = outputs.to(device, dtype = torch.float32)
      
      loss = loss_function(outputs, targets)
      
      loss_accumulate += loss
      count += 1
      
      outputs = outputs.detach().cpu().numpy()
      targets = targets.to('cpu').numpy()
      
      y_pred.extend(outputs.flatten().tolist())
      y_target.extend(targets.flatten().tolist())
      
    #Metrics
    loss = loss_accumulate/count
    
    
    #print(y_target, y_pred)
    fpr, tpr, thresholds = roc_curve(y_target, y_pred)
    precision = tpr / (tpr + fpr)
    
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    
    print("optimal threshold: " + str(thred_optim))
    wandb.log({'optimal threshold': thred_optim})
    y_predictions = [1 if i else 0 for i in (y_pred >= thred_optim)]
    #print(y_predictions)
    
    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    wandb.log({'AUROC': auc_k})
    #wandb.log({'AUROC': float(auc_k)})
    #auprc = str(average_precision_score(y_target, y_predictions)
    print("AUPRC: "+ str(average_precision_score(y_target, y_predictions)))
    wandb.log({'AUPRC': average_precision_score(y_target, y_predictions)})
    #wandb.log({'AUPRC': float(average_precision_score(y_target, y_predictions))})
    #print("AUPRC: "+ str(average_precision_score(y_target, y_pred)))
    
    cm1 = confusion_matrix(y_target, y_predictions)
    print('Confusion Matrix : \n', cm1)
    wandb.log({'Confusion Matrix': cm1})
    print('Recall : ', recall_score(y_target, y_predictions))
    wandb.log({'Recall': recall_score(y_target, y_predictions)})
    print('Precision : ', precision_score(y_target, y_predictions))
    wandb.log({'Precision': precision_score(y_target, y_predictions)})
    total1=sum(sum(cm1))
    
    #accuracy from confusion matrix
    accuracy1 = (cm1[0,0] + cm1[1,1]) / total1
    print ('Accuracy : ', accuracy1)
    wandb.log({'Accuracy': accuracy1})

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    
    return roc_auc_score(y_target, y_pred), average_precision_score(y_target, y_pred), f1_score(y_target, outputs), y_pred, loss.item()
  
  

def train(epoch, model, loader, validationLoader, loss_function, optimizer):
  
    lossHistory = []
    
    model.train()
    for step, data in tqdm(enumerate(loader, 0)):
        ids, mask, token_type_ids, targets = getDataFromLoader(data)
        
        #model ouputs
        outputs = model(ids, mask, token_type_ids)
        outputs = outputs.to(device, dtype = torch.float32)
        
        #loss meassure
        loss = loss_function(outputs, targets)
        lossHistory.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #printing loss every n steps
        if (step % 100 == 0):
            print('Training at Epoch {} iteration {} with loss {}'.format(epoch + 1, step, loss.cpu().detach().numpy()))



    #validation phase
    with torch.set_grad_enabled(False):
        auc, auprc, f1, predictions, loss = valid(model, validationLoader, loss_function)
        wandb.log({'AUROC': auc , 'AUPRC': auprc, 'F1': f1, 'Test loss': loss})
        print('Validation at Epoch {}, AUROC: {}, AUPRC: {}, F1: {}'.format(epoch + 1, auc, auprc, f1))
        
    print('The Total Accuracy for Epoch {}'.format(epoch))

    return model

  
def predict(model, tokenizer, text):
  
  model.eval()
  
  inputs = tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          return_token_type_ids=True
      )
  
  ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
  mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
  token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
  
  outputs = model(ids, mask, token_type_ids).squeeze().cpu()
  
  outputs = torch.argmax(torch.FloatTensor(outputs), axis=-1).tolist()
  
  return outputs
    
    
def multiPredict(model, tokenizer, path_text):
    predictions = {}
    with open(path_text) as json_file:
        dataT = json.load(json_file)
        
    for T in dataT:
        predictions[T] = predict(model, tokenizer, T)
    
    return predictions