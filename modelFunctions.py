import torch
import wandb
import json

from sklearn.metrics import roc_auc_score, classification_report, recall_score
from tqdm import tqdm

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def avg(l):
    return sum(l)/len(l)

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch, model, training_loader, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    predictions = []
    truePred = []

    roc_auc_arr = []
    psc_arr = []
    
    pr_arr = {
      'truePred':[],
      'predictions':[]
    }

    model.train()
    for step, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        print('a')
        print(targets)
        print('c')
        print(outputs)
        print('b')
        #print(outputs)
        predictions.extend(outputs)
        truePred.extend(targets)
        try:
           roc_auc_arr.append(roc_auc_score(targets.cpu(), outputs.cpu()))
        except ValueError:
          pass

        try: 
          psc_arr.append(recall_score(targets.cpu(), outputs.cpu()))
        except ValueError:
          pass


        
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if step % 5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 

            wandb.log({'ACC': accu_step})
            wandb.log({'LOSS': loss_step})

            # print(f"Training Loss per 5000 steps: {loss_step}")
            # print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    wandb.log({'ACC': accu_step})
    wandb.log({'LOSS': loss_step})

    # wandb.log({'AUC-ROC' : wandb.plot.roc_curve(truePred, predictions, labels=[0, 1])})
    
    # pr_arr['truePred'].extend(truePred)
    # pr_arr['predictions'].extend(predictions)
    
    # wandb.log({'Precision_recall' : wandb.plot.pr_curve(pr_arr['truePred'], pr_arr['predictions'], labels=[0, 1])})

    wandb.log({'AVG-AUC': avg(roc_auc_arr)})
    wandb.log({'AVG-PSC': avg(psc_arr)})

    return model, epoch_accu


def valid(model, testing_loader, loss_function):
    model.eval()   
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    
    y_pred = []
    y_true = []
    
    roc_auc_arr_valid = []
    psc_arr_valid = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            #print(outputs)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
        
            try:
              roc_auc_arr_valid.append(roc_auc_score(targets.cpu().detach().numpy(), torch.argmax(outputs.cpu(), axis=-1)).tolist())
            except ValueError:
              pass

            try: 
              psc_arr_valid.append(recall_score(targets.cpu().detach().numpy(), torch.argmax(outputs.cpu(), axis=-1)).tolist())
            except ValueError:
              pass
            
            print(outputs.tolist())
            y_pred.extend(outputs.tolist())
            y_true.extend(targets.tolist())
            #y_pred.extend(torch.argmax(outputs.cpu(), axis=-1).tolist())
            #y_true.extend(targets.tolist())

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if step%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                
                wandb.log({'ACC-T': accu_step})
                wandb.log({'LOSS-T': loss_step})
                
    y_pred = torch.argmax(torch.FloatTensor(y_pred), axis=-1).tolist()
    
    print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
          
    wandb.log({'valid-ACC': accu_step})
    wandb.log({'Valid-LOSS': loss_step})
    # wandb.log({'Valid-AVG-AUC': avg(roc_auc_arr_valid)})
    # wandb.log({'Valid-AVG-PSC': avg(psc_arr_valid)})
    #wandb.log({'AUC-ROC': wandb.plot.roc_curve(y_true, y_pred, labels=[0, 1])})
    #wandb.log({'Precision_recall': wandb.plot.pr_curve(y_true, y_pred, labels=[0, 1])})
    
    return epoch_accu
  
  
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