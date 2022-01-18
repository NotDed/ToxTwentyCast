#------------------------------------Imports------------------------------------
# conda install -c conda-forge optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score

import torch
import torchtext
from torchtext import data
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM

import wandb

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


import utilityFunctions
#-------------------------------------Paths-------------------------------------

data_path = '~/ToxTwentyCast/dataset/toxTwentyCast.csv'
output_path = 'outputs/'

#-------------------------------------Utility Functions-------------------------

def avg(l):
    return sum(l)/len(l)

def save_checkpoint(path, model, valid_loss):
    torch.save({'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}, path)


def load_checkpoint(path, model):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model_state_dict'])

    return state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, path)


def load_metrics(path):
    state_dict = torch.load(path)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'],

#------------------------------------Pretrain-----------------------------------


def pretrain(model,
             optimizer,
             train_iter,
             valid_iter,
             PAD_INDEX,
             UNK_INDEX,
             scheduler = None,
             valid_period = 1600,
             num_epochs = 1):

    # Pretrain linear layers, do not train bert
    #for param in model.roberta.parameters():
        #param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = True
        

    
    model.train()

    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    global_step = 0
    predictions = []
    truePred = []   
    accuracy= []

    # Train loop
    for epoch in range(num_epochs):
        for (source, target), _ in train_iter:
            #
            mask = (source != PAD_INDEX).type(torch.uint8)

            y_pred = model(input_ids=source,
                           attention_mask=mask).cuda()
            
            target = target.cuda()

            print('target: ',target)
            print('y_pred: ',y_pred)
            print('source: ',source.shape, ' target: ',target.shape, ' y_pred: ',y_pred.shape)

            loss = torch.nn.CrossEntropyLoss()(y_pred, target)

            loss.backward()
            #loss.device
            #Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            print(global_step, len(train_iter), valid_period)

            if global_step % valid_period == 0:
                model.eval()
                print(len(valid_iter))
                

                acc = []
                auc = []
                psc = []
                recall = []
                with torch.no_grad():
                    for (source, target), _ in valid_iter:
                        
                        #target = target.to(device)

                        
                        mask = (source != PAD_INDEX).type(torch.uint8)

                        y_pred = model(input_ids=source, attention_mask=mask).cuda()
                        predictions.extend(y_pred.tolist())
                        
                        target = target.cuda()
                        truePred.extend(target.tolist())
                        

                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                        #pdb.set_trace()
                        acc.append(accuracy_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        
                        #auc.append(roc_auc_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        
                        #try:
                        #   lol=(roc_auc_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        #except ValueError:
                        #    lol=0

                        auc.append(roc_auc_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        psc.append(precision_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        
                        recall.append(recall_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        #wandb.log({'AUC-ROC' : wandb.plot.roc_curve(target.cpu(),y_pred.cpu(), labels=[0, 1])})
                        #wandb.log({'Precision_recall' : wandb.plot.pr_curve(target.cpu(),y_pred.cpu(), labels=[0, 1])})
                        

                        valid_loss += loss.item()

                
                
                acc =  avg(acc[:-1])
                auc = avg(auc[:-1])
                psc = avg(psc[:-1])
                recall = avg(recall[:-1])
                
                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)

                model.train()
                # print summary
                
                wandb.log({'epoch': epoch, 'global_step': global_step, 'acc': acc, 'train_loss': train_loss,
                'valid_loss': valid_loss, 'auc': auc, 'recall':recall, 'psc':psc})
                print('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                train_loss = 0.0
                valid_loss = 0.0

    # Set bert parameters back to trainable
    
    wandb.log({'PRE-AUC-ROC' : wandb.plot.roc_curve(truePred, predictions, labels=[0, 1])})
    wandb.log({'PRE-Precision_recall' : wandb.plot.pr_curve(truePred, predictions, labels=[0, 1])})
    
    #for param in model.module.roberta.parameters():
    #    param.requires_grad = True

    #print('Pre-training done!')

#------------------------------------Train--------------------------------------

def train(model,
          optimizer,
          train_iter,
          valid_iter,
          PAD_INDEX,
          UNK_INDEX,
          scheduler = None,
          num_epochs = 1,
          valid_period= 1600,
          output_path = output_path):

    # Initialize losses and loss histories
    for param in model.module.roberta.parameters():
       param.requires_grad = False

    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('Inf')

    global_step = 0
    global_steps_list = []
    
    predictions = []
    truePred = []
    Acc = []
    model.train()
    
    # Train loop
    for epoch in range(num_epochs):
        #print(epoch)
        for (source, target), _ in train_iter:
            mask = (source != PAD_INDEX).type(torch.uint8)

            y_pred = model(input_ids=source, attention_mask=mask).cuda() 

            target = target.cuda()
             
            #print('target: ',target)
            #print('y_pred: ',y_pred)
            #print('source: ',source.shape, ' target: ',target.shape, ' y_pred: ',y_pred.shape)
            #output = model(input_ids=source,
             #             labels=target,
              #            attention_mask=mask)

            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
            #loss = output[0]

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            #print(global_step, len(train_iter), valid_period * (epoch+1))

            if global_step % valid_period == 0:
                model.eval()
                acc = []
                auc = []
                psc = []
                recall = []
                with torch.no_grad():

                    for (source, target), _ in valid_iter:
                        mask = (source != PAD_INDEX).type(torch.uint8)
                        
                        y_pred = model(input_ids=source, attention_mask=mask).cuda()
                        predictions.extend(y_pred.tolist())
                        
                        target = target.cuda()
                        truePred.extend(target.tolist())
                        #output = model(input_ids=source,
                        #               labels=target,
                        #               attention_mask=mask)

                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)

                        acc.append(accuracy_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        y_pred = model(input_ids=source, attention_mask=mask).cuda()
                        predictions.extend(y_pred.tolist())
                        try:
                            lol=(roc_auc_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        except ValueError:
                            lol=0

                        auc.append(roc_auc_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                        psc.append(precision_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))
                
                        recall.append(recall_score(target.cpu(), torch.argmax(y_pred.cpu(), axis=-1).tolist()))

                        valid_loss += loss.item()
                        
                        #print(valid_loss)

                # Store train and validation loss history
                acc =  avg(acc[:-1])
                auc =  avg(auc[:-1])
                psc =  avg(psc[:-1])
                recall =  avg(recall[:-1])
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)
                Acc.append(acc)
                
                

                # print summary
                
                wandb.log({'epoch': epoch, 'global_step': global_step, 'acc': acc, 'train_loss': train_loss,
                           'valid_loss': valid_loss, 'auc': auc, 'recall':recall, 'psc':psc})
                print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                # checkpoint
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(output_path + 'model.pkl', model, best_valid_loss)
                    save_metrics(output_path + 'metric.pkl', train_loss_list, valid_loss_list, global_steps_list)

                train_loss = 0.0
                valid_loss = 0.0
                model.train()
                
    
    
    #wandb.log({'AUC-ROC' : wandb.plot.roc_curve(truePred, predictions, labels=[0, 1])})
    #wandb.log({'Precision_recall' : wandb.plot.pr_curve(truePred, predictions, labels=[0, 1])})

    save_metrics(output_path + 'metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
    print('Training done!')
    return Acc

#------------------------------------Evaluation---------------------------------

def evaluate(model, test_loader, PAD_INDEX, UNK_INDEX):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)
                
                output = model(source, attention_mask=mask).cuda() 
                target = target.cuda()

                y_pred.extend(torch.argmax(output.cuda(), axis=-1).tolist())
                y_true.extend(target.cuda().tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    #Acc = accuracy_score(y_true, y_pred)
    #pdb.set_trace()
    
    wandb.log({'confusion_matrix': cm})
    #wandb.log({'AUC-ROC' : wandb.plot.roc_curve(target,output, labels=[0, 1])})
    #wandb.log({'Precision_recall' : wandb.plot.pr_curve(target,output, labels=[0, 1])})
    

    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
