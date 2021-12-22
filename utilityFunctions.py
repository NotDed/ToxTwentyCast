import torch
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM

def save_checkpoint(path, model, valid_loss):
    torch.save({'model_state_dict': model.module.state_dict(),
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
