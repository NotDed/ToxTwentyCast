import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)

def train_fn(data_loader, model, optimizer, device):
    model.train()

    for bi,d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        # token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        # token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_gra()
        outputs = model(
            ids=ids,
            mask=mask,
            # token_type_ids = token_type_ids  
        )

        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outouts =[]
    with torch.no_grad():
        for bi,d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            # token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            # token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)


            outputs = model(
                ids=ids,
                mask=mask,
                # token_type_ids = token_type_ids  
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return fin_outputs, fin_targets
        