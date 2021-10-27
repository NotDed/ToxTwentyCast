'''
!pip install sklearn
!pip install selfies
!pip install simpletransformers == 0.48.6
!pip install 'transformers==3.3.1'
!pip install tensorboardX
!pip install wandb
'''
import pandas as pd
import sys
import torch

import selfies
import logging

from sklearn import metrics
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from datetime import datetime

torch.cuda.empty_cache()

#loggers basic configurations
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#loading and spliting toxTwentyCast
toxTwentyCastdf = pd.read_csv("dataset/toxTwentyCast.csv")
train_df, val_df = train_test_split(toxTwentyCastdf, test_size=0.3)

# Optional model configuration
model_type = sys.argv[1]

epoch = int(sys.argv[2])

if model_type == 's50k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_50k'

elif model_type == 's70k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_70k'

elif model_type == 's120k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'

elif model_type == 's150k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_150k'

elif model_type == 's160k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'

else:
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'


#data and time for today

trainName = input("nombre para este entrenamiento: ")
todayDir = "outputs/" + trainName + "/"
todayBestDir = "outputs/" + trainName + "/best_model"

# Create a ClassificationModel
model_args = ClassificationArgs()

model_args.save_model_every_epoch=True
# model_args.save_steps = -1 #794
model_args.tokenizer_name=seyonec
model_args.output_dir = todayDir
model_args.best_model_dir = todayBestDir

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
# model_args.max_seq_length = 128
model_args.num_train_epochs = epoch
model_args.train_batch_size = 16
model_args.eval_batch_size = 16

model_args.evaluate_during_training = True
# model_args.evaluate_during_training_steps = 618
# model_args.evaluate_during_training_verbose = True
model_args.use_cached_eval_features = True

model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "roc_auc_score"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 3

model_args.learning_rate = 1e-4
model_args.max_seq_length = 512


model_args.adafactor_eps = 1e-12
model_args.config = {
        "dropout": 0.1,
    }

model_args.wandb_project = 'toxTwentyCastBin'

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", seyonec, args=model_args,
)

# Train the model
model.train_model(train_df, eval_df=val_df, acc=metrics.accuracy_score, aps=metrics.average_precision_score, roc = metrics.roc_auc_score)

# accuracy
# result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=metrics.accuracy_score)

# ROC-PRC
# result, model_outputs, wrong_predictions = model.eval_model(val_df, aps=metrics.average_precision_score)
