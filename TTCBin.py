'''
!pip install sklearn
!pip install selfies
!pip install simpletransformers == 0.48.6
!pip install 'transformers==3.3.1'
!pip install tensorboardX
!pip install wandb
'''
import pandas as pd

import selfies
import logging

from sklearn import metrics
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from datetime import datetime

#loggers basic configurations
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#loading and spliting toxTwentyCast
toxTwentyCastdf = pd.read_csv("dataset/toxTwentyCast.csv")
train_df, val_df = train_test_split(toxTwentyCastdf, test_size=0.3)

# Optional model configuration
model_type = sys.argv[1]

if model_type == 's120k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'

if model_type == 's150k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_150k'

if model_type == 's160k':
    seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'


#data and time for today
now = datetime.now()
todayDir = now.strftime("outputs/%d-%m-%Y-%H%M/")
todayBestDir = now.strftime("outputs/%d-%m-%Y-%H%M/best_model")

# Create a ClassificationModel
model_args = ClassificationArgs()

model_args.save_model_every_epoch=False
model_args.tokenizer_name=seyonec
model_args.output_dir = todayDir
model_args.best_model_dir = todayBestDir

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.max_seq_length = 512
model_args.num_train_epochs = 15
model_args.train_batch_size = 32

model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.use_cached_eval_features = True

model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 3

model_args.wandb_project = 'toxTwentyCast'

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", seyonec, args=model_args,
)

# Train the model
model.train_model(train_df, eval_df=val_df, acc=metrics.accuracy_score, aps=metrics.average_precision_score)
