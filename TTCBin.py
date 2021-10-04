'''
!pip install sklearn
!pip install selfies
!pip install simpletransformers
!pip install tensorboardX
!pip install wandb
'''
import pandas as pd

import selfies
import logging

from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from datetime import datetime

#loggers basic configurations
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#loading and spliting toxTwentyCast
toxTwentyCastdf = pd.read_csv("dataset/toxTwentyCast.csv")
train_df, eval_df = train_test_split(toxTwentyCastdf, test_size=0.3)

# Optional model configuration
seyonec120K = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'
seyonec150K = 'seyonec/BPE_SELFIES_PubChem_shard00_150k'
seyonec160K = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'


#data and time for today
now = datetime.now()
todayDir = now.strftime("outputs/%d-%m-%Y-%H%M/")
todayBestDir = now.strftime("outputs/%d-%m-%Y-%H%M/best_model")

# Create a ClassificationModel
model_args = ClassificationArgs()
model_args.num_train_epochs = 15
model_args.wandb_project = 'toxTwentyCast'
model_args.tokenizer_name=seyonec120K
model_args.save_model_every_epoch=False
model_args.evaluate_during_training = True
model_args.output_dir = todayDir
model_args.best_model_dir = todayBestDir
model_args.eval_batch_size = 128
model_args.train_batch_size = 128
model_args.use_early_stopping = True

# Create a ClassificationModel9
model = ClassificationModel(
    "roberta", seyonec120K, args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# accuracy
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

# ROC-PRC
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)
