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

#loggers basic configurations
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#loading and spliting toxTwentyCast
toxTwentyCastdf = pd.read_csv("dataset/toxTwentyCast.csv")
train_df, eval_df = train_test_split(toxTwentyCastdf, test_size=0.3)

# Optional model configuration
EPOCHS = 100
seyonec = 'seyonec/BPE_SELFIES_PubChem_shard00_120k'

# Create a ClassificationModel
model_args = ClassificationArgs(
    num_train_epochs = EPOCHS,
    wandb_project = 'toxTwentyCast',
    tokenizer_name=seyonec)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", seyonec, args=model_args,
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# accuracy
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

# ROC-PRC
result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)
