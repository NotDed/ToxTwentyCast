import pandas as pd
import sys
import torch

from simpletransformers.classification import ClassificationModel, ClassificationArgs


# Optional model configuration
mol = sys.argv[1]

save = sys.argv[2]

# Create a ClassificationModel
model_args = ClassificationArgs()
model_args.wandb_project = 'toxTwentyCastBin'

model = ClassificationModel(
    "roberta", save, args=model_args,
)

predictions, raw_outputs = model.predict([mol])

print(predictions, raw_outputs)
