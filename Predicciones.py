import pandas as pd
import sys
import torch

from simpletransformers.classification import ClassificationModel


# Optional model configuration
mol = sys.argv[1]

save = sys.argv[2]

model = ClassificationModel(
    "roberta", save, args=model_args,
)

predictions, raw_outputs = model.predict([mol])

print(predictions, raw_outputs)
