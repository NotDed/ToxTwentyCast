import transformers
from transformers import  Tokenizer
MAX_LEN = 258
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = "../input/roberta_model"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/toxTwentyCast.csv"
TOKENIZER = AutoTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)