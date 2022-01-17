import transformers
from transformers import  AutoTokenizer
MAX_LEN = 258
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = "seyonec/BPE_SELFIES_PubChem_shard00_160k"
MODEL_PATH = "seyonec/BPE_SELFIES_PubChem_shard00_160k"
TRAINING_FILE = "../input/toxTwentyCast.csv"
TOKENIZER = AutoTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True,
    add_prefix_space=True
)