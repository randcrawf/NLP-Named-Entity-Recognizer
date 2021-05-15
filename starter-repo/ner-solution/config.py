import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "bert-base-cased"
MODEL_PATH = "model.bin"
INPUT_FILE = "starter-repo/data/test/test.nolabels.txt"
OUTPUT_FILE = "starter-repo/data/test/test.out"
TRAINING_FILE = "starter-repo/ner-solution/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)