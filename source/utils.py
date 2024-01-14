from os.path import abspath, dirname, join


ROOT_PATH = dirname(abspath(__file__)).rsplit("/", maxsplit=1)[0]
ARCHIVE_PATH = join(ROOT_PATH, "archive")
DATA_PATH = join(ROOT_PATH, "data")
MODELS_PATH = join(ROOT_PATH, "models")

# CATEGORIES = ["cs.AI", "cs.CL", "cs.CV", "cs.IT", "cs.LG"]
CATEGORIES = ["cs.AI", "cs.LG"]

TRAIN_VAL_TEST_DATA_DIR = "train_val_test_data"
TOKENIZED_DATA_DIR = "tokenized_data"

CHECKPOINT = "distilbert-base-uncased"
