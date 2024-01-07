from os.path import abspath, dirname, join


ARCHIVE_PATH = join(dirname(abspath(__file__)).rsplit("/", maxsplit=1)[0], "archive")
DATA_PATH = join(dirname(abspath(__file__)).rsplit("/", maxsplit=1)[0], "data")
TRAIN_VAL_TEST_DATA_DIR = "train_val_test_data"
TOKENIZED_DATA_DIR = "tokenized_data"

CHECKPOINT = "distilbert-base-uncased"
