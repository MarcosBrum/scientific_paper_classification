import json
from os.path import join

from datasets import Dataset

from utils import ARCHIVE_PATH, CATEGORIES, DATA_PATH, TRAIN_VAL_TEST_DATA_DIR


# constants
DATA_FIELDS = ["id", "title", "categories", "abstract"]

RANDOM_STATE = 42

# load data
with open(join(ARCHIVE_PATH, "arxiv-metadata-oai-snapshot.json"), "r") as f:
    metadata = f.readlines()

metadata = [json.loads(doc) for doc in metadata]

# select recent papers (after incl. 2022) in Computer Science and fields
data = [
    doc for doc in metadata if (
        ("." in doc["id"] and doc["id"].split(".")[0] >= "2200") and
        any([cat in doc["categories"] for cat in CATEGORIES])
    )
]
data = [{field: doc[field] for field in DATA_FIELDS} for doc in data]

# prepare label cols for multicol classification
all_categories = set([doc['categories'] for doc in data])
all_categories = sorted(set([c for cat in all_categories for c in cat.split(' ') if c in CATEGORIES]))

for doc in data:
    for cat in all_categories:
        doc[cat] = 1 if cat in doc["categories"] else 0

dataset = Dataset.from_list(data)

# drop few populated categories
cols_drop = []
for col in all_categories:
    if sum(dataset[col]) < 10:
        cols_drop.append(col)
dataset = dataset.remove_columns(column_names=cols_drop)
categories_remain = [cat for cat in all_categories if cat not in cols_drop]

# drop rows w/o category left
dataset = dataset.filter(lambda r, cols: any([r.get(c, 0) != 0 for c in cols]),
                         fn_kwargs={"cols": categories_remain})

# train - validation - test split
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=RANDOM_STATE)
dataset_train_val_test = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=RANDOM_STATE)
dataset_train_val_test["validation"] = dataset_train_val_test.pop("test")
dataset_train_val_test["test"] = dataset["test"]

# save
dataset_train_val_test.save_to_disk(join(DATA_PATH, TRAIN_VAL_TEST_DATA_DIR))
