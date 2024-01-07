import re
from os.path import join

from datasets import load_from_disk
from transformers import AutoTokenizer

from utils import CHECKPOINT, DATA_PATH, TOKENIZED_DATA_DIR, TRAIN_VAL_TEST_DATA_DIR


# data
dataset = load_from_disk(dataset_path=join(DATA_PATH, TRAIN_VAL_TEST_DATA_DIR))

dataset = dataset.map(lambda x: {"abstract": [re.sub(r'\n+', ' ', a) for a in x["abstract"]]},
                      batched=True, num_proc=4)
dataset = dataset.map(lambda x: {"text": ["\n".join([t, a]) for t, a in zip(x["title"], x["abstract"])]},
                      batched=True, num_proc=4)

# tokenize
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
def custom_tokenizer(rows, _tokenizer):
    result = _tokenizer(
        rows["text"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in rows.items():
        result[key] = [values[i] for i in sample_map]
    return result

dataset = dataset.map(custom_tokenizer, batched=True, num_proc=4, fn_kwargs={"_tokenizer": tokenizer})

# save
dataset.save_to_disk(join(DATA_PATH, TOKENIZED_DATA_DIR))
