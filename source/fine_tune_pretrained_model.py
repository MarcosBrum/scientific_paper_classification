import json
from argparse import ArgumentParser
from os.path import join

from tqdm import tqdm

from datasets import load_from_disk
from transformers import get_scheduler, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torcheval.metrics import MultilabelAccuracy

from utils import CATEGORIES, CHECKPOINT, DATA_PATH, MODELS_PATH, TOKENIZED_DATA_DIR


FEAT_TOKENS = ["input_ids", "attention_mask"]
BATCH_SIZE = 512
NUM_EPOCHS = 50
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL_SAVE_PATH = join(MODELS_PATH, "model.pt")
METRICS_SAVE_PATH = join(MODELS_PATH, "metrics.json")

dataset = load_from_disk(join(DATA_PATH, TOKENIZED_DATA_DIR))

LABELS = sorted(set([label for label in dataset["train"].column_names if label in CATEGORIES]))
NUM_LABELS = len(LABELS)
dataset = dataset.map(lambda x: {"labels": [x[col] for col in LABELS]})
dataset = dataset.select_columns(["labels", *FEAT_TOKENS])

_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
data_collator_func = DataCollatorWithPadding(tokenizer=_tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,
                                                           num_labels=NUM_LABELS,
                                                           problem_type="multi_label_classification")

def main(model_save_path: str = MODEL_SAVE_PATH, metrics_save_path: str = METRICS_SAVE_PATH):

    dataset.set_format("torch")

    train_data = DataLoader(
        dataset=dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator_func
    )
    eval_data = DataLoader(
        dataset=dataset["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator_func
    )

    if not model:
        raise RuntimeError("Model not Initialized")
    
    model.to(DEVICE)
    optimizer = AdamW(params=model.parameters(), lr=5e-5)

    num_training_steps = NUM_EPOCHS * len(train_data)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print(f"Fine tuning loop:")
    model.train()
    metrics = list()
    for epoch in range(NUM_EPOCHS):
        print(f"{epoch = }")
        for batch in tqdm(train_data):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch_labels = batch.pop("labels", [])
            batch_labels = batch_labels.float()
            outputs = model(**batch, labels=batch_labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print("Evaluation:")
        metrics_epoch = MultilabelAccuracy(threshold = 0.5, criteria = "hamming")
        model.eval()
        for batch in tqdm(eval_data):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch_labels = batch.pop("labels", [])
            batch_labels = batch_labels.float()
            with torch.no_grad():
                outputs = model(**batch, labels=batch_labels)

            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().detach()
            metrics_epoch.update(input = predictions, target = batch_labels.int())

        metrics_epoch_val = metrics_epoch.compute()
        metrics.append({"epoch": epoch, "accuracy": metrics_epoch_val.item()})


    # store models and metrics
    model_save_path = model_save_path if model_save_path else MODEL_SAVE_PATH
    metrics_save_path = metrics_save_path if metrics_save_path else METRICS_SAVE_PATH
    with open(model_save_path, "wb") as models_file:
        torch.save(model.state_dict(), models_file)
    with open(metrics_save_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_save_path", help="Local file path to save the tuned model", required=False)
    parser.add_argument("--metrics_save_path", help="Local file path to save the metrics", required=False)

    args = parser.parse_args()

    main(args.model_save_path, args.metrics_save_path)
