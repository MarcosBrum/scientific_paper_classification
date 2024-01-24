import json
from argparse import ArgumentParser
from os.path import join

from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import MultilabelAccuracy

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from utils import CATEGORIES, CHECKPOINT, DATA_PATH, MODELS_PATH, TOKENIZED_DATA_DIR


FEAT_TOKENS = ["input_ids", "attention_mask"]
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = join(MODELS_PATH, "model.pt")
PREDICTIONS_SAVE_PATH = join(MODELS_PATH, "predictions.pt")
METRICS_EVALUATION_SAVE_PATH = join(MODELS_PATH, "metrics_evaluation.json")
CONFUSION_MATRIX_SAVE_PATH = join(MODELS_PATH, "multilabel_confusion_matrix.npy")
IDS_SAVE_PATH = join(MODELS_PATH, "ids.txt")

dataset = load_from_disk(join(DATA_PATH, TOKENIZED_DATA_DIR))

LABELS = sorted(set([label for label in dataset["train"].column_names if label in CATEGORIES]))
NUM_LABELS = len(LABELS)

_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
data_collator_func = DataCollatorWithPadding(tokenizer=_tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,
                                                           num_labels=NUM_LABELS,
                                                           problem_type="multi_label_classification",
                                                           state_dict=torch.load(MODEL_SAVE_PATH))

def main(predictions_save_path: str = PREDICTIONS_SAVE_PATH,
         metrics_evaluation_save_path: str = METRICS_EVALUATION_SAVE_PATH,
         confusion_matrix_save_path: str = CONFUSION_MATRIX_SAVE_PATH,
         ids_save_path: str = IDS_SAVE_PATH):

    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(lambda x: {"labels": [x[col] for col in LABELS]})
    
    ids = test_dataset["id"]
    test_dataset = test_dataset.select_columns(["labels", *FEAT_TOKENS])
    
    test_dataset.set_format("torch")
    test_data = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator_func
    )

    if not model:
        raise RuntimeError("Model not Initialized")
    
    model.to(DEVICE)

    print("Evaluate performance on test data:")
    
    predictions = list()
    metrics = MultilabelAccuracy(threshold = 0.5, criteria = "hamming")
    model.eval()
    for batch in tqdm(test_data):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch_labels = batch.pop("labels", [])
        batch_labels = batch_labels.float()
        with torch.no_grad():
            outputs = model(**batch, labels=batch_labels)

        logits = outputs.logits
        predictions_batch = (torch.sigmoid(logits) >= 0.5).int().cpu().detach()
        predictions.append(predictions_batch)
        metrics.update(input = predictions_batch, target = batch_labels.int())

    metrics_val = {"accuracy": metrics.compute().item()}

    predictions_tensor = torch.cat(tensors=predictions, dim=0)

    multilabel_conf_mt = multilabel_confusion_matrix(y_true=test_dataset["labels"], y_pred=predictions_tensor, labels=[1, 0])


    # store predictions
    predictions_save_path = predictions_save_path if predictions_save_path else PREDICTIONS_SAVE_PATH
    metrics_evaluation_save_path = metrics_evaluation_save_path if metrics_evaluation_save_path else METRICS_EVALUATION_SAVE_PATH
    confusion_matrix_save_path = confusion_matrix_save_path if confusion_matrix_save_path else CONFUSION_MATRIX_SAVE_PATH
    ids_save_path = ids_save_path if ids_save_path else IDS_SAVE_PATH

    with open(predictions_save_path, "wb") as predictions_file:
        torch.save(predictions_tensor, predictions_file)
    with open(metrics_evaluation_save_path, "w") as metrics_file:
        json.dump(metrics_val, metrics_file)
    with open(confusion_matrix_save_path, "wb") as conf_mt_file:
        np.save(file=conf_mt_file, arr=multilabel_conf_mt)
    with open(ids_save_path, "w") as f:
        f.write("\n".join(ids))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions_save_path", help="Local file path to save the model predictions", required=False)
    parser.add_argument("--metrics_evaluation_save_path", help="Local file path to save the evaluation metrics", required=False)
    parser.add_argument("--confusion_matrix_save_path", help="Local file path to save multilabel confusion matrix", required=False)
    parser.add_argument("--ids_save_path", help="Local file path to save shuffled ids", required=False)

    args = parser.parse_args()

    main(args.predictions_save_path, args.metrics_evaluation_save_path, args.confusion_matrix_save_path, args.ids_save_path)
