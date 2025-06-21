import os
import time
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
)
import evaluate
from src.utils.ner_data_utils import parse_conll, build_label_maps

# 1. Load and prepare data
conll_path = './data/raw/labeled_cnll_manual.txt'  # Adjust if needed
sentences, ner_tags = parse_conll(conll_path)
label2id, id2label = build_label_maps(ner_tags)
data = pd.DataFrame({'tokens': sentences, 'ner_tags': ner_tags})
dataset = Dataset.from_pandas(data)
if len(dataset) > 1:
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
else:
    train_dataset = dataset
    eval_dataset = dataset

# 2. Models to compare
model_names = [
    'xlm-roberta-base',
    'bert-base-multilingual-cased',
    'Davlan/bert-tiny-amharic-ner',
    'Davlan/afro-xlmr-mini',
    'distilbert-base-multilingual-cased'
]

results = []

# 3. Tokenization and label alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('I-') else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# 4. Training & evaluation loop
def train_and_evaluate(model_name, train_dataset, eval_dataset, label2id, id2label):
    global tokenizer
    print(f"\n==== Training {model_name} ====")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True)

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name.replace("/", "_")}',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='no',
        logging_dir=f'./logs/{model_name.replace("/", "_")}',
        fp16=True if torch.cuda.is_available() else False,
        report_to='none'
    )

    metric = evaluate.load('seqeval')
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return metric.compute(predictions=true_predictions, references=true_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    start_time = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.time() - start_time

    return {
        "model": model_name,
        "f1": metrics.get("eval_overall_f1", 0),
        "precision": metrics.get("eval_overall_precision", 0),
        "recall": metrics.get("eval_overall_recall", 0),
        "accuracy": metrics.get("eval_overall_accuracy", 0),
        "train_time_sec": elapsed
    }

for model_name in model_names:
    try:
        res = train_and_evaluate(model_name, train_dataset, eval_dataset, label2id, id2label)
        results.append(res)
    except Exception as e:
        print(f"Error with {model_name}: {e}")

# 5. Save and show results
df = pd.DataFrame(results)
df.to_csv('./results/model_comparison.csv', index=False)
print("\nModel comparison results:")
print(df)
