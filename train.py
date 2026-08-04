#!/usr/bin/env python3
"""Fine-tune OpenMed PII models on token-level BIO/NER data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(PROJECT_DIR / "models" / "OpenMed-PII-Vi-184M-v1")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--labels",
        type=Path,
        help="JSON array of label names. Required when training with new/different labels.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 on compatible CUDA GPUs.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 on CUDA GPUs.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--peft", action="store_true", help="Train a LoRA adapter instead of all weights.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="tensorboard")
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        parser.error("Choose only one of --bf16 and --fp16.")
    return args


def load_labels(labels_path: Path | None, model_name: str) -> list[str]:
    if labels_path:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("--labels must contain a JSON array of strings.")
        if len(labels) != len(set(labels)):
            raise ValueError("--labels contains duplicate label names.")
        return labels

    config = AutoConfig.from_pretrained(model_name)
    labels = []
    for index in range(config.num_labels):
        # JSON config maps may have string keys; Transformers may normalize them to ints.
        label = config.id2label.get(index, config.id2label.get(str(index)))
        if label is None:
            raise ValueError(f"Model config has no label for id {index}.")
        labels.append(label)
    return labels


def validate_examples(examples: dict[str, list[Any]], label_to_id: dict[str, int]) -> None:
    for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
        if len(tokens) != len(labels):
            raise ValueError(f"tokens and ner_tags lengths differ: {len(tokens)} vs {len(labels)}")
        for label in labels:
            if isinstance(label, str) and label not in label_to_id:
                raise ValueError(f"Unknown label: {label}")
            if isinstance(label, int) and not 0 <= label < len(label_to_id):
                raise ValueError(f"Label id {label} is outside 0..{len(label_to_id) - 1}")
            if not isinstance(label, (str, int)):
                raise ValueError("Each ner_tags entry must be a string label or an integer label id.")


def load_tokenizer(model_name: str):
    """Load OpenMed tokenizers despite legacy extra_special_tokens metadata.

    This model stores extra_special_tokens as a JSON list. Transformers 4.57
    expects a mapping for that field, while the actual PAD/CLS/SEP tokens are
    already declared separately in tokenizer_config.json.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except AttributeError as error:
        if "'list' object has no attribute 'keys'" not in str(error):
            raise
        LOGGER.warning(
            "Tokenizer has legacy list-valued extra_special_tokens metadata; "
            "retrying with a compatible empty mapping."
        )
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            extra_special_tokens={},
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    labels = load_labels(args.labels, args.model)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    raw_datasets: DatasetDict = load_dataset(
        "json", data_files={"train": str(args.train_file), "validation": str(args.validation_file)}
    )
    expected_columns = {"tokens", "ner_tags"}
    for split, dataset in raw_datasets.items():
        missing = expected_columns - set(dataset.column_names)
        if missing:
            raise ValueError(f"{split} is missing required fields: {sorted(missing)}")
        validate_examples(dataset, label_to_id)

    tokenizer = load_tokenizer(args.model)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required to align word-level NER labels.")

    def tokenize_and_align_labels(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=args.max_length,
            is_split_into_words=True,
        )
        aligned_labels = []
        for batch_index, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label = word_labels[word_id]
                    label_ids.append(label_to_id[label] if isinstance(label, str) else label)
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing and aligning BIO labels",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.peft:
        # These are the attention projection modules in Hugging Face DeBERTa-v2.
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query_proj", "key_proj", "value_proj"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, label_ids = eval_prediction
        predictions = np.argmax(logits, axis=2)
        true_predictions = [
            [labels[prediction] for prediction, label_id in zip(prediction_row, label_row) if label_id != -100]
            for prediction_row, label_row in zip(predictions, label_ids)
        ]
        true_labels = [
            [labels[label_id] for prediction, label_id in zip(prediction_row, label_row) if label_id != -100]
            for prediction_row, label_row in zip(predictions, label_ids)
        ]
        scores = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": scores["overall_precision"],
            "recall": scores["overall_recall"],
            "f1": scores["overall_f1"],
            "accuracy": scores["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved model and metrics to %s", args.output_dir)


if __name__ == "__main__":
    main()
