# Fine-tune OpenMed Vietnamese PII

Train the [OpenMed Vietnamese PII DeBERTa-v2 model](https://huggingface.co/OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1) for token-level NER using `uv`. By default, weights and tokenizer files are loaded locally from `models/OpenMed-PII-Vi-184M-v1`; use `--model` only to override that location.

## Dataset format

Supply one JSON Lines file per split. Every record must contain word-tokenized `tokens` and an equal-length `ner_tags` list. Tags may be BIO strings, as in the included example:

```json
{"tokens":["Nguyễn","Văn","An"],"ner_tags":["B-FIRSTNAME","B-MIDDLENAME","B-LASTNAME"]}
```

Use a label as `O` for non-entity tokens. Keep a validation set separate from training. Never commit raw clinical notes or identifiers.

`labels.json` is an ordered JSON array. Its tags must include every string used in the data. If you retain the source model's original 76 labels, omit `--labels`; the script then preserves the source model's label mapping. Pass `--labels` for a smaller or changed label set, which creates a newly initialized classifier head.

## Setup

```bash
uv sync
```

On an NVIDIA CUDA machine, add exactly one of `--bf16` (recommended for Ampere-or-newer GPUs) or `--fp16`. Do not add these flags on CPU or Apple Silicon.

## Full fine-tune

```bash
uv run train.py \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --labels labels.json \
  --output-dir outputs/full
```

## LoRA / PEFT

```bash
uv run train.py \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --labels labels.json \
  --output-dir outputs/lora \
  --peft --lora-r 16 --lora-alpha 32
```

With `--peft`, the output is a PEFT adapter plus the trainable token-classification head, not a standalone merged base model. Load it by first loading the same base model and then `PeftModel.from_pretrained(base_model, "outputs/lora")`.

For one GPU, increase `--train-batch-size` until memory is nearly full, then use `--gradient-accumulation-steps` to reach the desired effective batch size. Use `accelerate launch train.py ...` for multi-GPU runs.

## Validation

Training selects the checkpoint with the highest entity-level `seqeval` F1 and saves the final validation metrics to `metrics.json` in the output directory.
