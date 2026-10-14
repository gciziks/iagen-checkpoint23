from __future__ import annotations

import json
import csv
import gc
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from common import LABELS, MAX_LENGTH, MODEL_ID, ROOT, load_rows, prompt, require_gpu, seed_everything


class ClassificationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True, return_tensors="pt")
        labels = torch.full_like(batch["input_ids"], -100)
        for i, feature in enumerate(features):
            n = len(feature["labels"])
            labels[i, :n] = torch.tensor(feature["labels"], dtype=torch.long)
        batch["labels"] = labels
        return batch


def make_dataset(rows, tokenizer):
    inputs = []
    for row in rows:
        prefix = tokenizer(prompt(row["text"]), add_special_tokens=True)["input_ids"]
        answer = tokenizer(" " + row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = (prefix + answer)[:MAX_LENGTH]
        labels = ([-100] * len(prefix) + answer)[:MAX_LENGTH]
        inputs.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return Dataset.from_list(inputs)


def predict(model, tokenizer, rows, device):
    model.eval()
    result = []
    for row in rows:
        encoded = tokenizer(prompt(row["text"]), return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH).to(device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new = output[0, encoded["input_ids"].shape[1]:]
        answer = tokenizer.decode(new, skip_special_tokens=True).strip().lower()
        result.append(next((label for label in LABELS if label in answer), "outros"))
    return result


def main():
    device = require_gpu()
    seed_everything()
    set_seed(23)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, validation_rows = load_rows("train"), load_rows("validation")
    train_ds, val_ds = make_dataset(train_rows, tokenizer), make_dataset(validation_rows, tokenizer)
    configs = [(1e-4, 2), (1e-4, 3), (2e-4, 2), (2e-4, 3)]
    results_dir = ROOT / "results"
    artifacts = ROOT / "artifacts"
    results_dir.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    experiments = []
    for lr, epochs in configs:
        name = f"lr{lr:g}_ep{epochs}"
        run_dir = artifacts / name
        print(f"\n=== {name}; GPU={torch.cuda.get_device_name(device)} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                                 lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
        args = TrainingArguments(
            output_dir=str(ROOT / "results" / "checkpoints" / name), num_train_epochs=epochs,
            learning_rate=lr, per_device_train_batch_size=2, per_device_eval_batch_size=2,
            gradient_accumulation_steps=2, fp16=True, eval_strategy="epoch", save_strategy="no",
            logging_strategy="epoch", report_to="none", seed=23, data_seed=23,
            optim="adamw_torch", dataloader_num_workers=0,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=ClassificationCollator(tokenizer))
        trainer.train()
        val_loss = float(trainer.evaluate()["eval_loss"])
        val_preds = predict(model, tokenizer, validation_rows, device)
        from sklearn.metrics import accuracy_score, f1_score
        metric = {"accuracy": accuracy_score([x["label"] for x in validation_rows], val_preds),
                  "macro_f1": f1_score([x["label"] for x in validation_rows], val_preds, labels=LABELS,
                                       average="macro", zero_division=0)}
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
        record = {"experiment": name, "learning_rate": lr, "epochs": epochs, "lora_r": 8,
                  "gpu": torch.cuda.get_device_name(device), "validation_loss": val_loss, **metric}
        experiments.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        del trainer, model, base
        gc.collect()
        torch.cuda.empty_cache()
    experiments.sort(key=lambda x: (x["macro_f1"], x["accuracy"], -x["validation_loss"]), reverse=True)
    best = experiments[0]
    (artifacts / "best_adapter.txt").write_text(best["experiment"], encoding="utf-8")
    fields = list(experiments[0])
    with (results_dir / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(experiments)
    (results_dir / "experiments.json").write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Melhor configuração na validação: {best['experiment']}. Execute scripts/evaluate.py.")


if __name__ == "__main__":
    main()
