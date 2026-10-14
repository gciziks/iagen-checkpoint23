from __future__ import annotations

import csv
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from common import LABELS, MAX_LENGTH, MODEL_ID, ROOT, load_rows, prompt, require_gpu
from train import predict


def main():
    device = require_gpu()
    best_file = ROOT / "artifacts" / "best_adapter.txt"
    if not best_file.exists():
        raise FileNotFoundError("Execute python scripts/train.py antes da avaliação.")
    name = best_file.read_text(encoding="utf-8").strip()
    adapter = ROOT / "artifacts" / name
    rows = load_rows("test")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, adapter).to(device)
    predicted = predict(model, tokenizer, rows, device)
    gold = [row["label"] for row in rows]
    report = classification_report(gold, predicted, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold, predicted, labels=LABELS).tolist()
    result = {"experiment": name, "gpu": torch.cuda.get_device_name(device),
              "classification_report": report, "labels": LABELS, "confusion_matrix": matrix,
              "predictions": [{"text": row["text"], "gold": row["label"], "predicted": pred}
                              for row, pred in zip(rows, predicted)]}
    (ROOT / "results" / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "results" / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "gold", "predicted"])
        writer.writeheader(); writer.writerows(result["predictions"])
    print(json.dumps({"experiment": name, "gpu": result["gpu"], "accuracy": report["accuracy"],
                      "macro_f1": report["macro avg"]["f1-score"], "confusion_matrix": matrix},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
