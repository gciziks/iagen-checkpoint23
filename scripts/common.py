from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = ["acesso", "pagamento", "entrega", "cancelamento", "outros"]
SEED = 23
MAX_LENGTH = 192


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU não disponível. Instale PyTorch com CUDA e execute novamente.")
    return torch.device("cuda")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(partition: str) -> list[dict[str, str]]:
    path = ROOT / "data" / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row["label"] not in LABELS for row in rows):
        raise ValueError(f"Rótulo desconhecido em {path}")
    return rows


def prompt(text: str) -> str:
    return ("Classifique o chamado de suporte em exatamente uma categoria. "
            "Categorias: acesso, pagamento, entrega, cancelamento, outros. "
            "Responda somente com a categoria.\nChamado: " + text + "\nCategoria:")


def scores(gold: list[str], predicted: list[str]) -> dict[str, float]:
    return {"accuracy": accuracy_score(gold, predicted),
            "macro_f1": f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)}
