from __future__ import annotations

import sys
from datetime import datetime, timezone

from common import ROOT, require_gpu
import torch


def main() -> None:
    device = require_gpu()
    report = (f"Data UTC: {datetime.now(timezone.utc).isoformat()}\n"
              f"Python: {sys.version.split()[0]}\nPyTorch: {torch.__version__}\n"
              f"CUDA runtime (PyTorch): {torch.version.cuda}\n"
              f"GPU: {torch.cuda.get_device_name(device)}\n"
              f"VRAM total (GiB): {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f}\n"
              f"CUDA available: {torch.cuda.is_available()}\n")
    print(report)
    output = ROOT / "results" / "gpu_info.txt"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
