from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.common import LABELS, MAX_LENGTH, MODEL_ID, ROOT, prompt, require_gpu

st.set_page_config(page_title="Triagem de chamados", page_icon="🎫")
st.title("Triagem local de chamados")
st.caption("Qwen2.5-0.5B-Instruct + adaptador LoRA · inferência local na GPU")

if not torch.cuda.is_available():
    st.error("CUDA não disponível. Ative um ambiente Python com PyTorch CUDA para usar esta demonstração.")
    st.stop()

best_file = ROOT / "artifacts" / "best_adapter.txt"
if not best_file.exists():
    st.info("Nenhum adaptador treinado encontrado. Execute `python scripts/train.py` primeiro.")
    st.stop()

adapter_names = sorted(p.name for p in (ROOT / "artifacts").iterdir() if p.is_dir())
selected = st.selectbox("Adaptador", adapter_names,
                        index=adapter_names.index(best_file.read_text(encoding="utf-8").strip())
                        if best_file.read_text(encoding="utf-8").strip() in adapter_names else 0)

@st.cache_resource(show_spinner="Carregando modelo na GPU…")
def load_model(name: str):
    device = require_gpu()
    path = ROOT / "artifacts" / name
    tokenizer = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    return tokenizer, PeftModel.from_pretrained(base, path).to(device)

text = st.text_area("Descreva o chamado", placeholder="Ex.: Meu pedido está atrasado e o rastreio não atualiza…", height=130)
if st.button("Classificar", type="primary", disabled=not text.strip()):
    tokenizer, model = load_model(selected)
    device = require_gpu()
    encoded = tokenizer(prompt(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    label = next((x for x in LABELS if x in answer), "outros")
    st.success(f"Categoria prevista: **{label}**")
    st.caption(f"Execução em {torch.cuda.get_device_name(device)} · saída bruta: {answer or '(vazia)'}")
