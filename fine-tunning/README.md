# QLoRA Fine‑Tuning: Gemma‑3‑1B → Text‑to‑SQL

This repo contains two ways to fine‑tune **Gemma‑3‑1B** on the **gretel-synthetic-text-to-sql** dataset using **QLoRA**:
1) a Google Colab **notebook** (`qlora.ipynb`) and  
2) a **Python script** (`fine-tunning/prompt-fine-tunning/main.py`).

> QLoRA trains small LoRA adapters on top of a 4‑bit quantized base model, so you get strong results with modest GPU memory.

---

## 📁 Folder Structure (example)
```
WEEK-10/
├─ fine-tunning/
│  ├─ prompt-fine-tunning/
│  │  └─ main.py
│  └─ qlora-fine-tuning/          # (optional assets/outputs)
├─ qlora.ipynb                     # Colab/Notebook version
├─ .env                            # holds HF_TOKEN
├─ .gitignore
└─ README.md
```

---

## 🧰 Requirements

- Python **3.10+**
- CUDA GPU recommended (bf16/flash attention where available)
- PyTorch **2.4+**

Install the exact libs used in the notebook/script:

```bash
pip install "torch>=2.4.0" tensorboard
pip install "transformers>=4.51.3"
pip install --upgrade   "datasets==3.3.2"   "accelerate==1.4.0"   "evaluate==0.4.3"   "bitsandbytes==0.45.3"   "trl==0.21.0"   "peft==0.14.0"   sentencepiece python-dotenv
```

> On Windows without CUDA, `bitsandbytes` may fall back to CPU or fail; prefer Linux with NVIDIA GPUs.

---

## 🔐 Environment

Create a `.env` file in the project root:
```
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

You can also login interactively:
```bash
python -c "from huggingface_hub import login; login()"
```

---

## 📦 Data & Model

- **Dataset**: `philschmid/gretel-synthetic-text-to-sql`
- **Base model**: `google/gemma-3-1b-pt` (Causal LM)
- **Tokenizer**: `google/gemma-3-1b-it` (instruction tokenizer & chat template)

---

## ▶️ Run: Notebook

Open **`qlora.ipynb`** and run cells top‑to‑bottom. It will:
- load dataset (optionally subset to ~12.5k),
- build chat messages,
- load Gemma‑3‑1B in 4‑bit,
- attach LoRA (r=16, alpha=16, dropout=0.05),
- train with TRL `SFTTrainer` (`packing=True`, `max_length=512`),
- push adapters to the Hub (optional).

TensorBoard logging is enabled:
```bash
tensorboard --logdir runs
```

---

## ▶️ Run: Python Script

From the repo root:
```bash
python -m fine-tunning.prompt-fine-tunning.main
```
This mirrors the notebook config. If your `main.py` exposes CLI flags, you can pass them, e.g.:
```bash
python -m fine-tunning.prompt-fine-tunning.main   --model_id google/gemma-3-1b-pt   --epochs 3   --per_device_train_batch_size 1   --grad_accum 4   --max_length 512   --learning_rate 2e-4   --rank 16 --lora_alpha 16 --lora_dropout 0.05   --output_dir gemma-text-to-sql
```

> The training progress bar like `[2653/3285 … Epoch 2.42/3]` shows **current_step/total_steps**; total steps are computed from (packed batches × epochs ÷ grad accumulation).

---

## 🔧 Key Training Settings (defaults in this repo)

- Quantization: 4‑bit (bnb **nf4**, double quant, compute dtype bf16/fp16)
- Optimizer: `adamw_torch_fused`
- LR Schedule: warmup ratio **0.03**, then **constant**
- Gradient checkpointing: **on** (disables `use_cache` during training)
- Max grad norm: **0.3**
- Packing: **True** (efficient 512‑token batches)

---

## 📤 Outputs

- Adapters and tokenizer are written to `gemma-text-to-sql/` (default).
- If pushing to the Hub is enabled, they will be uploaded under your HF account.
- Use TensorBoard logs in `./runs` (or `args.logging_dir`).

---

## ✅ Baseline vs Fine‑Tuned (example)

| Model | Setup | Result |
|---|---|---|
| Gemma‑3‑1B (no FT) | zero‑shot on text‑to‑SQL | weak SQL generation |
| Gemma‑3‑1B + QLoRA (r=16, lr=2e‑4, dr=0.05) | 3 epochs, packed 512 | expected correct SQL on held‑out prompts |

*(Replace with your actual metrics, e.g., exact‑match, execution accuracy, BLEU, etc.)*

---

## 🔎 Quick Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "google/gemma-3-1b-pt"
adpt = "./gemma-text-to-sql"  # or 'your-hf-username/gemma-text-to-sql'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    base,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    quantization_config=dict(load_in_4bit=True)
)
model = PeftModel.from_pretrained(model, adpt)

prompt = (
    "Given the <USER_QUERY> and the <SCHEMA>, generate the SQL.\n"
    "<SCHEMA>\nemployees(id, name, dept, salary)\n</SCHEMA>\n"
    "<USER_QUERY>\nList names of employees in 'Sales' with salary > 50k\n</USER_QUERY>"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## 🛠 Troubleshooting

- **`use_cache=True is incompatible with gradient checkpointing`** → expected; training turns it off.
- **OOM** → reduce `max_length`, set `per_device_train_batch_size=1`, increase gradient accumulation, or lower LoRA rank.
- **bitsandbytes errors** → ensure CUDA toolkit & drivers match your PyTorch build.

---

## 📄 License & Credits

- Dataset: `philschmid/gretel-synthetic-text-to-sql`
- Model: `google/gemma-3-1b-pt` / `google/gemma-3-1b-it`
- Libraries: Hugging Face `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`

Happy fine‑tuning! 🚀
