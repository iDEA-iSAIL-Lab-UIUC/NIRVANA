# README

![alt text](cover1-compressed.jpg)

## 1 . Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python    | **3.9.19** | Tested with CPython 3.9.19 |
| PyTorch   | **2.2.2 + cu121** | Pre‑built CUDA 12.1 wheel |

> **Quick start:**  
> ```bash
> bash setup.sh
> ```  
> The script creates the Conda environment, installs PyTorch (CUDA 12.1), and places all project dependencies in editable mode.

---

## 2 . Running the Code

### 2.1  Pruning
```bash
cd Llama/
bash run.sh
````

* **Scope:** NIRVANA pruning is implemented for the Llama family.
* Support for Qwen and T5 is in progress; the main experiment in paper focuses on Llama, so current functionality should suffice.

### 2.2  Evaluation

```bash
cd lm_eval/
# Edit `run_eval.sh` to point to your pruned checkpoint
bash run_eval.sh
```

### 2.3  Post‑Training with LoRA

```bash
bash recover_ft.sh
```

Follow the interactive prompts to apply LoRA fine‑tuning on top of the pruned model.

---

## 3 . Hardware Notes

* **Zero‑shot evaluation/pruning of Llama‑3.1‑8B** requires roughly **80 GB** of GPU VRAM.
* Smaller models or mixed‑precision settings (e.g., `bfloat16`) reduce the footprint proportionally.

---

## 4 . Directory Overview

```
.
├── Llama/                 
│   ├── run.sh             # NIRVANA pruning scripts
│   └── recover_ft.sh      # LoRA post‑training
├── eval/              # Evaluation
│   └── run_eval.sh
└── setup.sh               # One‑click environment bootstrap
        
```

