# SEMDT: Semantic Decision Transformer for Trajectory Generation

Official implementation of **SEMDT** — a semantic grounding framework for trajectory generation from logged driving data, built on the Decision Transformer (DT).

## Overview

SEMDT augments a causal Decision Transformer with motion-aware natural-language captions generated offline by a frozen vision-language pipeline (BLIP-2 → GPT-2 refinement). At each timestep, SEMDT aligns the DT's per-timestep **decision latent** with a CLIP text embedding via a contrastive (InfoNCE) loss. Language acts as a controllable proxy for maneuver intent, disambiguating multi-modal futures that kinematics alone cannot resolve.

**Key contributions from the paper:**

1. **Semantic grounding reduces tail-risk.** Motion-aware captions consistently reduce MR@5m (the fraction of rollouts deviating >5m at the final step) across all tested backbones.
2. **Caption granularity is safety-relevant.** Coarse object-level captions can reduce ADE while *increasing* MR@5m. Only INTENT-level captions (encoding motion and interaction cues) reduce both mean error and tail failures simultaneously.
3. **Domain inductive bias matters more than scale.** Under identical state/action interfaces and identical semantic supervision, frozen general-purpose LLM backbones (GPT-2, LLaMA, Qwen, DeepSeek) consistently underperform the DT backbone — confirming that pretraining scale alone does not replace physically aligned inductive bias.
4. **λ = 0.1 is the sweet spot.** A modest alignment weight gives the best trade-off; over-weighting the semantic term (λ ≥ 0.5) degrades trajectory fidelity.

---

## Architecture

```
Offline Data
    └─ Sensory observations (camera frames)
          └─ BLIP-2 base caption
                └─ GPT-2 motion-aware refinement (state trend + intent prompt)
                      └─ CLIP text encoder → text embedding z_t

Kinematic history [RTG_t, s_t, a_t] × K timesteps
    └─ Causal Transformer (DT backbone)
          └─ per-timestep decision latent h_t
                ├─ adapter g(z_t) → semantic fusion → action head → â_t
                └─ InfoNCE contrastive loss aligning h̃_t ↔ z̃_t
```

Training objective:
```
L_total = (1 - λ) * L_traj  +  λ * L_text
```
where `L_traj` is ℓ₂ action regression and `L_text` is InfoNCE contrastive alignment.

---

## Repository Structure

```
.
├── decision_transformer/                # Core DT architecture
│   ├── models/
│   │   ├── decision_transformer.py      # Main DT model (RTG/state/action interleaving)
│   │   ├── trajectory_gpt2.py           # GPT-2 backbone (positional embeddings removed)
│   │   └── model.py                     # TrajectoryModel base class
│   └── training/
│       ├── seq_trainer.py               # Sequence trainer
│       ├── act_trainer.py               # Action trainer
│       └── trainer.py                   # Base trainer
│
├── evaluation/
│   └── evaluate_episodes.py
│
│── text_contrastive_head.py             # InfoNCE head with false-negative masking + queue
│── trajectory_experiment_visiontrap.py  # *** CORE: SEMDT training pipeline (all shared logic)
│
│── # ─── PAPER: Backbone Swap Experiments ────────────────────────────────────
│── gpt2_trajectory_model.py             # GPT-2 swap (from-scratch config)
│── llama_trajectory_model.py            # LLaMA swap
│── qwen_trajectory_model.py             # Qwen swap
│── deepseek_trajectory_model.py         # DeepSeek swap
│── trajectory_experiment_gpt2_vlm.py    # SemDT run with GPT-2 backbone
│── trajectory_experiment_llama_vlm.py   # SemDT run with LLaMA backbone
│── trajectory_experiment_qwen_vlm.py    # SemDT run with Qwen backbone
│── trajectory_experiment_deepseek_vlm.py# SemDT run with DeepSeek backbone
│
│── # ─── PAPER: Lambda Sensitivity Scripts ───────────────────────────────────
│── run_dt_vlm_lambda_05_test.sh         # λ=0.5 sensitivity — DT backbone
│── run_dt_vlm_lambda_08_test.sh         # λ=0.8 sensitivity — DT backbone
│── run_gpt2_vlm_lambda_05_test.sh       # λ=0.5 — GPT-2
│── run_gpt2_vlm_lambda_08_test.sh       # λ=0.8 — GPT-2
│── run_llama_vlm_lambda_05_test.sh      # λ=0.5 — LLaMA
│── run_llama_vlm_lambda_08_test.sh      # λ=0.8 — LLaMA
│── run_qwen_vlm_lambda_05_test.sh       # λ=0.5 — Qwen
│── run_qwen_vlm_lambda_08_test.sh       # λ=0.8 — Qwen
│── run_deepseek_vlm_lambda_05_test.sh   # λ=0.5 — DeepSeek
│── run_deepseek_vlm_lambda_08_test.sh   # λ=0.8 — DeepSeek
│
│── # ─── EXTENSION: LoRA Runs ────────────────────────────────────────────────
│── qwen_lora_trajectory_model.py        # Qwen2-1.5B-Instruct + LoRA (r=16, α=32)
│── trajectory_experiment_qwen_lora_vlm.py # LoRA-only Qwen run via VisionTRAP pipeline
│── run_qwen_lora_vlm_lambda_01_test.sh  # Launch script for Qwen LoRA-only
│── gpt2_lora_trajectory_model.py        # GPT-2 pre-trained + LoRA (c_attn, c_proj)
│── trajectory_experiment_gpt2_lora_vlm.py # LoRA-only GPT-2 run via VisionTRAP pipeline
│
│── # ─── EXTENSION: Instruction Tuning Runs ──────────────────────────────────
│── qwen_instruction_lora_model.py       # Qwen: instruction prefix + DT interleaving + LoRA
│── trajectory_experiment_qwen_instruction_lora.py # Qwen instruction (+/- LoRA) experiment
│── gpt2_instruction_lora_model.py       # GPT-2: instruction prefix + DT interleaving + LoRA
│── trajectory_experiment_gpt2_instruction_lora.py # GPT-2 instruction (+/- LoRA) experiment
│
│── # ─── Caption / Manifest Generation ──────────────────────────────────────
│── generate_prompted_metadata_captions.py  # INTENT-level caption generation pipeline
│── generate_gpt2_vlm_hybrid_manifest.py    # Hybrid BLIP-2 + GPT-2 manifest builder
│── generate_pure_vlm_manifest.py           # Base VLM manifest builder
│── visiontrap_batch_function.py            # Batch inference helpers
│── visiontrap_dataloader.py                # Data loading utilities
│── finetune_gpt2_improved.py               # GPT-2 caption refinement fine-tuning
│
├── requirements.txt                     # All dependencies for all experiment families
├── RUNBOOK.md                           # *** Exact commands to reproduce every result
├── DATA_SETUP.md                        # Dataset + manifest setup instructions
└── Archive/                             # Historical runs and reference implementations
    └── reference_implementation/        # Original Decision Transformer reference code
```

---

## Results Summary

### Table I — Caption Granularity (DT backbone, λ=0.1)

| Caption Type | ADE ↓ | FDE ↓ | MR@5m ↓ |
|---|---:|---:|---:|
| DT (kinematics-only) | 12.08 | 20.15 | 15.9% |
| SEMDT (OBJECT) | 2.42 | 5.17 | 25.5% |
| SEMDT (SCENE) | 1.88 | 3.87 | 17.0% |
| **SEMDT (INTENT)** | **1.67** | **3.27** | **12.8%** |

### Table II — Backbone Comparison (INTENT captions, λ=0.1)

| Backbone | ADE ↓ | FDE ↓ | MR@5m ↓ |
|---|---:|---:|---:|
| **DT (Causal Transformer)** | **1.67** | **3.27** | **12.8%** |
| GPT-2 | 2.05 | 4.44 | 25.5% |
| LLaMA | 3.42 | 7.25 | 68.2% |
| Qwen | 3.36 | 6.39 | 65.0% |
| DeepSeek | 3.52 | 8.20 | 69.3% |

### Extension — LoRA & Instruction Tuning

| Run | ADE ↓ | FDE ↓ | MR@5m ↓ | Method |
|---|---:|---:|---:|---|
| SemDT (DT, best) | 1.67 | 3.27 | 12.8% | DT from scratch |
| GPT-2 Instr+LoRA | 1.63 | 3.79 | 11.1% | GPT-2 pre-trained + LoRA + instruction prefix |
| GPT-2 Instr (no LoRA) | 1.04* | 2.46* | 11.1% | GPT-2 pre-trained + instruction prefix |
| Qwen Instr+LoRA | 1.99 | 3.02 | 11.1% | Qwen2-1.5B + LoRA + instruction prefix |
| GPT-2 LoRA-only | 2.24 | 4.20 | 22.2% | GPT-2 pre-trained + LoRA |
| Qwen LoRA-only | 2.59 | 5.42 | 22.2% | Qwen2-1.5B + LoRA |

*Best epoch (epoch 3); small test set variance applies.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up data (see DATA_SETUP.md for full instructions)

```
/data/nuscenes_fixed_matrices/         # Processed nuPlan scene matrices
/data/nuplan_text_finetuned/
    nuplan_text_manifest_vlm_metadata_enhanced.parquet  # INTENT caption + embedding manifest
```

### 3. Run core SEMDT (DT backbone, λ=0.1, INTENT captions) — best result

```bash
python3 trajectory_experiment_visiontrap.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 --max_iters 50 --batch_size 16 \
    --embed_dim 256 --n_layer 3 --n_head 4 \
    --context_length 15 --prediction_horizon 10 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 \
    --enable_visiontrap --enable_oversampling \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 50.0 \
    --traj_loss_type mse --lateral_weight 2.5 \
    --num_steps_per_iter 100 --warmup_steps 500
```

### 4. Reproduce all runs

See **RUNBOOK.md** for the exact command for every result in the paper and extension experiments.

---

## Data Pipeline

The INTENT caption generation pipeline (offline, frozen):

```
nuPlan camera frames
    └─ BLIP-2 (frozen)  →  base scene captions
          └─ GPT-2 (frozen, fine-tuned on driving descriptions)
                + state trends (acceleration, heading Δ, speed)
                → motion-aware INTENT captions + CLIP embeddings
                      → stored in nuplan_text_manifest_vlm_metadata_enhanced.parquet
```

To regenerate captions from scratch:
```bash
python3 generate_prompted_metadata_captions.py
python3 generate_gpt2_vlm_hybrid_manifest.py
```

---

## LoRA Experiments (Extension)

LoRA (Low-Rank Adaptation) injects small trainable `B×A` matrices into the frozen pre-trained backbone:
```
W_adapted = W_frozen + B × A,   rank r << d
```
Only `~1.2–1.3%` of parameters are trained. The semantic stream (InfoNCE) and embedding heads are always trained from scratch.

| File | Purpose |
|---|---|
| `qwen_lora_trajectory_model.py` | Qwen2-1.5B-Instruct backbone + LoRA, DT-style interface |
| `gpt2_lora_trajectory_model.py` | GPT-2 pre-trained backbone + LoRA, DT-style interface |
| `trajectory_experiment_qwen_lora_vlm.py` | Monkey-patches VisionTRAP pipeline with Qwen LoRA model |
| `trajectory_experiment_gpt2_lora_vlm.py` | Monkey-patches VisionTRAP pipeline with GPT-2 LoRA model |

---

## Instruction Tuning Experiments (Extension)

Instruction tuning prepends a tokenized INTENT caption as a natural-language prefix directly into the LLM backbone's input, giving the backbone explicit maneuver intent before processing kinematic tokens:

```
LLM input = [INTENT instruction tokens] + [RTG_1, state_1, action_1, ..., RTG_K, state_K, action_K]
```

This differs from the paper's SEMDT grounding (which aligns latents via contrastive loss) — here the LLM *sees* intent as part of its input sequence.

| File | Purpose |
|---|---|
| `qwen_instruction_lora_model.py` | Qwen instruction model: prefix + DT interleaving + optional LoRA |
| `gpt2_instruction_lora_model.py` | GPT-2 instruction model: prefix + DT interleaving + optional LoRA |
| `trajectory_experiment_qwen_instruction_lora.py` | Full training loop — use `--no_lora` to disable LoRA |
| `trajectory_experiment_gpt2_instruction_lora.py` | Full training loop — use `--no_lora` to disable LoRA |

---

## Citation

If you use this work, please cite:

```bibtex
@article{semdt2026,
  title={SEMDT: Semantic Decision Transformer for Trajectory Generation},
  journal={},
  year={2026}
}
```

---

## License

Research use only.
