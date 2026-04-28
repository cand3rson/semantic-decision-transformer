# SEMDT — Runbook

Complete, copy-paste commands to reproduce every result in the paper and all
extension experiments (LoRA / instruction tuning).

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure your data paths are set up as described in **DATA_SETUP.md**.

Set the correct GPU if needed — most scripts default to `CUDA_VISIBLE_DEVICES=0`
(or `1` in `trajectory_experiment_visiontrap.py`). Override on the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python3 ...
```

---

## Shared data arguments (copy these into every command)

```
--dataset_path /data/nuscenes_fixed_matrices
--text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet
--max_files 130
```

---

## Section 0 — Paper Q1 / Fig. 2: Kinematics-Only Baselines (no semantic grounding)

These runs establish the **tail-risk baseline before semantic grounding** for each backbone.
They are identical to the SEMDT runs below but with `--enable_visiontrap` removed,
so `L_text = 0` throughout training — purely kinematic trajectory learning.

```bash
bash run_dt_kinematics_only.sh        # DT backbone
bash run_gpt2_kinematics_only.sh      # GPT-2 backbone
bash run_llama_kinematics_only.sh     # LLaMA backbone
bash run_qwen_kinematics_only.sh      # Qwen backbone
bash run_deepseek_kinematics_only.sh  # DeepSeek backbone
```

Compare the MR@5m from these runs against the SEMDT INTENT runs (Section 1 & 3)
to reproduce **Fig. 2** in the paper.

---

## Section 1 — Paper: Core SEMDT (DT backbone)

### 1.1 Best run — SEMDT INTENT, λ=0.1

This is the primary result: **ADE=1.67, FDE=3.27, MR@5m=12.8%**

```bash
python3 trajectory_experiment_visiontrap.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 \
    --num_steps_per_iter 100 \
    --batch_size 16 \
    --embed_dim 256 \
    --n_layer 3 \
    --n_head 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 500 \
    --context_length 15 \
    --prediction_horizon 10 \
    --text_loss_weight 0.1 \
    --text_warmup_epochs 10 \
    --text_loss_cap 0.2 \
    --infonce_scale_factor 0.001 \
    --text_queue_size 2048 \
    --enable_visiontrap \
    --enable_oversampling \
    --target_straight_frac 0.25 \
    --target_mild_frac 0.35 \
    --target_strong_frac 0.40 \
    --max_movement_filter 10.0 \
    --movement_variance_filter 50.0 \
    --max_objects 128 \
    --traj_loss_type mse \
    --lateral_weight 2.5 \
    --time_weighting none \
    2>&1 | tee training_semdt_best_$(date +%Y%m%d_%H%M%S).log
```

---

## Section 1b — Paper Table II: All Backbones at λ=0.1 (SEMDT INTENT)

Use these scripts to reproduce **Table II** (backbone comparison under identical supervision):

```bash
bash run_dt_vlm_lambda_01_test.sh        # DT  → ADE=1.67, FDE=3.27, MR=12.8%
bash run_gpt2_vlm_lambda_01_test.sh      # GPT-2 → ADE=2.05, FDE=4.44, MR=25.5%
bash run_llama_vlm_lambda_01_test.sh     # LLaMA → ADE=3.42, FDE=7.25, MR=68.2%
bash run_qwen_vlm_lambda_01_test.sh      # Qwen  → ADE=3.36, FDE=6.39, MR=65.0%
bash run_deepseek_vlm_lambda_01_test.sh  # DeepSeek → ADE=3.52, FDE=8.20, MR=69.3%
```

---

## Section 2 — Paper: Lambda (λ) Sensitivity Tests

Use the provided shell scripts; they are pre-configured.

```bash
# DT backbone
bash run_dt_vlm_lambda_05_test.sh    # λ=0.5
bash run_dt_vlm_lambda_08_test.sh    # λ=0.8

# Backbone swaps at λ=0.5 / 0.8
bash run_gpt2_vlm_lambda_05_test.sh
bash run_gpt2_vlm_lambda_08_test.sh
bash run_llama_vlm_lambda_05_test.sh
bash run_llama_vlm_lambda_08_test.sh
bash run_qwen_vlm_lambda_05_test.sh
bash run_qwen_vlm_lambda_08_test.sh
bash run_deepseek_vlm_lambda_05_test.sh
bash run_deepseek_vlm_lambda_08_test.sh
```

To replicate the λ=0.1 baseline for any backbone, run the matching
`trajectory_experiment_*_vlm.py` script with `--text_loss_weight 0.1`.

---

## Section 3 — Paper: Backbone Swap Comparisons (INTENT, λ=0.1)

All five share the same argument block; only the entry point changes.
The `--embed_dim`, `--n_layer`, `--n_head` flags are accepted but **ignored**
by pretrained backbones (their architecture is fixed by the checkpoint).

### 3.1 SemDT DT (baseline — see Section 1.1)

### 3.2 SemDT GPT-2

```bash
python3 trajectory_experiment_gpt2_vlm.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 --num_steps_per_iter 100 \
    --batch_size 16 --embed_dim 256 --n_layer 3 --n_head 4 \
    --learning_rate 1e-4 --weight_decay 1e-4 --warmup_steps 500 \
    --context_length 15 --prediction_horizon 10 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 --text_queue_size 2048 \
    --enable_visiontrap --enable_oversampling \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 50.0 \
    --max_objects 128 --traj_loss_type mse --lateral_weight 2.5 --time_weighting none \
    2>&1 | tee training_gpt2_semdt_$(date +%Y%m%d_%H%M%S).log
```

### 3.3 SemDT LLaMA

```bash
python3 trajectory_experiment_llama_vlm.py \
    [same args as 3.2 — just change the entry point]
    2>&1 | tee training_llama_semdt_$(date +%Y%m%d_%H%M%S).log
```

### 3.4 SemDT Qwen

```bash
python3 trajectory_experiment_qwen_vlm.py \
    [same args as 3.2]
    2>&1 | tee training_qwen_semdt_$(date +%Y%m%d_%H%M%S).log
```

### 3.5 SemDT DeepSeek

```bash
python3 trajectory_experiment_deepseek_vlm.py \
    [same args as 3.2]
    2>&1 | tee training_deepseek_semdt_$(date +%Y%m%d_%H%M%S).log
```

---

## Section 4 — Extension: LoRA-Only Runs

LoRA adapters are injected into the frozen pre-trained backbone.
Only `~1.2–1.3%` of parameters are trained.
The semantic VisionTRAP stream (InfoNCE) is identical to the paper baseline.

### 4.1 Qwen LoRA-only (best: ADE=2.59, FDE=5.42, MR=22.2%)

```bash
bash run_qwen_lora_vlm_lambda_01_test.sh
```

Or manually:

```bash
python3 trajectory_experiment_qwen_lora_vlm.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 --num_steps_per_iter 100 \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-4 --warmup_steps 500 \
    --context_length 15 --prediction_horizon 10 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 --text_queue_size 2048 \
    --enable_visiontrap --enable_oversampling \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 200.0 \
    --max_objects 128 --traj_loss_type mse --lateral_weight 3.5 --time_weighting none \
    --lr_schedule cosine \
    2>&1 | tee training_qwen_lora_only_$(date +%Y%m%d_%H%M%S).log
```

### 4.2 GPT-2 LoRA-only (best: ADE=2.24, FDE=4.20, MR=22.2%)

```bash
python3 trajectory_experiment_gpt2_lora_vlm.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 --num_steps_per_iter 100 \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-4 --warmup_steps 500 \
    --context_length 15 --prediction_horizon 10 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 --text_queue_size 2048 \
    --enable_visiontrap --enable_oversampling \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 200.0 \
    --max_objects 128 --traj_loss_type mse --lateral_weight 3.5 --time_weighting none \
    --lr_schedule cosine \
    2>&1 | tee training_gpt2_lora_only_$(date +%Y%m%d_%H%M%S).log
```

---

## Section 5 — Extension: Instruction Tuning Runs

Instruction tuning prepends a tokenized INTENT caption as a natural-language
prefix **directly into the LLM backbone's input sequence**, giving the backbone
explicit maneuver intent before it processes the kinematic tokens.

Architecture:
```
LLM input = [tokenized INTENT instruction] + [RTG_1, s_1, a_1, ..., RTG_K, s_K, a_K]
```

The last state token of the DT interleaving is used as the agent embedding,
decoded by an action head to predict H future steps.

### 5.1 Qwen Instruction + LoRA (best: ADE=1.99, FDE=3.02, MR=11.1%)

```bash
python3 trajectory_experiment_qwen_instruction_lora.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --context_length 15 --prediction_horizon 10 \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-4 \
    --num_epochs 5 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 \
    --enable_oversampling --oversampling_temperature 0.75 \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 200.0 \
    --llm_model_name Qwen/Qwen2-1.5B-Instruct \
    2>&1 | tee training_qwen_instruction_lora_$(date +%Y%m%d_%H%M%S).log
```

### 5.2 Qwen Instruction-only (no LoRA)

Same as 5.1, add `--no_lora`:

```bash
python3 trajectory_experiment_qwen_instruction_lora.py \
    [same args as 5.1] \
    --no_lora \
    2>&1 | tee training_qwen_instruction_nolora_$(date +%Y%m%d_%H%M%S).log
```

### 5.3 GPT-2 Instruction + LoRA (best epoch: ADE=1.63, FDE=3.79, MR=11.1%)

```bash
python3 trajectory_experiment_gpt2_instruction_lora.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --context_length 15 --prediction_horizon 10 \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-4 \
    --num_epochs 5 \
    --text_loss_weight 0.1 --text_warmup_epochs 10 \
    --text_loss_cap 0.2 --infonce_scale_factor 0.001 \
    --enable_oversampling --oversampling_temperature 0.75 \
    --target_straight_frac 0.25 --target_mild_frac 0.35 --target_strong_frac 0.40 \
    --max_movement_filter 10.0 --movement_variance_filter 200.0 \
    --gpt2_model_name gpt2 \
    2>&1 | tee training_gpt2_instruction_lora_$(date +%Y%m%d_%H%M%S).log
```

### 5.4 GPT-2 Instruction-only (no LoRA) (best epoch: ADE=1.04, FDE=2.46, MR=11.1%)

Same as 5.3, add `--no_lora`:

```bash
python3 trajectory_experiment_gpt2_instruction_lora.py \
    [same args as 5.3] \
    --no_lora \
    2>&1 | tee training_gpt2_instruction_nolora_$(date +%Y%m%d_%H%M%S).log
```

---

## Section 6 — Caption Generation Pipeline (offline, run once)

These scripts generate the text manifest consumed by all training runs.
You do **not** need to re-run these if the parquet manifest file already exists.

```bash
# Step 1: generate base BLIP-2 + GPT-2 motion-aware captions
python3 generate_prompted_metadata_captions.py

# Step 2: build the combined manifest with CLIP embeddings
python3 generate_gpt2_vlm_hybrid_manifest.py

# Output: /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet
```

---

## Hyperparameter Reference

| Parameter | Value used | What it controls |
|---|---|---|
| `max_files` | 130 | Number of nuPlan scenes loaded |
| `context_length` (K) | 15 | Steps of history fed to the model |
| `prediction_horizon` (H) | 10 | Steps of future trajectory to predict |
| `batch_size` | 16 | Training batch size |
| `embed_dim` | 256 | DT hidden size (ignored for pretrained backbones) |
| `n_layer` | 3 | DT transformer layers (ignored for pretrained backbones) |
| `n_head` | 4 | DT attention heads (ignored for pretrained backbones) |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `warmup_steps` | 500 | Linear LR warmup (VisionTRAP loop) |
| `max_iters` | 50 | Training iterations (VisionTRAP loop) |
| `num_steps_per_iter` | 100 | Batches per iteration (VisionTRAP loop) |
| `num_epochs` | 5 | Epochs (instruction tuning loop) |
| `text_loss_weight` (λ) | 0.1 | InfoNCE weight in total loss |
| `text_warmup_epochs` | 10 | Epochs before full λ is applied |
| `text_loss_cap` | 0.2 | Maximum contribution of text loss (prevents domination) |
| `infonce_scale_factor` | 0.001 | Scale applied to raw InfoNCE value before capping |
| `text_queue_size` | 2048 | Negative sample queue size |
| `max_movement_filter` | 10.0 | Drop samples with any movement > this (m/timestep) |
| `movement_variance_filter` | 50–200 | Drop samples with movement variance > this |
| `lateral_weight` | 2.5–3.5 | MSE weight on lateral dimension |
| `target_straight_frac` | 0.25 | Oversampling target for straight maneuvers |
| `target_mild_frac` | 0.35 | Oversampling target for mild lateral maneuvers |
| `target_strong_frac` | 0.40 | Oversampling target for strong lateral maneuvers |
| LoRA `r` | 16 | LoRA rank |
| LoRA `alpha` | 32 | LoRA scaling factor |
| LoRA `dropout` | 0.05 | LoRA dropout |
