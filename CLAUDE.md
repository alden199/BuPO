# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BuPO (Bottom-up Policy Optimization) is a research framework for decomposing LLM policies into internal layer and modular policies. Built on top of verl (Volcano Engine Reinforcement Learning), it implements a two-phase RL algorithm that optimizes internal, lower-layer policies before fine-tuning the full model.

## Environment Setup

```bash
# Create and activate conda environment
conda create -y -n bupo python=3.10.17
conda activate bupo

# Install dependencies
pip install -r requirements.txt

# Install flash-attn (CUDA required for compilation)
# Option 1: Use precompiled wheel (recommended)
pip install flash-attn --no-build-isolation

# Option 2: If CUDA toolkit needed for source compilation
conda install -c nvidia cuda-toolkit=12.4 -y
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# Install BuPO in editable mode
pip install -e .
```

## Training Commands

### BuPO Training

```bash
# For Qwen models
bash run_code/BuPO_qwen3.sh

# For Llama models
bash run_code/BuPO_llama.sh
```

Key BuPO parameters in training scripts:
- `k`: Internal layer policy index (which layer to optimize)
- `iterative_steps`: Number of steps for internal policy optimization
- `internal_policy_interative=True`: Enables BuPO mode

### Baseline GRPO Training

```bash
# Standard GRPO without internal policy optimization
bash run_code/GRPO_qwen3.sh
bash run_code/GRPO_llama.sh
```

### Evaluation

```bash
# Generate predictions on test datasets
bash scripts/run_eval.sh
```

### Visualization

```bash
# 1. Generate evaluation dataset
bash scripts/run_eval.sh

# 2. Plot internal policy entropy flow
python visualization/plot_internal_entropy.py
```

## Architecture

### Core Components

**verl/models/custom_model/**
- Custom model implementations that expose internal hidden states
- Modified from HuggingFace transformers to:
  - Return internal layer outputs (not just final layer)
  - Support internal policy computation
  - Use packed inputs with flash attention
- Files: `modeling_llama.py`, `modeling_qwen2.py`, `modeling_qwen3.py`
- Configurations: `configuration_*.py` files

**verl/workers/actor/dp_actor.py**
- Main actor implementation for policy optimization
- `_forward_micro_batch_layer_k()` (line 357): Computes importance ratio of internal layer policy
  - Used when `internal_policy_interative=True`
  - Switches between internal layer optimization and full model optimization based on `global_steps` vs `iterative_steps`
  - Called at lines 808 and 908 during PPO updates

**verl/trainer/main_ppo.py**
- Main PPO training entry point
- Configured via Hydra with extensive command-line overrides
- Handles multi-GPU/multi-node distributed training with Ray

### Data Flow

1. **Input**: Parquet files in `data/` directory containing math reasoning prompts
   - Training: `deepmath-5k.parquet`
   - Validation: `aime_2024.parquet`, `aime_2025.parquet`, `amc2023.parquet`, `math500.parquet`

2. **Training Phase 1 (steps 1-iterative_steps)**:
   - Forward through model up to layer k
   - Compute internal policy log probs
   - Update only layers 0 to k using PPO

3. **Training Phase 2 (steps > iterative_steps)**:
   - Standard full model PPO optimization
   - Uses complete model policy

4. **Output**: Checkpoints saved to `checkpoints/BuPO/{experiment_name}/`

### Key Configuration Parameters

Training scripts use Hydra config overrides. Critical parameters:

```bash
# Model and data
actor_rollout_ref.model.path="${MODEL_PATH}"
data.train_files="${TRAIN_FILE}"
data.max_prompt_length=1024
data.max_response_length=8192

# BuPO specific
actor_rollout_ref.actor.internal_policy_interative=True  # Enable BuPO
actor_rollout_ref.actor.internal_layer=${k}              # Which layer
actor_rollout_ref.actor.iterative_steps=${iterative_steps}  # Phase 1 length

# PPO hyperparameters
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.2
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.optim.lr=1e-6

# Rollout (using vLLM)
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.n=8  # responses per prompt
```

## Development Workflow

### Modifying Internal Policy Logic

The core BuPO logic is in `verl/workers/actor/dp_actor.py`:

1. `_forward_micro_batch_layer_k()`: Modify how internal layer outputs are processed
2. Lines 806-820 and 906-920: Modify when to switch between Phase 1 and Phase 2
3. Test changes by running with small `iterative_steps` value

### Adding New Models

Follow `verl/models/README.md`:

1. Copy model file from HuggingFace transformers to `verl/models/custom_model/`
2. Modify to:
   - Return internal hidden states at each layer
   - Support packed inputs (input_ids, cu_seqlens, max_seqlen_in_batch)
   - Remove KV cache code (training only)
3. Add corresponding configuration file
4. Register in `verl/models/registry.py`

### Running Tests

```bash
# Run specific training script with minimal steps for testing
bash run_code/BuPO_qwen3.sh trainer.total_training_steps=10
```

### Debugging

```bash
# Enable detailed logging
export VERL_LOG_LEVEL=DEBUG

# Use smaller batch sizes and fewer GPUs for faster iteration
trainer.n_gpus_per_node=2 \
data.train_batch_size=16
```

## Important Notes

- This is an editable install (`pip install -e .`), so code changes take effect immediately without reinstalling
- Training requires significant GPU memory (designed for 8x GPUs per node)
- Set `MODEL_PATH` and `DATA_PATH` environment variables before running training scripts
- WandB logging is available but requires configuration (see commented lines in training scripts)
- The project uses Ray for distributed training - ensure Ray is properly configured if using multiple nodes
- Flash attention is strongly recommended for performance but requires CUDA toolkit for compilation
