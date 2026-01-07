# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mol-LLM Custom is a molecular LLM training framework combining Large Language Models (LLaDA-8B, LLaMA, Mistral) with Graph Neural Networks for molecular structure encoding. It uses a 3-stage training pipeline with LoRA fine-tuning and supports 30+ chemistry tasks.

## Common Commands

### Training

```bash
# Set environment variable first
export TOKENIZERS_PARALLELISM=false

# Stage 1: LLM Pretraining (string-only)
python stage3.py \
  trainer.devices="'0,1,2,3,4,5,6,7'" \
  filename=stage1_llm_pretraining \
  trainer=llada8b \
  trainer.mol_representation=string_only

# Stage 2: Q-Former & GNN Pretraining
python stage3.py \
  trainer.devices="'0,1,2,3,4,5,6,7'" \
  filename=stage2_qformer_pretraining \
  gnn=gine_tokengt \
  trainer=llada8b_stage2 \
  pretrained_ckpt_path="/path/to/stage1/last.ckpt"

# Stage 3: Full Multimodal Training
python stage3.py \
  trainer.devices="'0,1,2,3,4,5,6,7'" \
  filename=stage3_mol-llm \
  trainer=llada8b_stage3 \
  trainer.mol_representation=string+graph \
  pretrained_ckpt_path="/path/to/stage2/last.ckpt"

# Testing/Inference
python stage3.py \
  --config-name test_llada \
  mode=test \
  ckpt_path="/path/to/checkpoint.ckpt"

# Resume from checkpoint
python stage3.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Common Config Overrides

```bash
trainer.devices='0,1,2,3'    # GPU selection
trainer.max_epochs=12        # Training epochs
trainer.batch_size=4         # Per-GPU batch size
data.data_tag=512_Truncation # Data configuration
debug=true                   # Use small dataset subset
```

## Architecture

### Training Pipeline (3 Stages)

1. **Stage 1**: Pretrain LLM on molecular datasets (string-only, LoRA)
2. **Stage 2**: Pretrain Q-Former and GNN for multimodal alignment
3. **Stage 3**: End-to-end multimodal fine-tuning (string + graph)

### Model Stack

```
LLM (LLaDA-8B/LLaMA/Mistral) + LoRA
        ↑
    Q-Former (query-based text-graph fusion)
        ↑
Graph Encoders (GINE + TokenGT)
```

### Key Files

| File | Purpose |
|------|---------|
| `stage3.py` | Main training entry point (PyTorch Lightning) |
| `model/blip2_stage3.py` | Core model architecture (Blip2Stage3 LightningModule) |
| `model/blip2_llada.py` | LLaDA LLM integration |
| `model/blip2qformer.py` | Q-Former implementation |
| `model/gin_model.py` | GINE graph encoder |
| `model/gine_tokengt.py` | TokenGT graph encoder |
| `data_module.py` | Lightning DataModule |
| `data_utils.py` | Task definitions, metrics, data collation |
| `InstructGraph.py` | Instruction-tuned dataset loading |

### Configuration System (Hydra)

- Main config: `configs/train_llada.yaml`
- Trainer configs: `configs/trainer/` (llada8b.yaml, mistral7b_80gb.yaml, etc.)
- Data configs: `configs/data/multi_task_stage*.yaml`
- GNN configs: `configs/gnn/` (gine_tokengt.yaml, TokenGT.yaml)

### Trainable vs Frozen Parameters

- **Frozen**: Base LLM weights
- **Trainable**: LoRA layers, embed_tokens, lm_head, Q-Former, GNN, projectors

LoRA config in `lora_config_llada.json`: targets q/k/v/o_proj, gate/up/down_proj with rank 16-64.

### Special Tokens

Defined in `model/added_tokens.py`:
- SELFIES tokens (from `model/selfies_dict.txt`)
- Molecule format tokens (MOL_2D, MOL_3D, MOL_EMBEDDING)
- Task-specific tokens (DESCRIPTION, IUPAC, MOLFORMULA)

## Data & Tasks

30+ chemistry tasks including:
- Property prediction (BBBP, ClinTox, HIV, ESOL, LogP, QM9 properties)
- Reaction prediction (forward, retrosynthesis, reagent)
- Name conversion (SMILES↔IUPAC, InChI)
- Molecular generation (text-to-mol, mol-to-text)

Molecular representations: SMILES strings, SELFIES, PyTorch Geometric graphs.

## Checkpoint Management

See `CHECKPOINT_GUIDE.md` for detailed usage. Key config options:
- `save_on_n_steps`: Save every N steps (default: 500)
- `save_top_k_checkpoints`: Number of step checkpoints to keep
- `save_top_k_best`: Number of best model checkpoints to keep (default: 3)

Checkpoints only save trainable parameters (LoRA, embeddings, Q-Former, GNN, projectors).

## Development Notes

- No pytest/linting setup - pure PyTorch Lightning
- GPU requirements: ~80GB VRAM (8x A100 recommended for LLaDA-8B)
- WandB integration for experiment tracking (configure `wandb_entity`, `wandb_project`)
- Debug mode (`debug=true`) uses small dataset subset
