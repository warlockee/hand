# Research Rules

## Goal
3D hand pose estimation on FreiHand. Primary metric: PA-MPJPE (mm, lower is better).
Target: <5.5mm. Read `{results_dir}/report.md` for current leaderboard.

## Current state
- Cycle: {cycle}
- Completed: {completed}
- Queued: {queued}

## Architecture
- ViT backbone (pretrained) → MANO regression head
- Multi-task loss: 3D joints + mesh vertices + 2D reprojection + MANO params + edge regularization
- Two-phase training: frozen backbone warmup, then end-to-end fine-tuning
- Config in `configs/base.yaml`, model in `src/model.py`

## Strategy
1. Establish baselines with ViT-B, ViT-L, ViT-H
2. Sweep learning rates and loss weights on best backbone
3. Try architectural improvements: attention pooling, iterative refinement, graph conv heads
4. Explore extra data: InterHand2.6M, HO3D pretraining
5. Resolution scaling: 224 → 256 → 288

## Idea format
```markdown
## idea-XXXX: Short descriptive title
- **Priority**: high|medium|low
- **Category**: baseline|hyperparameter|loss|architecture|data|training
- **Approach Family**: vit-mano
- **Parent**: idea-NNNN (or none)
- **Hypothesis**: Why this might improve PA-MPJPE.

\```yaml
model:
  backbone: vit_huge_patch14_dinov2.lvd142m
training:
  phase2_lr: 1.0e-4
\```
```

## Config keys
- `model.backbone`: timm model name
- `model.img_size`: input resolution (224, 256, 288)
- `training.phase1_epochs`, `training.phase2_epochs`: epoch counts
- `training.phase1_lr`, `training.phase2_lr`: learning rates
- `training.batch_size`, `training.weight_decay`, `training.max_grad_norm`
- `loss.w_joint`, `loss.w_vertex`, `loss.w_reproj`, `loss.w_mano`, `loss.w_edge`
- `data.data_dir`, `data.img_size`, `data.augment`, `data.scale_to_mm`

## Rules
- **Append-only** — never edit or delete existing ideas
- **Unique IDs** — increment from highest existing
- **Complete configs** — every idea must specify all changed params
- **One change at a time** — isolate variables for clear attribution
- **No code changes** — only modify via config overrides in ideas.md
