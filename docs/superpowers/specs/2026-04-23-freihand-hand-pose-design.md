# FreiHand 3D Hand Pose Estimation — Design Spec

## Goal

Build a top-performing model on the FreiHand benchmark targeting <5.5mm PA-MPJPE, orchestrated by orze on 8× H100 GPUs.

## Architecture

### Backbone
- ViT-H/14 pretrained (DINOv2 or MAE from timm)
- 632M parameters, outputs 1280-dim patch tokens
- Global average pooling → 1280-dim feature vector

### Regression Head
- MLP: 1280 → 1024 → 512 → output
- Outputs MANO parameters:
  - Pose θ: 48 dims (16 joints × 3 axis-angle)
  - Shape β: 10 dims
  - Camera (weak perspective): 3 dims (scale, tx, ty)
- Auxiliary: direct 3D joint regressor (1280 → 1024 → 21×3 = 63 dims)

### MANO Layer
- Differentiable MANO forward pass: (θ, β) → 778 mesh vertices + 21 joints
- Use `manotorch` or custom implementation from MANO pickle files
- User must download MANO model from https://mano.is.tue.mpg.de/

### Loss Function
```
L = λ_j * L1(joints_3d_pred, joints_3d_gt)
  + λ_v * L1(verts_pred, verts_gt)
  + λ_2d * L1(joints_2d_proj, joints_2d_gt)
  + λ_mano * L2(mano_pred, mano_gt)
  + λ_edge * EdgeLengthRegularization
  + λ_norm * NormalConsistencyRegularization
```

Default weights: λ_j=5.0, λ_v=5.0, λ_2d=5.0, λ_mano=0.1, λ_edge=1.0, λ_norm=0.1

## Data

### FreiHand Dataset
- 32,560 training images (×4 augmentation = 130,240 total)
- 3,960 evaluation images
- Annotations: 2D/3D joints (21 keypoints), MANO params, camera intrinsics
- Images: 224×224 hand crops

### Augmentation
- Random rotation: ±90°
- Random scale: 0.75–1.25
- Color jitter: brightness, contrast, saturation, hue
- Random horizontal flip (with left/right joint remapping)
- Random erasing (cutout)

### Optional Extra Data (later ideas)
- InterHand2.6M for pretraining
- HO3D for domain diversity
- Synthetic hand data

## Training Strategy

### Phase 1: Head Warmup
- Freeze ViT backbone
- Train regression head only
- 5 epochs, lr=1e-3, batch_size=64/GPU (512 total)

### Phase 2: End-to-End Fine-tuning
- Unfreeze all layers
- 80 epochs, lr=1e-4, cosine decay to 1e-6
- AdamW, weight_decay=0.01
- Gradient clipping: max_norm=1.0
- Mixed precision (fp16)

### Phase 3: High-Resolution Fine-tuning (optional)
- Increase input to 256×256 or 288×288
- Lower lr=1e-5, 10 epochs

## Evaluation

### Metrics (all in mm, Procrustes-aligned)
- **PA-MPJPE**: primary metric (lower is better)
- **PA-MPVPE**: mesh vertex error
- **F@5mm**: F-score at 5mm threshold
- **F@15mm**: F-score at 15mm threshold

### Evaluation Protocol
- Procrustes alignment before computing errors
- Official FreiHand evaluation script for final numbers
- Codalab submission for leaderboard

## Orze Project Structure

```
hand/
├── orze.yaml              # orze config
├── GOAL.md                # research objective
├── ideas.md               # seed experiments
├── configs/
│   └── base.yaml          # default hyperparameters
├── train_idea.py          # orze training contract
├── evaluate.py            # evaluation script
├── src/
│   ├── model.py           # ViT + MANO regression head
│   ├── mano_layer.py      # differentiable MANO
│   ├── losses.py          # multi-task loss
│   ├── dataset.py         # FreiHand dataloader
│   ├── augmentations.py   # data augmentation pipeline
│   └── utils.py           # Procrustes alignment, metrics
├── scripts/
│   └── download_freihand.py  # dataset download helper
└── results/               # orze output
```

## train_idea.py Contract

Receives from orze:
```bash
python train_idea.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config configs/base.yaml
```

Reads `idea_config.yaml` from `results/{idea_id}/` for merged config.

Outputs `results/{idea_id}/metrics.json`:
```json
{
  "status": "COMPLETED",
  "pa_mpjpe": 5.42,
  "pa_mpvpe": 5.81,
  "f_at_5": 0.756,
  "f_at_15": 0.983,
  "training_time": 3600,
  "epoch": 80,
  "best_epoch": 67
}
```

## Seed Ideas (~10)

1. **ViT-H/14 DINOv2 baseline**: default config, phase 1+2
2. **ViT-L/14 DINOv2**: smaller backbone, faster iteration
3. **ViT-H/14 MAE pretrained**: different pretraining
4. **LR sweep**: lr ∈ [5e-5, 1e-4, 3e-4]
5. **Loss weight sweep**: λ_j ∈ [1.0, 5.0, 10.0]
6. **Higher resolution**: 256×256 input
7. **Heavy augmentation**: add mixup + random erasing
8. **Longer training**: 120 epochs with warmup restart
9. **Joint refinement head**: iterative refinement (2 passes)
10. **Token attention pooling**: replace GAP with attention pooling

## Dependencies

- torch>=2.0
- torchvision
- timm (ViT backbones)
- manotorch or chumpy+MANO (MANO layer)
- pillow, opencv-python
- numpy, scipy (Procrustes alignment)
- pyyaml

## Risks

- MANO model download requires manual registration at mano.is.tue.mpg.de
- FreiHand dataset download may require manual steps
- ViT-H is large — single-GPU OOM possible if batch size too high (should be fine at bs=64 on H100 80GB)
- Procrustes alignment in eval loop can be slow — vectorize with scipy
