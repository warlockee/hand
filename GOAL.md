# Research Goal

## Task: 3D Hand Pose Estimation on FreiHand

- **Dataset**: FreiHand — 32,560 training images (130K with augmentation variants), 3,960 evaluation images. RGB hand crops with MANO annotations (3D joints, mesh vertices, camera intrinsics, MANO pose/shape).
- **Metric**: PA-MPJPE (Procrustes-Aligned Mean Per Joint Position Error, mm) — lower is better
- **Target**: <5.5mm PA-MPJPE
- **Secondary Metrics**: PA-MPVPE (mesh vertex error), F@5mm, F@15mm
- **Current SOTA**: ~5.0mm PA-MPJPE (HaMeR/MeshGraphormer family)

## Current State

Starting from scratch. No trained models or baselines yet.

## Directions

- Backbone variants: ViT-H vs ViT-L, DINOv2 vs MAE vs CLIP pretrained
- Loss weight tuning: relative importance of joint/vertex/reproj losses
- Training schedule: warmup duration, learning rate, cosine vs step decay
- Input resolution: 224 vs 256 vs 288
- Data augmentation intensity and types
- Architecture: attention pooling vs GAP, iterative refinement heads
- Extra training data: InterHand2.6M, HO3D for pretraining
