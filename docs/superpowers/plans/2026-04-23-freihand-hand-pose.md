# FreiHand 3D Hand Pose Estimation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a top-performing 3D hand pose model on FreiHand (<5.5mm PA-MPJPE) using ViT-H + MANO regression, orchestrated by orze on 8× H100 GPUs.

**Architecture:** ViT-H/14 backbone (pretrained) → MLP regression head → MANO parameters (pose, shape, camera). Multi-task loss on 3D joints, mesh vertices, 2D reprojection. Two-phase training: head warmup (frozen backbone) then end-to-end fine-tuning. Orze manages the experiment queue, GPU scheduling, and research loop.

**Tech Stack:** PyTorch 2.8, timm (ViT backbones), manotorch (differentiable MANO), orze (experiment orchestration), scipy (Procrustes alignment)

---

## File Structure

```
hand/
├── orze.yaml                  # orze config — experiment orchestration
├── GOAL.md                    # research objective for orze agents
├── ideas.md                   # seed experiment ideas
├── configs/
│   └── base.yaml              # default hyperparameters
├── train_idea.py              # orze training contract entry point
├── evaluate.py                # post-training evaluation script
├── src/
│   ├── __init__.py
│   ├── mano_layer.py          # differentiable MANO forward pass
│   ├── model.py               # ViT backbone + regression head
│   ├── losses.py              # multi-task loss functions
│   ├── dataset.py             # FreiHand dataset + augmentations
│   ├── metrics.py             # PA-MPJPE, PA-MPVPE, F-scores, Procrustes
│   └── utils.py               # config loading, checkpoint helpers
├── scripts/
│   └── download_freihand.py   # dataset download + extraction
├── tests/
│   ├── test_mano_layer.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_metrics.py
└── results/                   # orze output (created at runtime)
```

---

### Task 1: Environment Setup & Data Download

**Files:**
- Create: `requirements.txt`
- Create: `scripts/download_freihand.py`

- [ ] **Step 1: Install Python dependencies**

```bash
pip install timm manotorch pyyaml tqdm
```

Verify:
```bash
python3 -c "import timm; print(timm.__version__)"
python3 -c "from manotorch.manolayer import ManoLayer; print('MANO OK')"
```

- [ ] **Step 2: Write requirements.txt**

```
torch>=2.0
torchvision
timm>=1.0
manotorch>=0.2
einops
scipy
opencv-python
pillow
pyyaml
tqdm
```

- [ ] **Step 3: Write FreiHand download script**

```python
#!/usr/bin/env python3
"""Download and extract FreiHand dataset."""
import os
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

FREIHAND_URL = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
FREIHAND_EVAL_URL = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip"

def download(url, dest):
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest), reporthook=lambda b, bs, ts: print(f"\r{b*bs/1e6:.0f}/{ts/1e6:.0f} MB", end=""))
    print()

def extract_zip(path, dest):
    print(f"Extracting {path} ...")
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(dest)

def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/freihand")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_zip = data_dir / "FreiHAND_pub_v2.zip"
    eval_zip = data_dir / "FreiHAND_pub_v2_eval.zip"

    download(FREIHAND_URL, train_zip)
    download(FREIHAND_EVAL_URL, eval_zip)
    extract_zip(train_zip, data_dir)
    extract_zip(eval_zip, data_dir)

    print(f"FreiHand dataset ready at {data_dir}")
    expected = data_dir / "training" / "rgb"
    if not expected.exists():
        print(f"WARNING: expected {expected} not found — check zip structure")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Download FreiHand data**

```bash
python3 scripts/download_freihand.py data/freihand
```

Expected: `data/freihand/training/rgb/` with 32560 images, `data/freihand/evaluation/rgb/` with 3960 images. JSON annotation files in `data/freihand/`.

If download URLs fail (they may require manual download from the project page), download manually from https://lmb.informatik.uni-freiburg.de/projects/freihand/ and extract to `data/freihand/`.

- [ ] **Step 5: Verify data structure**

```bash
ls data/freihand/
python3 -c "
import json
from pathlib import Path
d = Path('data/freihand')
# Check training images
imgs = sorted((d / 'training' / 'rgb').glob('*.jpg'))
print(f'Training images: {len(imgs)}')
# Check annotations
for f in ['training_K.json', 'training_xyz.json', 'training_verts.json', 'training_mano.json']:
    p = d / f
    if p.exists():
        data = json.load(open(p))
        print(f'{f}: {len(data)} entries')
    else:
        print(f'{f}: NOT FOUND')
"
```

Expected: 32560 training images, matching annotation counts (or 130240 for augmented versions).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/download_freihand.py
git commit -m "feat: add requirements and FreiHand download script"
```

---

### Task 2: Metrics Module (Procrustes, PA-MPJPE, F-scores)

**Files:**
- Create: `src/__init__.py`
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests for metrics**

```python
# tests/test_metrics.py
import torch
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import (
    procrustes_align,
    pa_mpjpe,
    pa_mpvpe,
    f_score,
    compute_all_metrics,
)


def test_procrustes_identity():
    """Identical inputs → zero error after alignment."""
    pts = torch.randn(21, 3)
    aligned = procrustes_align(pts.numpy(), pts.numpy())
    np.testing.assert_allclose(aligned, pts.numpy(), atol=1e-5)


def test_procrustes_rotation():
    """Rotated input should align back to target."""
    target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    # 90-degree rotation around Z
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    source = (target @ R.T)
    aligned = procrustes_align(source, target)
    np.testing.assert_allclose(aligned, target, atol=1e-5)


def test_pa_mpjpe_zero():
    """Identical predictions → 0 error."""
    pred = torch.randn(4, 21, 3)
    err = pa_mpjpe(pred, pred)
    assert err.shape == (4,)
    assert (err < 1e-5).all()


def test_pa_mpjpe_nonzero():
    """Shifted predictions → nonzero error."""
    gt = torch.zeros(2, 21, 3)
    pred = torch.ones(2, 21, 3) * 10
    err = pa_mpjpe(pred, gt)
    assert err.shape == (2,)
    assert (err > 0).all()


def test_f_score_perfect():
    """Identical meshes → F-score = 1.0."""
    verts = torch.randn(1, 778, 3)
    precision, recall, f = f_score(verts, verts, threshold=5.0)
    assert abs(f.item() - 1.0) < 1e-5


def test_f_score_far():
    """Completely wrong mesh → F-score ≈ 0."""
    pred = torch.zeros(1, 778, 3)
    gt = torch.ones(1, 778, 3) * 1000
    _, _, f = f_score(pred, gt, threshold=5.0)
    assert f.item() < 0.01


def test_compute_all_metrics():
    """Smoke test for full metrics computation."""
    pred_joints = torch.randn(2, 21, 3)
    gt_joints = pred_joints + torch.randn(2, 21, 3) * 0.01
    pred_verts = torch.randn(2, 778, 3)
    gt_verts = pred_verts + torch.randn(2, 778, 3) * 0.01
    m = compute_all_metrics(pred_joints, gt_joints, pred_verts, gt_verts)
    assert "pa_mpjpe" in m
    assert "pa_mpvpe" in m
    assert "f_at_5" in m
    assert "f_at_15" in m
    for v in m.values():
        assert isinstance(v, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement metrics module**

```python
# src/__init__.py
```

```python
# src/metrics.py
"""Procrustes alignment and FreiHand evaluation metrics."""
import numpy as np
import torch
from scipy.spatial.distance import cdist


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Procrustes alignment: translate, scale, rotate source to match target.

    Args:
        source: (N, 3) predicted points
        target: (N, 3) ground truth points

    Returns:
        (N, 3) aligned source
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s_centered = source - mu_s
    t_centered = target - mu_t

    norm_s = np.linalg.norm(s_centered)
    norm_t = np.linalg.norm(t_centered)

    s_centered /= (norm_s + 1e-10)
    t_centered /= (norm_t + 1e-10)

    U, _, Vt = np.linalg.svd(t_centered.T @ s_centered)
    R = U @ Vt

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    aligned = (s_centered @ R.T) * norm_t + mu_t
    return aligned


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-Aligned Mean Per Joint Position Error.

    Args:
        pred: (B, J, 3) predicted joints
        gt: (B, J, 3) ground truth joints

    Returns:
        (B,) per-sample PA-MPJPE in same units as input
    """
    B = pred.shape[0]
    errors = torch.zeros(B)
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    for i in range(B):
        aligned = procrustes_align(pred_np[i], gt_np[i])
        errors[i] = np.linalg.norm(aligned - gt_np[i], axis=1).mean()

    return errors


def pa_mpvpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-Aligned Mean Per Vertex Position Error.

    Same as pa_mpjpe but for mesh vertices (B, V, 3).
    """
    return pa_mpjpe(pred, gt)


def f_score(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """F-score between predicted and GT meshes at a distance threshold.

    Args:
        pred: (B, V, 3) predicted vertices
        gt: (B, V, 3) ground truth vertices
        threshold: distance threshold in mm

    Returns:
        (precision, recall, f_score) each (B,)
    """
    B = pred.shape[0]
    precisions = torch.zeros(B)
    recalls = torch.zeros(B)
    f_scores = torch.zeros(B)

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    for i in range(B):
        dist_p2g = cdist(pred_np[i], gt_np[i]).min(axis=1)
        dist_g2p = cdist(gt_np[i], pred_np[i]).min(axis=1)

        precision = (dist_p2g < threshold).mean()
        recall = (dist_g2p < threshold).mean()

        denom = precision + recall
        if denom > 0:
            f = 2 * precision * recall / denom
        else:
            f = 0.0

        precisions[i] = precision
        recalls[i] = recall
        f_scores[i] = f

    return precisions, recalls, f_scores


def compute_all_metrics(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
) -> dict[str, float]:
    """Compute all FreiHand metrics.

    Inputs should be in mm. Returns dict with scalar values.
    """
    joint_err = pa_mpjpe(pred_joints, gt_joints).mean().item()
    vert_err = pa_mpvpe(pred_verts, gt_verts).mean().item()
    _, _, f5 = f_score(pred_verts, gt_verts, threshold=5.0)
    _, _, f15 = f_score(pred_verts, gt_verts, threshold=15.0)

    return {
        "pa_mpjpe": joint_err,
        "pa_mpvpe": vert_err,
        "f_at_5": f5.mean().item(),
        "f_at_15": f15.mean().item(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/metrics.py tests/test_metrics.py
git commit -m "feat: add Procrustes alignment and FreiHand metrics"
```

---

### Task 3: MANO Layer

**Files:**
- Create: `src/mano_layer.py`
- Create: `tests/test_mano_layer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mano_layer.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mano_layer import MANOHead


@pytest.fixture
def mano():
    return MANOHead()


def test_mano_output_shapes(mano):
    """MANO layer should output correct vertex and joint shapes."""
    B = 4
    pose = torch.zeros(B, 48)
    shape = torch.zeros(B, 10)
    verts, joints = mano(pose, shape)
    assert verts.shape == (B, 778, 3)
    assert joints.shape == (B, 21, 3)


def test_mano_differentiable(mano):
    """MANO layer must support backprop."""
    pose = torch.randn(2, 48, requires_grad=True)
    shape = torch.randn(2, 10, requires_grad=True)
    verts, joints = mano(pose, shape)
    loss = joints.sum()
    loss.backward()
    assert pose.grad is not None
    assert shape.grad is not None


def test_mano_zero_pose(mano):
    """Zero pose should give a rest-pose hand (no NaNs)."""
    pose = torch.zeros(1, 48)
    shape = torch.zeros(1, 10)
    verts, joints = mano(pose, shape)
    assert not torch.isnan(verts).any()
    assert not torch.isnan(joints).any()


def test_mano_batch_consistency(mano):
    """Same input in a batch should give same output."""
    pose = torch.randn(1, 48)
    shape = torch.randn(1, 10)
    pose_batch = pose.expand(3, -1).contiguous()
    shape_batch = shape.expand(3, -1).contiguous()
    verts, joints = mano(pose_batch, shape_batch)
    torch.testing.assert_close(verts[0], verts[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(verts[1], verts[2], atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_mano_layer.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement MANO layer wrapper**

```python
# src/mano_layer.py
"""Differentiable MANO hand model wrapper using manotorch."""
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer


class MANOHead(nn.Module):
    """Wraps manotorch ManoLayer, exposing (pose, shape) → (vertices, joints)."""

    def __init__(self, mano_assets_root: str = "assets/mano", side: str = "right", joint_type: str = "mano"):
        super().__init__()
        self.mano = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root=mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.joint_type = joint_type

    def forward(
        self, pose: torch.Tensor, shape: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose: (B, 48) axis-angle per joint (16 joints × 3)
            shape: (B, 10) MANO shape parameters

        Returns:
            vertices: (B, 778, 3)
            joints: (B, 21, 3)
        """
        mano_out = self.mano(pose, shape)
        vertices = mano_out.verts
        joints = mano_out.joints

        return vertices, joints
```

- [ ] **Step 4: Set up MANO assets**

manotorch downloads assets automatically on first use to `~/.manotorch/`. If that fails, manually:

```bash
python3 -c "from manotorch.manolayer import ManoLayer; m = ManoLayer(rot_mode='axisang', side='right', use_pca=False, flat_hand_mean=True); print('MANO assets OK')"
```

If manotorch has its own assets path, the `mano_assets_root` default in `MANOHead.__init__` should point there. Update the default if needed based on where manotorch stores its data.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_mano_layer.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/mano_layer.py tests/test_mano_layer.py
git commit -m "feat: add differentiable MANO layer wrapper"
```

---

### Task 4: Model (ViT Backbone + Regression Head)

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import HandPoseModel


@pytest.fixture
def model_small():
    """Small ViT for fast testing."""
    return HandPoseModel(backbone_name="vit_small_patch14_dinov2.lvd142m", img_size=224)


def test_model_output_keys(model_small):
    """Model should output all required keys."""
    x = torch.randn(2, 3, 224, 224)
    out = model_small(x)
    assert "mano_pose" in out
    assert "mano_shape" in out
    assert "camera" in out
    assert "joints_3d" in out
    assert "vertices" in out


def test_model_output_shapes(model_small):
    """All outputs should have correct shapes."""
    x = torch.randn(2, 3, 224, 224)
    out = model_small(x)
    assert out["mano_pose"].shape == (2, 48)
    assert out["mano_shape"].shape == (2, 10)
    assert out["camera"].shape == (2, 3)
    assert out["joints_3d"].shape == (2, 21, 3)
    assert out["vertices"].shape == (2, 778, 3)


def test_model_differentiable(model_small):
    """Model should support backprop."""
    x = torch.randn(1, 3, 224, 224)
    out = model_small(x)
    loss = out["joints_3d"].sum()
    loss.backward()
    # Check backbone grads exist
    for p in model_small.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()
            break


def test_model_freeze_backbone(model_small):
    """Frozen backbone should have no gradients."""
    model_small.freeze_backbone()
    x = torch.randn(1, 3, 224, 224)
    out = model_small(x)
    loss = out["joints_3d"].sum()
    loss.backward()
    for p in model_small.backbone.parameters():
        assert p.grad is None or (p.grad == 0).all()


def test_model_unfreeze_backbone(model_small):
    """Unfreezing should re-enable gradients."""
    model_small.freeze_backbone()
    model_small.unfreeze_backbone()
    for p in model_small.backbone.parameters():
        assert p.requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement model**

```python
# src/model.py
"""ViT backbone + MANO regression head for 3D hand pose estimation."""
import torch
import torch.nn as nn
import timm

from src.mano_layer import MANOHead


class HandPoseModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_huge_patch14_dinov2.lvd142m",
        img_size: int = 224,
        mano_assets_root: str = "assets/mano",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, img_size=img_size, num_classes=0
        )
        feat_dim = self.backbone.num_features

        self.head_pose = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 48),
        )
        self.head_shape = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.head_camera = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.mano = MANOHead(mano_assets_root=mano_assets_root)

        self._init_head(self.head_pose)
        self._init_head(self.head_shape)
        self._init_head(self.head_camera)

    @staticmethod
    def _init_head(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        mano_pose = self.head_pose(features)
        mano_shape = self.head_shape(features)
        camera = self.head_camera(features)
        vertices, joints_3d = self.mano(mano_pose, mano_shape)

        return {
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "camera": camera,
            "joints_3d": joints_3d,
            "vertices": vertices,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: all 5 tests PASS (note: first run downloads ViT-S weights ~89MB)

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add ViT + MANO regression model"
```

---

### Task 5: Loss Functions

**Files:**
- Create: `src/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_losses.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.losses import (
    joint_loss,
    vertex_loss,
    mano_param_loss,
    reprojection_loss,
    edge_length_loss,
    HandPoseLoss,
)


def test_joint_loss_zero():
    pred = torch.randn(4, 21, 3)
    assert joint_loss(pred, pred).item() < 1e-6


def test_joint_loss_positive():
    pred = torch.zeros(4, 21, 3)
    gt = torch.ones(4, 21, 3)
    assert joint_loss(pred, gt).item() > 0


def test_vertex_loss_shape():
    pred = torch.randn(2, 778, 3)
    gt = torch.randn(2, 778, 3)
    l = vertex_loss(pred, gt)
    assert l.dim() == 0  # scalar


def test_mano_param_loss():
    pred_pose = torch.randn(2, 48)
    gt_pose = torch.randn(2, 48)
    pred_shape = torch.randn(2, 10)
    gt_shape = torch.randn(2, 10)
    l = mano_param_loss(pred_pose, gt_pose, pred_shape, gt_shape)
    assert l.item() > 0


def test_reprojection_loss():
    joints_3d = torch.randn(2, 21, 3)
    camera = torch.tensor([[1.0, 0.0, 0.0]] * 2)
    gt_2d = torch.randn(2, 21, 2)
    l = reprojection_loss(joints_3d, camera, gt_2d)
    assert l.dim() == 0


def test_edge_length_loss_zero():
    """Same mesh → zero edge loss."""
    verts = torch.randn(2, 778, 3)
    faces = torch.tensor([[0, 1, 2], [1, 2, 3]])
    l = edge_length_loss(verts, verts, faces)
    assert l.item() < 1e-6


def test_hand_pose_loss_smoke():
    """Full loss module should compute without error."""
    faces = torch.tensor([[0, 1, 2], [1, 2, 3]])
    loss_fn = HandPoseLoss(faces=faces)

    pred = {
        "joints_3d": torch.randn(2, 21, 3),
        "vertices": torch.randn(2, 778, 3),
        "mano_pose": torch.randn(2, 48),
        "mano_shape": torch.randn(2, 10),
        "camera": torch.tensor([[1.0, 0.0, 0.0]] * 2),
    }
    gt = {
        "joints_3d": torch.randn(2, 21, 3),
        "vertices": torch.randn(2, 778, 3),
        "joints_2d": torch.randn(2, 21, 2),
        "mano_pose": torch.randn(2, 48),
        "mano_shape": torch.randn(2, 10),
    }
    total, components = loss_fn(pred, gt)
    assert total.dim() == 0
    assert "joint" in components
    assert "vertex" in components
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_losses.py -v
```

- [ ] **Step 3: Implement loss functions**

```python
# src/losses.py
"""Multi-task loss functions for hand pose estimation."""
import torch
import torch.nn as nn


def joint_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, gt)


def vertex_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, gt)


def mano_param_loss(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pred_shape: torch.Tensor,
    gt_shape: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.mse_loss(pred_pose, gt_pose) + nn.functional.mse_loss(
        pred_shape, gt_shape
    )


def project_3d_to_2d(
    joints_3d: torch.Tensor, camera: torch.Tensor
) -> torch.Tensor:
    """Weak perspective projection: 2d = s * xy + t."""
    s = camera[:, 0:1]  # (B, 1)
    tx = camera[:, 1:2]
    ty = camera[:, 2:3]
    x = s.unsqueeze(-1) * joints_3d[:, :, 0:1] + tx.unsqueeze(-1)
    y = s.unsqueeze(-1) * joints_3d[:, :, 1:2] + ty.unsqueeze(-1)
    return torch.cat([x, y], dim=-1)


def reprojection_loss(
    joints_3d: torch.Tensor, camera: torch.Tensor, gt_2d: torch.Tensor
) -> torch.Tensor:
    proj_2d = project_3d_to_2d(joints_3d, camera)
    return nn.functional.l1_loss(proj_2d, gt_2d)


def edge_length_loss(
    pred_verts: torch.Tensor, gt_verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    """Penalizes differences in edge lengths between pred and GT meshes."""
    edges = torch.cat(
        [faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], dim=0
    )
    pred_edge_vec = pred_verts[:, edges[:, 0]] - pred_verts[:, edges[:, 1]]
    gt_edge_vec = gt_verts[:, edges[:, 0]] - gt_verts[:, edges[:, 1]]
    pred_len = pred_edge_vec.norm(dim=-1)
    gt_len = gt_edge_vec.norm(dim=-1)
    return nn.functional.l1_loss(pred_len, gt_len)


class HandPoseLoss(nn.Module):
    def __init__(
        self,
        faces: torch.Tensor,
        w_joint: float = 5.0,
        w_vertex: float = 5.0,
        w_reproj: float = 5.0,
        w_mano: float = 0.1,
        w_edge: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("faces", faces)
        self.w_joint = w_joint
        self.w_vertex = w_vertex
        self.w_reproj = w_reproj
        self.w_mano = w_mano
        self.w_edge = w_edge

    def forward(
        self, pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_joint = joint_loss(pred["joints_3d"], gt["joints_3d"])
        l_vert = vertex_loss(pred["vertices"], gt["vertices"])
        l_mano = mano_param_loss(
            pred["mano_pose"], gt["mano_pose"], pred["mano_shape"], gt["mano_shape"]
        )
        l_reproj = reprojection_loss(
            pred["joints_3d"], pred["camera"], gt["joints_2d"]
        )
        l_edge = edge_length_loss(pred["vertices"], gt["vertices"], self.faces)

        total = (
            self.w_joint * l_joint
            + self.w_vertex * l_vert
            + self.w_reproj * l_reproj
            + self.w_mano * l_mano
            + self.w_edge * l_edge
        )

        components = {
            "joint": l_joint.item(),
            "vertex": l_vert.item(),
            "reproj": l_reproj.item(),
            "mano": l_mano.item(),
            "edge": l_edge.item(),
        }
        return total, components
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_losses.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: add multi-task loss functions for hand pose"
```

---

### Task 6: FreiHand Dataset & Augmentations

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
import torch
import pytest
import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import FreiHandDataset, get_augmentation


@pytest.fixture
def mock_freihand(tmp_path):
    """Create a tiny mock FreiHand dataset."""
    rgb_dir = tmp_path / "training" / "rgb"
    rgb_dir.mkdir(parents=True)

    n = 8
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(rgb_dir / f"{i:08d}.jpg")

    # Annotations
    K = [[[600, 0, 112], [0, 600, 112], [0, 0, 1]]] * n
    xyz = [np.random.randn(21, 3).tolist()] * n
    verts = [np.random.randn(778, 3).tolist()] * n
    # MANO: list of [pose_48, shape_10]
    mano = [{"pose": np.random.randn(48).tolist(), "shape": np.random.randn(10).tolist()}] * n

    json.dump(K, open(tmp_path / "training_K.json", "w"))
    json.dump(xyz, open(tmp_path / "training_xyz.json", "w"))
    json.dump(verts, open(tmp_path / "training_verts.json", "w"))
    json.dump(mano, open(tmp_path / "training_mano.json", "w"))

    return tmp_path


def test_dataset_length(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224)
    assert len(ds) == 8


def test_dataset_item_keys(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224)
    item = ds[0]
    assert "image" in item
    assert "joints_3d" in item
    assert "vertices" in item
    assert "joints_2d" in item
    assert "mano_pose" in item
    assert "mano_shape" in item
    assert "K" in item


def test_dataset_item_shapes(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["joints_3d"].shape == (21, 3)
    assert item["vertices"].shape == (778, 3)
    assert item["joints_2d"].shape == (21, 2)
    assert item["mano_pose"].shape == (48,)
    assert item["mano_shape"].shape == (10,)


def test_dataset_image_normalized(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224)
    item = ds[0]
    assert item["image"].dtype == torch.float32
    assert item["image"].min() >= -3.0  # after normalization
    assert item["image"].max() <= 3.0


def test_augmentation_returns_same_keys(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224, augment=True)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["joints_3d"].shape == (21, 3)


def test_dataloader_batch(mock_freihand):
    ds = FreiHandDataset(mock_freihand, split="train", img_size=224)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["joints_3d"].shape == (4, 21, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset.py -v
```

- [ ] **Step 3: Implement dataset**

```python
# src/dataset.py
"""FreiHand dataset loader with augmentations."""
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_augmentation(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.0),  # disabled: would need joint remapping
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def project_3d_to_2d_np(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D joints to 2D using camera intrinsics. (N,3) + (3,3) → (N,2)."""
    uv = (K @ xyz.T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)
    return uv


class FreiHandDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        scale_to_mm: float = 1000.0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.scale_to_mm = scale_to_mm

        if split == "train":
            self.rgb_dir = self.data_dir / "training" / "rgb"
            prefix = "training"
        else:
            self.rgb_dir = self.data_dir / "evaluation" / "rgb"
            prefix = "evaluation"

        self.images = sorted(self.rgb_dir.glob("*.jpg"))

        self.K = np.array(json.load(open(self.data_dir / f"{prefix}_K.json")), dtype=np.float32)
        self.xyz = np.array(json.load(open(self.data_dir / f"{prefix}_xyz.json")), dtype=np.float32)
        self.verts = np.array(json.load(open(self.data_dir / f"{prefix}_verts.json")), dtype=np.float32)

        mano_path = self.data_dir / f"{prefix}_mano.json"
        if mano_path.exists():
            raw = json.load(open(mano_path))
            if isinstance(raw[0], dict):
                self.mano_pose = np.array([m["pose"] for m in raw], dtype=np.float32)
                self.mano_shape = np.array([m["shape"] for m in raw], dtype=np.float32)
            else:
                arr = np.array(raw, dtype=np.float32)
                self.mano_pose = arr[:, :48]
                self.mano_shape = arr[:, 48:58]
        else:
            n = len(self.images)
            self.mano_pose = np.zeros((n, 48), dtype=np.float32)
            self.mano_shape = np.zeros((n, 10), dtype=np.float32)

        if augment:
            self.transform = get_augmentation(img_size)
        else:
            self.transform = get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        img_tensor = self.transform(img)

        xyz = self.xyz[idx]  # (21, 3)
        K = self.K[idx]  # (3, 3)
        joints_2d = project_3d_to_2d_np(xyz, K)

        # Normalize 2D joints to [0, 1] range based on image size
        orig_size = max(img.size)
        joints_2d = joints_2d / orig_size

        return {
            "image": img_tensor,
            "joints_3d": torch.from_numpy(xyz * self.scale_to_mm).float(),
            "vertices": torch.from_numpy(self.verts[idx] * self.scale_to_mm).float(),
            "joints_2d": torch.from_numpy(joints_2d).float(),
            "mano_pose": torch.from_numpy(self.mano_pose[idx]).float(),
            "mano_shape": torch.from_numpy(self.mano_shape[idx]).float(),
            "K": torch.from_numpy(K).float(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dataset.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add FreiHand dataset loader with augmentations"
```

---

### Task 7: Utils (Config Loading, Checkpoints)

**Files:**
- Create: `src/utils.py`

- [ ] **Step 1: Implement utils**

```python
# src/utils.py
"""Config loading and checkpoint helpers."""
import yaml
import torch
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str | Path,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})
```

- [ ] **Step 2: Commit**

```bash
git add src/utils.py
git commit -m "feat: add config and checkpoint utils"
```

---

### Task 8: Training Script (Orze Contract)

**Files:**
- Create: `train_idea.py`
- Create: `configs/base.yaml`

- [ ] **Step 1: Create base config**

```yaml
# configs/base.yaml
model:
  backbone: vit_huge_patch14_dinov2.lvd142m
  img_size: 224

training:
  phase1_epochs: 5
  phase1_lr: 1.0e-3
  phase2_epochs: 80
  phase2_lr: 1.0e-4
  min_lr: 1.0e-6
  batch_size: 64
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_epochs: 3
  num_workers: 8

loss:
  w_joint: 5.0
  w_vertex: 5.0
  w_reproj: 5.0
  w_mano: 0.1
  w_edge: 1.0

data:
  data_dir: data/freihand
  img_size: 224
  augment: true
  scale_to_mm: 1000.0
```

- [ ] **Step 2: Write training script**

```python
#!/usr/bin/env python3
"""Orze training contract for FreiHand hand pose estimation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))

from src.model import HandPoseModel
from src.losses import HandPoseLoss
from src.dataset import FreiHandDataset
from src.metrics import compute_all_metrics
from src.utils import load_config, merge_configs, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--idea-id", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--ideas-md", default="ideas.md")
    p.add_argument("--config", default="configs/base.yaml")
    return p.parse_args()


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, max_grad_norm):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt = {k: v.to(device) for k, v in batch.items() if k != "image" and k != "K"}

        optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            pred = model(images)
            loss, _ = loss_fn(pred, gt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred_joints = []
    all_gt_joints = []
    all_pred_verts = []
    all_gt_verts = []

    for batch in loader:
        images = batch["image"].to(device)
        with autocast(dtype=torch.float16):
            pred = model(images)

        all_pred_joints.append(pred["joints_3d"].float().cpu())
        all_gt_joints.append(batch["joints_3d"])
        all_pred_verts.append(pred["vertices"].float().cpu())
        all_gt_verts.append(batch["vertices"])

    pred_joints = torch.cat(all_pred_joints)
    gt_joints = torch.cat(all_gt_joints)
    pred_verts = torch.cat(all_pred_verts)
    gt_verts = torch.cat(all_gt_verts)

    return compute_all_metrics(pred_joints, gt_joints, pred_verts, gt_verts)


def main():
    args = parse_args()

    result_dir = Path(args.results_dir) / args.idea_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load base + idea config
    cfg = load_config(args.config)
    idea_config_path = result_dir / "idea_config.yaml"
    if idea_config_path.exists():
        idea_cfg = load_config(idea_config_path)
        cfg = merge_configs(cfg, idea_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Model
    model = HandPoseModel(
        backbone_name=cfg["model"]["backbone"],
        img_size=cfg["model"]["img_size"],
    ).to(device)

    # Data
    data_cfg = cfg.get("data", {})
    train_ds = FreiHandDataset(
        data_cfg.get("data_dir", "data/freihand"),
        split="train",
        img_size=data_cfg.get("img_size", 224),
        augment=data_cfg.get("augment", True),
        scale_to_mm=data_cfg.get("scale_to_mm", 1000.0),
    )
    val_ds = FreiHandDataset(
        data_cfg.get("data_dir", "data/freihand"),
        split="train",  # FreiHand uses training set for validation via split
        img_size=data_cfg.get("img_size", 224),
        augment=False,
        scale_to_mm=data_cfg.get("scale_to_mm", 1000.0),
    )

    tcfg = cfg.get("training", {})
    bs = tcfg.get("batch_size", 64)
    nw = tcfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Loss
    faces = model.mano.mano.th_faces.long()
    lcfg = cfg.get("loss", {})
    loss_fn = HandPoseLoss(
        faces=faces,
        w_joint=lcfg.get("w_joint", 5.0),
        w_vertex=lcfg.get("w_vertex", 5.0),
        w_reproj=lcfg.get("w_reproj", 5.0),
        w_mano=lcfg.get("w_mano", 0.1),
        w_edge=lcfg.get("w_edge", 1.0),
    ).to(device)

    scaler = GradScaler()
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)

    best_metrics = None
    best_epoch = 0

    # Phase 1: Frozen backbone
    model.freeze_backbone()
    p1_epochs = tcfg.get("phase1_epochs", 5)
    p1_lr = tcfg.get("phase1_lr", 1e-3)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p1_lr, weight_decay=tcfg.get("weight_decay", 0.01),
    )

    print(f"Phase 1: {p1_epochs} epochs, lr={p1_lr}, backbone frozen")
    for epoch in range(p1_epochs):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, max_grad_norm)
        print(f"  Epoch {epoch+1}/{p1_epochs} loss={loss:.4f}")

    # Phase 2: End-to-end
    model.unfreeze_backbone()
    p2_epochs = tcfg.get("phase2_epochs", 80)
    p2_lr = tcfg.get("phase2_lr", 1e-4)
    min_lr = tcfg.get("min_lr", 1e-6)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p2_lr, weight_decay=tcfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=p2_epochs, eta_min=min_lr,
    )

    print(f"Phase 2: {p2_epochs} epochs, lr={p2_lr}, end-to-end")
    for epoch in range(p2_epochs):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, max_grad_norm)
        scheduler.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == p2_epochs - 1:
            metrics = evaluate(model, val_loader, device)
            print(f"  Epoch {p1_epochs + epoch + 1} loss={loss:.4f} PA-MPJPE={metrics['pa_mpjpe']:.2f}mm")

            if best_metrics is None or metrics["pa_mpjpe"] < best_metrics["pa_mpjpe"]:
                best_metrics = metrics
                best_epoch = p1_epochs + epoch + 1
                save_checkpoint(model, optimizer, best_epoch, metrics, result_dir / "best_model.pt")
        else:
            print(f"  Epoch {p1_epochs + epoch + 1} loss={loss:.4f}")

    # Save final metrics
    training_time = time.time() - start_time
    output = {
        "status": "COMPLETED",
        "pa_mpjpe": best_metrics["pa_mpjpe"] if best_metrics else 999.0,
        "pa_mpvpe": best_metrics["pa_mpvpe"] if best_metrics else 999.0,
        "f_at_5": best_metrics["f_at_5"] if best_metrics else 0.0,
        "f_at_15": best_metrics["f_at_15"] if best_metrics else 0.0,
        "training_time": training_time,
        "best_epoch": best_epoch,
        "total_epochs": p1_epochs + p2_epochs,
    }
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. PA-MPJPE={output['pa_mpjpe']:.2f}mm in {training_time:.0f}s")
    print(f"Results saved to {result_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add configs/base.yaml train_idea.py
git commit -m "feat: add training script and base config (orze contract)"
```

---

### Task 9: Evaluation Script

**Files:**
- Create: `evaluate.py`

- [ ] **Step 1: Write evaluation script**

```python
#!/usr/bin/env python3
"""Post-training evaluation for orze (optional eval_script)."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))

from src.model import HandPoseModel
from src.dataset import FreiHandDataset
from src.metrics import compute_all_metrics
from src.utils import load_config, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--idea-id", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--gpu", default="0")
    p.add_argument("--config", default="configs/base.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.results_dir) / args.idea_id
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    idea_config_path = result_dir / "idea_config.yaml"
    if idea_config_path.exists():
        from src.utils import merge_configs
        cfg = merge_configs(cfg, load_config(idea_config_path))

    model = HandPoseModel(
        backbone_name=cfg["model"]["backbone"],
        img_size=cfg["model"]["img_size"],
    ).to(device)

    ckpt_path = result_dir / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, evaluating random model")

    data_cfg = cfg.get("data", {})
    val_ds = FreiHandDataset(
        data_cfg.get("data_dir", "data/freihand"),
        split="train",
        img_size=data_cfg.get("img_size", 224),
        augment=False,
        scale_to_mm=data_cfg.get("scale_to_mm", 1000.0),
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    all_pred_joints, all_gt_joints = [], []
    all_pred_verts, all_gt_verts = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            with autocast(dtype=torch.float16):
                pred = model(images)

            all_pred_joints.append(pred["joints_3d"].float().cpu())
            all_gt_joints.append(batch["joints_3d"])
            all_pred_verts.append(pred["vertices"].float().cpu())
            all_gt_verts.append(batch["vertices"])

    metrics = compute_all_metrics(
        torch.cat(all_pred_joints), torch.cat(all_gt_joints),
        torch.cat(all_pred_verts), torch.cat(all_gt_verts),
    )

    eval_report = {**metrics, "checkpoint": str(ckpt_path)}
    with open(result_dir / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Evaluation: PA-MPJPE={metrics['pa_mpjpe']:.2f}mm PA-MPVPE={metrics['pa_mpvpe']:.2f}mm "
          f"F@5={metrics['f_at_5']:.4f} F@15={metrics['f_at_15']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: add evaluation script for post-training eval"
```

---

### Task 10: Orze Configuration (orze.yaml, GOAL.md, ideas.md)

**Files:**
- Create: `orze.yaml`
- Create: `GOAL.md`
- Create: `ideas.md`

- [ ] **Step 1: Write GOAL.md**

```markdown
# Research Goal

## Task: 3D Hand Pose Estimation on FreiHand

- **Dataset**: FreiHand — 32,560 training images (130K with augmentation variants), 3,960 evaluation images. RGB hand crops with MANO annotations.
- **Metric**: PA-MPJPE (Procrustes-Aligned Mean Per Joint Position Error, mm) — lower is better
- **Target**: <5.5mm PA-MPJPE
- **Secondary Metrics**: PA-MPVPE, F@5mm, F@15mm
- **Current SOTA**: ~5.0mm PA-MPJPE (HaMeR/MeshGraphormer family)

## Approach

ViT backbone (pretrained) + MANO regression head. Multi-task loss on 3D joints, mesh vertices, 2D reprojection, edge regularization.

## Directions to Explore

- Backbone variants: ViT-H vs ViT-L, DINOv2 vs MAE vs CLIP pretrained
- Loss weight tuning: relative importance of joint/vertex/reproj losses
- Training schedule: warmup duration, learning rate, cosine vs step decay
- Input resolution: 224 vs 256 vs 288
- Data augmentation: intensity, types, mixup
- Architecture: attention pooling vs GAP, iterative refinement, graph conv refinement
- Extra training data: InterHand2.6M, HO3D, synthetic data
```

- [ ] **Step 2: Write orze.yaml**

```yaml
train_script: train_idea.py
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: python3

timeout: 14400
poll: 30
stall_minutes: 30
max_idea_failures: 3

gpu_scheduling:
  mode: exclusive

report:
  title: "FreiHand Hand Pose"
  primary_metric: pa_mpjpe
  sort: ascending
  columns:
    - {key: "pa_mpjpe", label: "PA-MPJPE (mm)", fmt: ".2f"}
    - {key: "pa_mpvpe", label: "PA-MPVPE (mm)", fmt: ".2f"}
    - {key: "f_at_5", label: "F@5mm", fmt: ".4f"}
    - {key: "f_at_15", label: "F@15mm", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}

notifications:
  enabled: true
  on: [completed, failed, new_best, heartbeat]
  heartbeat_interval: 1800
  channels:
    - type: telegram
      bot_token: "8214019299:AAEbnK-wEqabNue7EHbVhfwlniLB3dtC68Y"
      chat_id: "CHAT_ID_PLACEHOLDER"
```

**NOTE**: Replace `CHAT_ID_PLACEHOLDER` with actual chat_id once retrieved from the Telegram bot.

- [ ] **Step 3: Write ideas.md with seed experiments**

```markdown
## idea-001: ViT-H DINOv2 Baseline
- **Priority**: high
- **Category**: baseline
- **Approach Family**: vit-mano
- **Parent**: none
- **Hypothesis**: ViT-H/14 DINOv2 is the strongest pretrained backbone for dense prediction. Default config should establish a solid baseline.

```yaml
model:
  backbone: vit_huge_patch14_dinov2.lvd142m
training:
  phase1_epochs: 5
  phase2_epochs: 80
```

## idea-002: ViT-L DINOv2 (Faster Iteration)
- **Priority**: high
- **Category**: baseline
- **Approach Family**: vit-mano
- **Parent**: none
- **Hypothesis**: ViT-L is 3x faster to train than ViT-H. Use as fast iteration backbone to test ideas before scaling.

```yaml
model:
  backbone: vit_large_patch14_dinov2.lvd142m
training:
  phase2_epochs: 60
```

## idea-003: ViT-H MAE Pretrained
- **Priority**: medium
- **Category**: backbone
- **Approach Family**: vit-mano
- **Parent**: none
- **Hypothesis**: MAE pretraining focuses on pixel reconstruction which may transfer better to dense mesh regression than DINOv2's contrastive objective.

```yaml
model:
  backbone: vit_huge_patch14_224.mae
training:
  phase2_epochs: 80
```

## idea-004: LR Sweep Low
- **Priority**: high
- **Category**: hyperparameter
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Lower learning rate may prevent overshooting on fine-grained joint regression.

```yaml
training:
  phase2_lr: 5.0e-5
```

## idea-005: LR Sweep High
- **Priority**: high
- **Category**: hyperparameter
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Higher learning rate may converge faster with proper warmup.

```yaml
training:
  phase2_lr: 3.0e-4
  warmup_epochs: 5
```

## idea-006: Joint Loss Dominant
- **Priority**: medium
- **Category**: loss
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Increasing joint loss weight may improve PA-MPJPE directly since that is the primary metric.

```yaml
loss:
  w_joint: 10.0
  w_vertex: 2.0
```

## idea-007: Higher Resolution 256
- **Priority**: medium
- **Category**: architecture
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Larger input (256x256) preserves more spatial detail for small finger joints.

```yaml
model:
  img_size: 256
data:
  img_size: 256
training:
  batch_size: 48
```

## idea-008: Strong Augmentation
- **Priority**: medium
- **Category**: data
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Aggressive augmentation should improve generalization on the small FreiHand training set.

```yaml
data:
  augment: true
training:
  phase2_epochs: 100
```

## idea-009: Longer Training 120 Epochs
- **Priority**: low
- **Category**: training
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: 80 epochs may not be enough for ViT-H to fully converge. More epochs with cosine decay could squeeze out extra performance.

```yaml
training:
  phase2_epochs: 120
  min_lr: 1.0e-7
```

## idea-010: Edge + Normal Loss Emphasis
- **Priority**: low
- **Category**: loss
- **Approach Family**: vit-mano
- **Parent**: idea-001
- **Hypothesis**: Stronger mesh regularization may produce smoother, more accurate mesh vertices.

```yaml
loss:
  w_edge: 3.0
  w_mano: 0.5
```
```

- [ ] **Step 4: Commit**

```bash
git add orze.yaml GOAL.md ideas.md
git commit -m "feat: add orze config, research goal, and 10 seed experiment ideas"
```

---

### Task 11: Integration Test & Launch

- [ ] **Step 1: Validate orze config**

```bash
orze --check
```

Expected: config validation passes (may warn about missing chat_id).

- [ ] **Step 2: Smoke test training on 1 GPU with tiny run**

Create a quick test to verify the full pipeline works end-to-end:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_idea.py \
  --idea-id smoke-test \
  --results-dir results \
  --config configs/base.yaml
```

If the full dataset is too slow for a smoke test, temporarily override epochs:

```bash
mkdir -p results/smoke-test
cat > results/smoke-test/idea_config.yaml << 'EOF'
training:
  phase1_epochs: 1
  phase2_epochs: 1
  batch_size: 4
  num_workers: 2
EOF
CUDA_VISIBLE_DEVICES=0 python3 train_idea.py \
  --idea-id smoke-test \
  --results-dir results \
  --config configs/base.yaml
```

Expected: `results/smoke-test/metrics.json` with `status: COMPLETED`.

- [ ] **Step 3: Verify metrics.json output**

```bash
cat results/smoke-test/metrics.json
```

Expected: JSON with pa_mpjpe, pa_mpvpe, f_at_5, f_at_15, status=COMPLETED.

- [ ] **Step 4: Set Telegram chat_id**

Once the user has messaged the bot:
```bash
curl -s "https://api.telegram.org/bot8214019299:AAEbnK-wEqabNue7EHbVhfwlniLB3dtC68Y/getUpdates" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result'][-1]['message']['chat']['id'])"
```

Update `orze.yaml` with the actual chat_id.

- [ ] **Step 5: Launch orze**

```bash
orze start -c orze.yaml
```

Expected: orze starts, claims idea-001, begins training on GPU 0. Other ideas queue for remaining GPUs.

- [ ] **Step 6: Verify orze is running**

```bash
orze --admin &
cat results/status.json
```

Expected: status shows running experiments, admin panel accessible at http://localhost:8787.

- [ ] **Step 7: Commit any final adjustments**

```bash
git add -A
git commit -m "feat: integration verified, orze launched"
```
