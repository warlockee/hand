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
    uv = (K @ xyz.T).T
    return uv[:, :2] / (uv[:, 2:3] + 1e-8)


class FreiHandDataset(Dataset):
    """FreiHand has 32,560 unique hands × 4 background augmentations = 130,240 images.
    Annotations are for the 32,560 unique hands. Image i maps to annotation i % 32560.
    """

    def __init__(
        self, data_dir: str | Path, split: str = "train",
        img_size: int = 224, augment: bool = False, scale_to_mm: float = 1000.0,
    ):
        self.data_dir = Path(data_dir)
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
        self.n_annot = len(self.K)

        mano_path = self.data_dir / f"{prefix}_mano.json"
        if mano_path.exists():
            raw = json.load(open(mano_path))
            # FreiHand format: each entry is [[61 floats]] — pose(48) + shape(10) + trans(3)
            if isinstance(raw[0], list) and isinstance(raw[0][0], list):
                arr = np.array([r[0] for r in raw], dtype=np.float32)
            elif isinstance(raw[0], list):
                arr = np.array(raw, dtype=np.float32)
            elif isinstance(raw[0], dict):
                arr = np.column_stack([
                    np.array([m["pose"] for m in raw], dtype=np.float32),
                    np.array([m["shape"] for m in raw], dtype=np.float32),
                ])
            else:
                arr = np.array(raw, dtype=np.float32)
            self.mano_pose = arr[:, :48]
            self.mano_shape = arr[:, 48:58]
        else:
            self.mano_pose = np.zeros((self.n_annot, 48), dtype=np.float32)
            self.mano_shape = np.zeros((self.n_annot, 10), dtype=np.float32)

        self.transform = get_augmentation(img_size) if augment else get_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        img_tensor = self.transform(img)

        annot_idx = idx % self.n_annot

        xyz = self.xyz[annot_idx]
        K = self.K[annot_idx]
        joints_2d = project_3d_to_2d_np(xyz, K)
        orig_size = max(img.size)
        joints_2d = joints_2d / orig_size

        return {
            "image": img_tensor,
            "joints_3d": torch.from_numpy(xyz * self.scale_to_mm).float(),
            "vertices": torch.from_numpy(self.verts[annot_idx] * self.scale_to_mm).float(),
            "joints_2d": torch.from_numpy(joints_2d).float(),
            "mano_pose": torch.from_numpy(self.mano_pose[annot_idx]).float(),
            "mano_shape": torch.from_numpy(self.mano_shape[annot_idx]).float(),
            "K": torch.from_numpy(K).float(),
        }
