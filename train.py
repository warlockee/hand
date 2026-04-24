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
from torch.amp import GradScaler, autocast

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from src.model import HandPoseModel
from src.losses import HandPoseLoss
from src.dataset import FreiHandDataset
from src.metrics import compute_all_metrics


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--idea-id", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--ideas-md", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, max_grad_norm):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt = {k: v.to(device) for k, v in batch.items() if k not in ("image", "K")}

        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.float16):
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
    all_pred_j, all_gt_j, all_pred_v, all_gt_v = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        with autocast("cuda", dtype=torch.float16):
            pred = model(images)

        all_pred_j.append(pred["joints_3d"].float().cpu())
        all_gt_j.append(batch["joints_3d"])
        all_pred_v.append(pred["vertices"].float().cpu())
        all_gt_v.append(batch["vertices"])

    return compute_all_metrics(
        torch.cat(all_pred_j), torch.cat(all_gt_j),
        torch.cat(all_pred_v), torch.cat(all_gt_v),
    )


def main():
    args = parse_args()

    result_dir = Path(args.results_dir) / args.idea_id
    result_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    idea_cfg_path = result_dir / "idea_config.yaml"
    if idea_cfg_path.exists():
        idea_cfg = yaml.safe_load(idea_cfg_path.read_text()) or {}
        cfg = deep_merge(base_cfg, idea_cfg)
    else:
        cfg = base_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})
    lcfg = cfg.get("loss", {})

    model = HandPoseModel(
        backbone_name=mcfg.get("backbone", "vit_huge_patch14_dinov2.lvd142m"),
        img_size=mcfg.get("img_size", 224),
    ).to(device)

    data_dir = dcfg.get("data_dir", "data/freihand")
    img_size = dcfg.get("img_size", mcfg.get("img_size", 224))
    train_ds = FreiHandDataset(data_dir, split="train", img_size=img_size, augment=dcfg.get("augment", True))
    val_ds = FreiHandDataset(data_dir, split="train", img_size=img_size, augment=False)

    bs = tcfg.get("batch_size", 64)
    nw = tcfg.get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    faces = model.mano.th_faces.long()
    loss_fn = HandPoseLoss(
        faces=faces,
        w_joint=lcfg.get("w_joint", 5.0), w_vertex=lcfg.get("w_vertex", 5.0),
        w_reproj=lcfg.get("w_reproj", 5.0),
        w_edge=lcfg.get("w_edge", 1.0),
    ).to(device)

    scaler = GradScaler("cuda")
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    best_metrics = None
    best_epoch = 0

    # Phase 1: frozen backbone
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

    # Phase 2: end-to-end
    model.unfreeze_backbone()
    p2_epochs = tcfg.get("phase2_epochs", 80)
    p2_lr = tcfg.get("phase2_lr", 1e-4)
    min_lr = tcfg.get("min_lr", 1e-6)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p2_lr, weight_decay=tcfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=min_lr)

    print(f"Phase 2: {p2_epochs} epochs, lr={p2_lr}, end-to-end")
    for epoch in range(p2_epochs):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, max_grad_norm)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == p2_epochs - 1:
            metrics = evaluate(model, val_loader, device)
            print(f"  Epoch {p1_epochs + epoch + 1} loss={loss:.4f} PA-MPJPE={metrics['pa_mpjpe']:.2f}mm")

            if best_metrics is None or metrics["pa_mpjpe"] < best_metrics["pa_mpjpe"]:
                best_metrics = metrics
                best_epoch = p1_epochs + epoch + 1
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                }, result_dir / "best_model.pt")
        else:
            print(f"  Epoch {p1_epochs + epoch + 1} loss={loss:.4f}")

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


if __name__ == "__main__":
    main()
