"""Procrustes alignment and FreiHand evaluation metrics."""
import numpy as np
import torch
from scipy.spatial.distance import cdist


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    aligned = (s_centered @ R.T) * norm_t + mu_t
    return aligned


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    errors = np.zeros(B, dtype=np.float64)
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    for i in range(B):
        aligned = procrustes_align(pred_np[i], gt_np[i])
        errors[i] = np.linalg.norm(aligned - gt_np[i], axis=1).mean()

    return torch.from_numpy(errors).float()


def pa_mpvpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return pa_mpjpe(pred, gt)


def f_score(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
