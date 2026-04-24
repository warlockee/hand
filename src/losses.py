"""Multi-task loss functions for hand pose estimation."""
import torch
import torch.nn as nn


def joint_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, gt)


def vertex_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(pred, gt)


def project_3d_to_2d(joints_3d: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    s = camera[:, 0:1].unsqueeze(-1)
    tx = camera[:, 1:2].unsqueeze(-1)
    ty = camera[:, 2:3].unsqueeze(-1)
    x = s * joints_3d[:, :, 0:1] + tx
    y = s * joints_3d[:, :, 1:2] + ty
    return torch.cat([x, y], dim=-1)


def reprojection_loss(
    joints_3d: torch.Tensor, camera: torch.Tensor, gt_2d: torch.Tensor
) -> torch.Tensor:
    proj_2d = project_3d_to_2d(joints_3d, camera)
    return nn.functional.l1_loss(proj_2d, gt_2d)


def edge_length_loss(
    pred_verts: torch.Tensor, gt_verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], dim=0)
    pred_edge_vec = pred_verts[:, edges[:, 0]] - pred_verts[:, edges[:, 1]]
    gt_edge_vec = gt_verts[:, edges[:, 0]] - gt_verts[:, edges[:, 1]]
    return nn.functional.l1_loss(pred_edge_vec.norm(dim=-1), gt_edge_vec.norm(dim=-1))


class HandPoseLoss(nn.Module):
    def __init__(
        self, faces: torch.Tensor,
        w_joint: float = 5.0, w_vertex: float = 5.0, w_reproj: float = 5.0,
        w_edge: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("faces", faces)
        self.w_joint = w_joint
        self.w_vertex = w_vertex
        self.w_reproj = w_reproj
        self.w_edge = w_edge

    def forward(
        self, pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_joint = joint_loss(pred["joints_3d"], gt["joints_3d"])
        l_vert = vertex_loss(pred["vertices"], gt["vertices"])
        l_reproj = reprojection_loss(pred["joints_3d"], pred["camera"], gt["joints_2d"])
        l_edge = edge_length_loss(pred["vertices"], gt["vertices"], self.faces)

        total = (
            self.w_joint * l_joint + self.w_vertex * l_vert +
            self.w_reproj * l_reproj + self.w_edge * l_edge
        )
        components = {
            "joint": l_joint.item(), "vertex": l_vert.item(),
            "reproj": l_reproj.item(), "edge": l_edge.item(),
        }
        return total, components
