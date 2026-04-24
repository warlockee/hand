"""Hand mesh regression head — direct vertex + joint regression.

Uses a learned linear regressor from mesh vertices to joints (J-regressor)
instead of requiring the proprietary MANO pickle files. The model learns
to output 778 vertices + 21 joints directly.
"""
import torch
import torch.nn as nn


class MANOHead(nn.Module):
    """Direct mesh + joint regression (no MANO pickle dependency).

    Instead of MANO params → vertices → joints, we regress vertices directly
    and use a learned J-regressor to get joints from vertices.
    """

    N_VERTS = 778
    N_JOINTS = 21
    N_FACES = 1538

    def __init__(self, feat_dim: int = 1024):
        super().__init__()
        self.vert_regressor = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.N_VERTS * 3),
        )
        # Learned joint regressor: maps 778 vertices → 21 joints
        self.j_regressor = nn.Linear(self.N_VERTS, self.N_JOINTS, bias=False)

        self._init_weights()
        self._build_faces()

    def _init_weights(self):
        for m in self.vert_regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.j_regressor.weight)

    def _build_faces(self):
        # Placeholder faces for edge regularization — approximate hand topology
        # Real faces would come from MANO, but we generate a dummy triangulation
        # that's sufficient for edge length regularization
        faces = []
        for i in range(0, min(self.N_VERTS - 2, self.N_FACES), 1):
            faces.append([i, i + 1, i + 2])
        self.register_buffer("th_faces", torch.tensor(faces, dtype=torch.long))

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, feat_dim) backbone features

        Returns:
            vertices: (B, 778, 3)
            joints: (B, 21, 3)
        """
        B = features.shape[0]
        verts = self.vert_regressor(features).view(B, self.N_VERTS, 3)

        # J-regressor: for each of XYZ, joints = weight_matrix @ verts
        # j_regressor.weight: (21, 778)
        joints = torch.einsum("jv,bvc->bjc", self.j_regressor.weight, verts)

        return verts, joints
