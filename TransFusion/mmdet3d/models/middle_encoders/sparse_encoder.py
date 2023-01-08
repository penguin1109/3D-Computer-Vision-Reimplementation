from mmcv.runner import auto_fp16
from ..registry import MIDDLE_ENCODERS
import torch.nn as nn
import torch

@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    """Sparse Encoder
    - Used for the SECOND model for 3D object detection
    """
    def __init__(self,
                in_channels,
                sparse_shape,
                ):
        super().__init__()
    
    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """
        voxel_features: (N, c) 크기의 vector로, N개의 Voxel마다 C차원의 vector로 feature represent가 된다.
        coors: Coordinates (batch_idx, z_idx, y_idx, x_idx) -> 각각의 Voxel마다
        batch_size: 말 그대로 배치 크기
        """
        pass
