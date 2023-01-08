import torch

from .. import builder
from ..registry import VOXEL_ENCODERS

""" voxel_encoder
- voxel마다 feature vector을 뽑아내는 방법이다.
- 이때 원래 point cloud가 있는 공간에 voxel grid를 만들어 주는 것이기 때문에 각각의 voxel에 들어있는 point들의 feature vector을 평균을 내어 주는 방법으로 
voxel encoding을 진행한다.
"""

