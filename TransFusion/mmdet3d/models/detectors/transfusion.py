from .. import builder
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F

@DETECTORS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    def __init__(self, **kwargs):
        super(TransFusionDetector, self).__init__()
    