import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from ..builder import BACKBONES

@BACKBONES.register_module()
class SECOND(nn.Module):
    def __init__(self,
                in_channels=256,
                out_channels=[128, 256],
                layer_nums=[5,5],
                layer_strides=[1,2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=1e-2),
                conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels] + out_channels[:-1]


    def forward(self, x):
        pass