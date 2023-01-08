import torch.nn as nn

from ..builder import NECKS

@NECKS.register
class SECONDFPN(nn.Module):
    def __init__(self,
                in_channels=[128, 256],
                out_channels=[256, 256],
                upsample_strides=[1,2],
                norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True):
        super(SECONDFPN, self).__init__()
    
    def forward(self, x):
        pass