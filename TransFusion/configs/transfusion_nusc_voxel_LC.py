point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 5.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2] ## 3D Feature Extraction에서 사용하는 VoxelNet의 voxel 크기
out_size_factor = 8
img_scale = (800, 448) ## 2배로 원래 이미지를 줄임
num_views = 6 ## BACK, BACK_LEFT, BACK_RIGHT, FRONT, FRONT_LEFT, FRONT_RIGHT, LIDAR_TOP (Bird's Eye View)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)




model = dict(
    type='TransFusionDetector',
    freeze_img=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1025, 2048],
        out_channels=256,
        num_outs=5,
    ),
    ## Point Cloud를 Voxel Grid로 바꾸어주는 작업이다.
    pts_voxel_layer=dict(
        max_num_points=10, ## 하나의 voxel에 들어가는 최대 point의 개수
        voxel_size=voxel_size, ## voxel 하나의 크기
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range
    ),
    ## 하나의 Voxel에 들어있는 point들의 feature에 대한 평균이 곧 voxel feature vetor이다.
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', ## Voxel Feature Encoding
        num_features=5
    ),
    ## 이제 바뀐 Voxel Encoded Vector을 3D Backbone에 넣어준다.
    # Sparsely Convolutional Layer (=SECOND)
    pts_middle_encoder=dict(
       type='SparseEncoder',
       in_channels=5,
       sparse_shape=[41, 1440, 1440],
       output_channels = 128,
       order = (
        'conv', 'norm', 'act'
       ),
       encoder_channels = ((16, 16, 32), (32, 32, 64), (64, 64,128), (128, 128)),
       encoder_paddings=((0,0,1), (0,0,1),(0,0,[0,1,1]), (0,0)),
       block_type = 'basicblock'
    ),
    ## Define a backbone
    pts_backbone=dict(
        type='SECOND', 
        in_channels=256,
        out_channels=[128, 256],
        layer_num=[5,5],
        layer_strides=[1,2], ## 한번 이미지를 줄이게 됨
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.1),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    ## Define a neck 
    pts_neck=dict(
        type-'SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1,2], ## deconvolution으로 resize
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    ## One-Stage Bounding box detection head
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True,
        num_views=num_views,
    )


)