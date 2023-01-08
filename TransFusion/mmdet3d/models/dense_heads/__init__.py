""" dense_heads
- 만약에 PartA2 RoI Head와 같이 second stage에 사용된다면 roi_heads/bbox_heads에 넣어야 한다.
- 하지만 TransFusionHead는 one-stage head이기 때문에 dense_heads 폴더에 넣어주게 된다.
- 보통 head를 implement하기 위해서는 loss와 get_targets를 정의해 주는 것이 필요하다.
"""
from .transfusion_head import TransFusionHead