_base_ = "./pspnet_r50-d8_4xb4-40k_pascal-context-480x480.py"
model = dict(pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101))
