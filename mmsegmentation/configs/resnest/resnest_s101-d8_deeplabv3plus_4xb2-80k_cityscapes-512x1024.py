_base_ = (
    "../deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py"  # noqa
)
model = dict(
    pretrained="open-mmlab://resnest101",
    backbone=dict(
        type="ResNeSt",
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
    ),
)
