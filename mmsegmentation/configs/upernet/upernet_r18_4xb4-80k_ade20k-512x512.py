_base_ = [
    "../_base_/models/upernet_r50.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
model = dict(
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150),
)
