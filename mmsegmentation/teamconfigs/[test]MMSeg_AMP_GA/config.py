_base_ = ["dataset.py", "default_runtime.py", "schedule.py", "segformer_mit-b0.py"]

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoderWithoutArgmax",
    init_cfg=dict(
        type="Pretrained",
        # load ADE20k pretrained EncoderDecoder from mmsegmentation
        checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth",
    ),
    data_preprocessor=data_preprocessor,
    backbone=dict(embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type="SegformerHeadWithoutAccuracy",
        num_classes=29,
        in_channels=[64, 128, 320, 512],
        # Loss 설정
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)
