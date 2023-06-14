_base_ = ["dataset.py", "default_runtime.py", "schedule.py", "segformer_mit-b0.py"]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=4, val_interval=2)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    # Log를 몇 개마다 남길지 -> wandb와 연결됨(log가 interval마다 프린트되면 wandb에서도 interval마다 log가 남음)
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    # 기본적으로 max_keep_ckpts만큼의 checkpoint가 저장되고 
    # best score를 가지는 checkpoint 하나, 마지막 checkpoint 하나가 저장
    # 총 max_keep_ckpts + 1 + 1개의 checkpoint가 저장
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1,
        max_keep_ckpts=3,
        save_best="mDice",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoderWithoutArgmax",
    init_cfg=dict(
        type="Pretrained",
        # load ADE20k pretrained EncoderDecoder from mmsegmentation
        checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type="SegformerHeadWithoutAccuracy",
        num_classes=29,
        # Loss 설정
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)
