# optimizer
optimizer = dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    ),
]
# training schedule for 20k
# Train data: 640
# Valid data: 160
# Batch size: 8
# 1-epoch = 640 / 8 = 80-iterations
max_iters = 20000
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters//20)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    # Log를 몇 개마다 남길지
    logger=dict(type="LoggerHook", interval=10, log_metric_by_epoch=False),
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
