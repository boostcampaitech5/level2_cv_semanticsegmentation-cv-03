# dataset settings
dataset_type = "XRayDataset"
data_root = "../../../data"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadXRayAnnotations"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="TransposeAnnotations"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadXRayAnnotations"),
    dict(type="TransposeAnnotations"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, split="train1", pipeline=train_pipeline
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, split="val1", pipeline=test_pipeline
    ),
)

# 형식만 필요, 값들은 사실 무의미
# Don't fix
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, split="val1", pipeline=test_pipeline
    ),
)

val_evaluator = dict(type="DiceMetric")
test_evaluator = val_evaluator
