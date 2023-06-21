default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# wandb 설정
exp_name = "[test]ExpName"
init_kwargs = dict(
    name=exp_name, project="Xray-Segmentation", entity="ganisokay"
)
log_code_name = exp_name.replace("[", "").replace("]", "-")
wandb_kwargs = dict(
    init_kwargs=init_kwargs, log_code_name=log_code_name
)
vis_backends = [dict(type="LocalVisBackend"), dict(type="WandbVisBackend", **wandb_kwargs)]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")

log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

tta_model = dict(type="SegTTAModel")
