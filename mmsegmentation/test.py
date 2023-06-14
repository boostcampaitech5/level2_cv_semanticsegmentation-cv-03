# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch

from mmseg.registry import MODELS

from mmengine.dataset import Compose
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.datasets import BaseSegDataset
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.decode_heads import ASPPHead, FCNHead, SegformerHead
from mmseg.models.utils.wrappers import resize
from mmseg.structures.seg_data_sample import SegDataSample

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner import Runner
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.structures import PixelData

from mmcv.transforms import BaseTransform

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="[test]ExpName")
    parser.add_argument(
        "--work-dir",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument("--checkpoint", help="latest.pth")
    parser.add_argument(
        "--out",
        type=str,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if "visualization" in default_hooks:
        visualization_hook = default_hooks["visualization"]
        # Turn on visualization
        visualization_hook["draw"] = True
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.show_dir:
            visulizer = cfg.visualizer
            visulizer["save_dir"] = args.show_dir
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    return cfg


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _preprare_data(cfg, imgs, model):
    for type_ in ["LoadXRayAnnotations", "TransposeAnnotations"]:
        for t in cfg.test_pipeline:
            if t.get("type") == type_:
                cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]["type"] = "LoadImageFromNDArray"

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data["inputs"].append(data_["inputs"])
        data["data_samples"].append(data_["data_samples"])

    return data, is_batch


def main():
    args = parse_args()

    config_file = f"teamconfigs/{args.config}/config.py"
    args.out = f"teamconfigs/{args.config}/predictions"
    args.work_dir = f"teamconfigs/{args.config}/checkpoints"

    for ckpt in os.listdir(args.work_dir):
        if ckpt.find("best") != -1:
            args.checkpoint = osp.join(args.work_dir, ckpt)

    if not os.path.isdir(args.out):
        os.makedirs(args.out, exist_ok=True)

    # load config
    cfg = Config.fromfile(config_file)
    cfg.work_dir = args.work_dir

    cfg.vis_backends = [dict(type="LocalVisBackend")]
    cfg.visualizer = dict(
        type="SegLocalVisualizer", vis_backends=cfg.vis_backends, name="visualizer"
    )

    # runner
    runner = Runner.from_cfg(cfg)

    # Load model
    model = MODELS.build(cfg.model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # Load Dataset
    df = pd.read_csv(osp.join(cfg.data_root, "test.csv"))

    rles = []
    filename_and_class = []
    # for idx in tqdm(range(len(df))):
    for idx in tqdm(range(2)):
        img_path = df.iloc[idx]["filenames"]
        image_name = img_path.split("/")[-1]
        img = cv2.imread(img_path)

        # prepare data
        data, is_batch = _preprare_data(cfg, img, model)

        # forward the model
        with torch.no_grad():
            results = model.test_step(data)
        output = results[0].pred_sem_seg.data

        for c, segm in enumerate(output):
            rle = encode_mask_to_rle(segm)
            rles.append(rle)
            filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    classes, image_name = zip(*[x.split("_") for x in filename_and_class])
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    df.to_csv(os.path.join(args.out, f"{args.config}.csv"), index=False)
    print("CSV file creation successful")


if __name__ == "__main__":
    main()
