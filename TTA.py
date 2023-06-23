import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import random
from argparse import ArgumentParser

from dataset import XRayDataset, XRayInferenceDataset
from torch.utils.data import Dataset

import albumentations as A
import wandb
import ttach as tta
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import cv2
import ttach as tta

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
# https://github.com/qubvel/ttach/blob/master/ttach/transforms.py -- 참고
test_transform = tta.Compose([tta.Resize(sizes=(1024, 1024), original_size=(2048,2048), interpolation='bilinear'), tta.HorizontalFlip()])

class CustomModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        
        if isinstance(output, dict):
            output = output["out"]
        return output

class XRayInferenceDataset_TTA(Dataset):
    def __init__(self, data_root, transforms: A = None):
        """
        Args:
            data_root   :   csv 파일 위치
        """
        self.df = pd.read_csv(os.path.join(data_root, f"test.csv"))
        self.data_root = data_root
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filenames"]
        image_name = os.path.join(image_path.split("/")[-2], image_path.split("/")[-1])

        image = cv2.imread(image_path)

        image = image / 255.0
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name

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


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def test(data_loader, classes, best_model_dir, save_dir, is_csv=True, thr=0.5):
    print("Start inference ...")
    idx2class = {i: v for i, v in enumerate(classes)}

    model = torch.load(os.path.join(best_model_dir, "best_model.pt"))["model"]
    model = CustomModel(model)
    model = tta.SegmentationTTAWrapper(model, test_transform, merge_mode='mean')
    model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.cuda()
            # outputs = model(images)["out"]
            outputs = model(images)
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{idx2class[c]}_{image_name}")

    if is_csv:
        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        df = pd.DataFrame(
            {
                "image_name": image_name,
                "class": classes,
                "rle": rles,
            }
        )

        df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
        print("CSV file creation successful")
    else:
        return rles, filename_and_class
    

def main(args):
    
    save_csv = os.path.join(args.save_csv, args.exp_name)
    save_checkpoint = os.path.join(args.save_checkpoint, args.exp_name)
    
    test_dataset = XRayInferenceDataset_TTA(args.data_root, transforms=test_transform)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    
    # Inference
    test(test_loader, CLASSES, save_checkpoint, save_csv, args.make_csv)
    
    
if __name__ == "__main__":
    parser = ArgumentParser()

    # Path
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="./predictions",
    )
    
    parser.add_argument("--exp-name", type=str, default="[test]ExpName")
    
    # Inference
    parser.add_argument("--make-csv", type=bool, default=True)

    args = parser.parse_args()

    main(args)