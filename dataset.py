from torch.utils.data import Dataset
import os
import albumentations as A
import cv2
import numpy as np
import json
import torch
import pandas as pd

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

CLASS2IDX = {v: i for i, v in enumerate(CLASSES)}


class XRayDataset(Dataset):
    def __init__(self, data_root, transforms: A = None, split: str = None):
        """
        Args:
            data_root   :   csv 파일 위치
            split       :   train1, train2, ..., val4, val5
        """
        self.df = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
        self.data_root = data_root
        self.is_train = True if "train" in split else False
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filenames"]
        label_path = row["labelnames"]

        image = cv2.imread(image_path)
        image = image / 255.0

        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            class_idx = CLASS2IDX[ann["label"]]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_idx] = class_label

        if self.transforms is not None:
            inputs = (
                {"image": image, "mask": label} if self.is_train else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
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

        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result["image"]

        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name
