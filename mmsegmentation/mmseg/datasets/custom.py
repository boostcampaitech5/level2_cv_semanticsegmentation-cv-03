import os

import pandas as pd

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, data_root, split, **kwargs):
        self.df = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
        self.data_root = data_root
        self.is_train = True if "train" in split else False
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        for idx in range(len(self.df)):
            data_info = dict(
                img_path=self.df.iloc[idx]["filenames"],
                seg_map_path=self.df.iloc[idx]["labelnames"],
            )
            data_list.append(data_info)

        return data_list
