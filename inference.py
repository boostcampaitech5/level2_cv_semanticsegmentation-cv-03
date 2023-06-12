import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


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
    model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.cuda()
            outputs = model(images)["out"]

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
