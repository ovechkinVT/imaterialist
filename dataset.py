import os
import json
import numpy as np

from PIL import Image

from utils import rle_decode, get_box_from_mask


import torch
from torch.utils.data import Dataset

class IMaterialistDataset(Dataset):
    def __init__(self, data_path, label_path=None, transforms=None):

        self.transforms = transforms

        # images name list
        self.data_path = data_path
        self.imgs = sorted(os.listdir(data_path))

        # mask info
        if label_path is not None:
            self.label_path = label_path
            with open(label_path) as f:
                self.label_info = json.load(f)
        else:
            self.label_info = None

    def __getitem__(self, idx):

        # load image and labels
        img_path = os.path.join(self.data_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.label_info is not None:
            img_height, img_width = self.label_info[idx]["Height"], self.label_info[idx]["Width"]
            masks = self.label_info[idx]["masks"]
            masks = [rle_decode(mask, shape=(img_height, img_width)) for mask in masks]
            labels = [int(i.split("_")[0]) for i in self.label_info[idx]["ClassIds"]]

            # create boxes
            boxes = [get_box_from_mask(maks) for maks in masks]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # transform masks
            masks = np.concatenate([mask[None, ...] for mask in masks])
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            # transofrm labels
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["image_id"] = torch.tensor([idx])
            # target["area"] = area

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        else:
            target = {}
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    from utils import get_transform

    path = "/data/kaggle-fashion/train"
    label_path = "../imgs_label_info.json"
    dataset = IMaterialistDataset(path, label_path, transforms=get_transform(train=True))
    img, target = dataset[0]
    print(f"Image shape: {img.shape}, label info keys: {list(target.keys())}, labels ={target['labels']} ")