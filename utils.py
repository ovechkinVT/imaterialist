import numpy as np

import transforms as T
import torch



def collate_fn(batch):
    return tuple(zip(*batch))

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).transpose()

def get_box_from_mask(mask):
    assert isinstance(mask, np.ndarray)
    x_min, x_max = np.where(mask.max(axis=0)==1)[0][[0,-1]]
    y_min, y_max = np.where(mask.max(axis=1)==1)[0][[0,-1]]
    return x_min, y_min, x_max, y_max


import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":

    mask_rle = "0 2 3 1 6 3"
    mask = rle_decode(mask_rle, shape=(3,3))
    box = get_box_from_mask(mask)
    print(mask_rle, mask, box)