from tqdm import tqdm
import json
import sys
import math

import torch.utils.data
import numpy as np

from utils import get_transform, collate_fn, warmup_lr_scheduler
from dataset import IMaterialistDataset
from model import get_instance_segmentation_model
import plotting


from tensorboardX import SummaryWriter


TRAIN_PATH = "/data/kaggle-fashion/train"
TEST_PATH = "/data/kaggle-fashion/test"
LABEL_PATH = "../imgs_label_info.json"


# label description
with open("/data/kaggle-fashion/label_descriptions.json") as f:
    desc = json.load(f)
class_names =  { categ["id"]:categ["name"] for categ in desc["categories"]}
del desc



def train_one_epoch(model, optimizer, data_loader, val_data, device, epoch, log_writer, class_names):
    # https://github.com/pytorch/vision.git
    model.train()
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = \
    print('Epoch: [{}]'.format(epoch))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    for batch_i, (images, targets) in tqdm(enumerate(data_loader)): #, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = utils.reduce_dict(loss_dict)
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        losses_reduced = sum(loss for loss in loss_dict.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced, targets[0]["image_id"])
            # sys.exit(1)
            continue


        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_writer.add_scalar("total loss", losses_reduced.item(), batch_i)
        for loss_name,loss_value in loss_dict.items():
            log_writer.add_scalar(loss_name, loss_value.item(), batch_i)



        # save prediction example
        if batch_i % 100 == 0:
            ind = np.random.choice(len(data_loader_val))
            img, _ = val_data[ind]
            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])[0]

            prediction = plotting.prepare_prediction_for_ploting(prediction)
            prediction_img = plotting.plot_img(np.transpose(img, (1,2,0)), prediction, class_names)
            log_writer.add_image('prediction_img', np.transpose(prediction_img, (2, 0, 1)), batch_i)
            model.train()


if __name__ == "__main__":

    import torch
    from datetime import datetime

    log_writer = SummaryWriter(comment='_initial_run')

    print("[{0}] Loading data: START ".format(datetime.now().isoformat(' ', 'seconds')))

    train_data = IMaterialistDataset(TRAIN_PATH, LABEL_PATH, transforms=get_transform(train=True))
    # test_data = IMaterialistDataset(TEST_PATH, transforms=get_transform(train=False))


    torch.manual_seed(432)
    indices = torch.randperm(len(train_data)).tolist()

    val_size = 5000
    val_data = torch.utils.data.Subset(train_data, indices[-val_size:])
    train_data = torch.utils.data.Subset(train_data, indices[:-val_size])

    data_loader_train = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    print("[{0}] Loading data: SUCCESS".format(datetime.now().isoformat(' ', 'seconds')))

    num_classes = len(class_names)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load("../../imaterialist/model_firts_weigths.pytorch"))
    model = model.to(device)

    print("[{0}] Loading model: SUCCESS".format(datetime.now().isoformat(' ', 'seconds')))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, val_data, device, epoch, log_writer, class_names)
        lr_scheduler.step()



    torch.save(model.state_dict(), "model_weigths_{}.pytorch".format(datetime.now().isoformat(' ', 'seconds')))
