import numpy as np

import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import PIL
import matplotlib.pyplot as plt

from io import BytesIO

def prepare_prediction_for_ploting(prediction):
    for k, v in prediction.items():
        prediction[k] = v.to("cpu").numpy()

    score_filter = np.where(prediction["scores"] > 0.5)

    for k, v in prediction.items():
        prediction[k] = v[score_filter]

    return prediction

def random_color_map(colors):
    color = np.random.choice(colors)
    # del colors[colors.index(color)]
    cmap = LinearSegmentedColormap.from_list(color, [color]*2, N=2)
    return cmap, colors


def plot_img(img, prediction, class_names):
    _, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img, alpha=1)
    colors = ["red", "chartreuse", "deepskyblue", "royalblue", "mediumvioletred", "c", "deeppink", "lime"]

    for i in range(len(prediction["labels"])):
        cmap, colors = random_color_map(colors)

        mask = (np.squeeze(prediction["masks"][i]) > 0.5).astype(float)
        mask[mask == 0] = np.nan
        ax.imshow(mask, alpha=0.1, cmap=cmap, )

        box = prediction["boxes"][i]
        score = prediction["scores"][i]
        label = prediction["labels"][i]
        rect = patches.Rectangle((box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]), linewidth=1, edgecolor=cmap(0),
                                 facecolor='none')
        ax.add_patch(rect)

        ax.text(x=box[0], y=box[1], s="{} ({:.2f})".format(class_names[label], score),
                fontsize=15, color="black")

    buffer_ = BytesIO()
    plt.savefig(buffer_, format="png", bbox_inches='tight', pad_inches=0)
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    ar = np.asarray(image)

    return ar