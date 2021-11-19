import numpy as np
from skimage import morphology
import os
import matplotlib.pyplot as plt
import matplotlib
from skimage.segmentation import mark_boundaries
import config
from torchvision.models import wide_resnet50_2, resnet18
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet


def build_model():

    if config.ARCH == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        # The following dictionary contains the model layers and the sqrt of output filters
        layers_dictionary = OrderedDict(
            [('layer1', (model.layer1[-1], 64)), ('layer2', (model.layer2[-1], 128)), ('layer3', (model.layer3[-1],  256)),
             ('layer4', (model.layer4[-1], 512))])

    elif config.ARCH == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        # The following dictionary contains the model layers and the sqrt of output filters
        layers_dictionary = OrderedDict(
            [('layer1', (model.layer1[-1], 256)), ('layer2', (model.layer2[-1], 512)), ('layer3', (model.layer3[-1], 1024))])

    elif config.ARCH == 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        layers_dictionary = OrderedDict(
            [('layer1', (model._blocks[13], 128)),
             ('layer2', (model._blocks[19], 128)),
             ('layer3', (model._blocks[26], 176))])

    elif config.ARCH == 'efficientnet-b6':
        model = EfficientNet.from_pretrained('efficientnet-b6')
        layers_dictionary = OrderedDict(
            [('layer1', (model._blocks[15], 144)),
             ('layer2', (model._blocks[22], 144)),
             ('layer3', (model._blocks[30], 200))])

    elif config.ARCH == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        layers_dictionary = OrderedDict(
            [('layer1', (model._blocks[18], 160)),
             ('layer2', (model._blocks[28], 220)),
             ('layer3', (model._blocks[37], 220))])

    return model, layers_dictionary


def plot_fig(test_img, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        #ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax = ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, 'image_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x