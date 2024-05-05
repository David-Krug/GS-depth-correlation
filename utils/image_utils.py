#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from matplotlib.colors import Normalize
import torch
import numpy as np
from matplotlib import cm
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_colormap(img, colormap='jet'):
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = img.squeeze()
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.squeeze(img)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = cm.get_cmap(colormap)(img)
    img = transforms.ToTensor()(img)
    return img[:3,:,:][None]

def create_colorbar_vis(img, path, name='image', colormap='magma', original_grayscale=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = img.squeeze()
    #original_grayscale = img
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.squeeze(img)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = cm.get_cmap(colormap)(img)

    # Make plot
    # fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    h, w = img.shape[:2]
    fig, axs = plt.subplots(figsize=(w / 100, h / 100), dpi=100)

        # Plot grayscale colorbar
    if original_grayscale is not None:
        if isinstance(original_grayscale, torch.Tensor):
            original_grayscale = original_grayscale.cpu().numpy()
        original_grayscale = original_grayscale.squeeze()
        min_val = np.min(original_grayscale)
        max_val = np.max(original_grayscale)
        gmap = axs.imshow(original_grayscale, cmap="gray", vmin=255-max_val, vmax=255-min_val)
        axs.set_title('Original adjusted grayscale image')

        # Plot image

        axs.imshow(img)
        axs.set_title(name)
        axs.axis('off')

        cbar = plt.colorbar(gmap,ax=axs, shrink=0.5)
        cbar.set_label('Original grayscale values mapped to full range')

        # Add colorbar mapping grayscale values to the specified colormap
        norm = Normalize(vmin=0, vmax=255)  # Normalize the color range
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Set an empty array

        # Add colorbar
        cbar = plt.colorbar(sm, ax=axs, shrink=0.5)
        cbar.set_label('Values mapped to colormap ' + colormap)
    else:
    
        # Plot image

        axs.imshow(img)
        axs.set_title(name)
        axs.axis('off')

        # Add colorbar mapping grayscale values to the specified colormap
        norm = Normalize(vmin=0, vmax=255)  # Normalize the color range
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Set an empty array

        gm = plt.cm.ScalarMappable(cmap='gray', norm=norm)
        gm.set_array([])  # Set an empty array

        # Add colorbar
        cbar = plt.colorbar(sm, ax=axs, shrink=0.5)
        cbar.set_label('Values mapped to colormap ' + colormap)

        cbar = plt.colorbar(gm, ax=axs, shrink=0.5)
        cbar.set_label('Grayscale values')

    plt.tight_layout()

    # Convert plot to tensor
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
