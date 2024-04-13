"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer
import imageio


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_torus(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

############################ Torus modifications ##########################

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (1+ 0.5*torch.cos(Theta)) * torch.cos(Phi)
    y = (1+ 0.5*torch.cos(Theta)) * torch.sin(Phi)
    z = 0.5*torch.sin(Theta)

###########################################################################


    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points*4 - points.min()) / (points.max() - points.min()*2)


    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

################################# Rotation #################################

    all_renders = []
    for angle in range(0,360,10):

        R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        rend = rend*255
        all_renders.append(rend.astype(np.uint8))

    imageio.mimsave('results/torus.gif', all_renders, duration = int(1000/5), loop=0)   


############################################################################
   

if __name__ == "__main__":

    render_torus(image_size=256)


