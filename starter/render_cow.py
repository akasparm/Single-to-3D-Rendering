"""
Sample code to render a cow.

Usage:
    python -m starter.render_cow
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import pytorch3d.transforms as transform
import numpy as np

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
import imageio


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=torch.device("cpu"),
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    all_renders = []

    for view_angle in range(0,361, 5):
        # Compute the rotation matrix for the current view angle
        angle = torch.tensor([view_angle], dtype=torch.float32)
        cos_angle = torch.cos(angle * (2 * np.pi / 360))  # Convert degrees to radians
        sin_angle = torch.sin(angle * (2 * np.pi / 360))
        
        R = torch.tensor([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=torch.float32, device=device)

        # Set the camera's rotation matrix for the current view
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.unsqueeze(0), T=torch.tensor([[0, 0, 3]], device=device), fov=60, device=device
        )

        # Render the cow from the current view
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend_uint8 = (rend * 255).astype(np.uint8)
        all_renders.append(rend_uint8)
      
    plt.imsave('results/render_cow.jpg', rend)
    imageio.mimsave('results/rotate_cow.gif', all_renders, duration=int(1000/15), loop = 0)



if __name__ == "__main__":
  render_cow()
