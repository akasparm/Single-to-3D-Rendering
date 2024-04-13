import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
from starter.utils import get_device, get_mesh_renderer
import imageio
import numpy as np

def render_tetrahedron(device=None):
    if device is None:
        device = get_device()

    # Define the vertices and faces of a tetrahedron.
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5],  # Vertex 1
        [0.5, -0.5, -0.5],   # Vertex 2
        [0.5, 0.5, -0.5],    # Vertex 3
        [-0.5, 0.5, -0.5],   # Vertex 4
        [-0.5, -0.5, 0.5],   # Vertex 5
        [0.5, -0.5, 0.5],    # Vertex 6
        [0.5, 0.5, 0.5],     # Vertex 7
        [-0.5, 0.5, 0.5],    # Vertex 8
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2],  # Face 1/1 (front face)
        [0, 2, 3],  # Face 2/1 (front face)
        [4, 5, 6],  # Face 1/2 (back face)
        [4, 6, 7],  # Face 2/2 (back face)
        [0, 1, 5],  # Face 1/3 (bottom face)
        [0, 5, 4],  # Face 2/3 (bottom face)
        [2, 3, 7],  # Face 1/4 (top face)
        [2, 7, 6],  # Face 2/4 (top face)
        [0, 3, 7],  # Face 1/5 (left face)
        [0, 7, 4],  # Face 2/5 (left face)
        [1, 2, 6],  # Face 1/6 (right face)
        [1, 6, 5],  # Face 2/6 (right face)
    ], dtype=torch.int64)

    # Make sure vertices have shape (1, N_v, 3) and faces have shape (1, N_f, 3).
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # Create a single-color texture.
    textures = torch.ones_like(vertices) * torch.tensor([0.7, 0.7, 1])  # Blue color

    # Create a PyTorch3D Mesh object.
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=256)

    ############ ---------- Rotation ------------ ##################

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

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)


        # Render the cow from the current view
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend_uint8 = (rend * 255).astype(np.uint8)
        all_renders.append(rend_uint8)
      
    imageio.mimsave('results/cube.gif', all_renders, duration=int(1000/15), loop = 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    render_tetrahedron()
    print("Cube generated!")

