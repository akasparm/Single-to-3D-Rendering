"""
Sample code to render various representations.

Usage:
    python -m starter.flower_vase
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
from numpy.ma import remainder
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image
import imageio


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_flower_vase(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    data = load_rgbd_data()
    # print(data)
    
    img1, depth1, mask1, camera1 = torch.Tensor(data['rgb1']), torch.Tensor(data['depth1']), torch.Tensor(data['mask1']), data['cameras1']
    img2, depth2, mask2, camera2 = torch.Tensor(data['rgb2']), torch.Tensor(data['depth2']), torch.Tensor(data['mask2']), data['cameras2']

    verts1, rgb1 = unproject_depth_image(img1, mask1, depth1, camera1)
    verts2, rgb2 = unproject_depth_image(img2, mask2, depth2, camera2)
    verts_combined, rgb_combined = torch.cat((verts1, verts2), 0), torch.cat((rgb1, rgb2), 0)

    verts1, rgb1 = verts1.unsqueeze(0), rgb1.unsqueeze(0)
    verts2, rgb2 = verts2.unsqueeze(0), rgb2.unsqueeze(0)
    verts_combined, rgb_combined = verts_combined.unsqueeze(0), rgb_combined.unsqueeze(0)

    all_renders = []



    ################################ FLOWER VASE 1 ################################
    for angle in range(0, 361, 50):

      R, T = pytorch3d.renderer.look_at_view_transform(7, 10, angle)
      R = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, np.pi]), "XYZ") @ R
      camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
      point_cloud = pytorch3d.structures.Pointclouds(points=verts1, features=rgb1)
      rend = renderer(point_cloud, cameras=camera)
      rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
      # plt.imsave("results/flower_vase1.jpg", rend)
      rend_uint8 = (rend * 255).astype(np.uint8)
      all_renders.append(rend_uint8)

    imageio.mimsave('results/flower_vase1.gif', all_renders, duration = int(1000/10), loop=0)
    print("Flower_vase_1.gif generated")


    ################################ FLOWER VASE 2 ################################
    for angle in range(0, 361, 50):

      R, T = pytorch3d.renderer.look_at_view_transform(7, 10, angle)
      R = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, np.pi]), "XYZ") @ R
      camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
      point_cloud = pytorch3d.structures.Pointclouds(points=verts2, features=rgb2)
      rend = renderer(point_cloud, cameras=camera)
      rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
      # plt.imsave("results/flower_vase2.jpg", rend)
      rend_uint8 = (rend * 255).astype(np.uint8)
      all_renders.append(rend_uint8)

    imageio.mimsave('results/flower_vase2.gif', all_renders, duration = int(1000/10), loop=0)
    print("Flower_vase_2.gif generated")

    ############################# FLOWER VASE COMBINED ############################
    for angle in range(0, 361, 50):

      R, T = pytorch3d.renderer.look_at_view_transform(7, 10, angle)
      R = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, np.pi]), "XYZ") @ R
      camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
      point_cloud = pytorch3d.structures.Pointclouds(points=verts_combined, features=rgb_combined)
      rend = renderer(point_cloud, cameras=camera)
      rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)      
      # plt.imsave("results/flower_vase_combined.jpg", rend)
      rend_uint8 = (rend * 255).astype(np.uint8)
      all_renders.append(rend_uint8)
    
    imageio.mimsave('results/flower_vase_combined.gif', all_renders, duration = int(1000/10), loop=0)
    print("Flower_vase_combined.gif generated")



def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=50, device=None):
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

    x = 1+ 0.5*torch.cos(Theta) * torch.cos(Phi)
    y = 1+ 0.5*torch.cos(Theta) * torch.sin(Phi)
    z = 0.5*torch.sin(Theta)

###########################################################################


    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())


    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

################################# Rotation #################################
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    all_renders = []
    for view_angle in range(0,361,20):

        angle = torch.tensor([view_angle], dtype=torch.float32)
        cos_angle = torch.cos(angle * (2 * torch.pi / 360))  # Convert degrees to radians
        sin_angle = torch.sin(angle * (2 * torch.pi / 360))

        R = torch.tensor([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=torch.float32, device=device)    

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.unsqueeze(0), T=torch.tensor([[0, 0, 3]], device=device), device=device
        )

        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        all_renders.append(rend.astype(np.uint8))
############################################################################
    
    imageio.mimsave('results/torus.gif', all_renders, duration = int(1000/50))


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="flower_vase",
        choices=["point_cloud", "parametric", "implicit", "flower_vase"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "flower_vase":
        image = render_flower_vase(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    # plt.imsave(args.output_path, image)

