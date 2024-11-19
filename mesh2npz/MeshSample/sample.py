import argparse
import trimesh
import numpy as np
from MeshSample.inside_mesh import inside_mesh
from pykdtree.kdtree import KDTree
import torch
import mcubes

def normalize_mesh(mesh):
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)

def compute_volume_points(intersector, count, max_batch_size=1000000):
    coordinates = np.random.rand(count, 3) * 2 - 1
    
    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    max_batch_size = min(count, max_batch_size)
    
    while head < coordinates.shape[0]:
        occupancies[head:head + max_batch_size] = intersector.query(coordinates[head:head + max_batch_size]).astype(int).reshape(-1, 1)
        head += max_batch_size

    return np.concatenate([coordinates, occupancies], -1)

def compute_near_surface_points(mesh, intersector, count, epsilon, max_batch_size=1000000):
    coordinates = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(*(count, 3)) * epsilon

    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    max_batch_size = min(count, max_batch_size)
    
    while head < coordinates.shape[0]:
        occupancies[head:head + max_batch_size] = intersector.query(coordinates[head:head + max_batch_size]).astype(int).reshape(-1, 1)
        head += max_batch_size

    return np.concatenate([coordinates, occupancies], -1)

def compute_obj(mesh, intersector, max_batch_size=1000000, res=1024):
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coordinates = coords.reshape(res * res * res, 3).numpy()

    occupancies = np.zeros((res * res * res, 1), dtype=int)
    head = 0
    
    while head < coordinates.shape[0]:
        occupancies[head:head + max_batch_size] = intersector.query(coordinates[head:head + max_batch_size]).astype(int).reshape(-1, 1)
        head += max_batch_size
    
    occupancies = occupancies.reshape(res, res, res)
    vertices, triangles = mcubes.marching_cubes(occupancies, 0)
    mcubes.export_obj(vertices, triangles, "data/car_gt_" + str(res) + ".obj")

def generate_gt_obj(filepath):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 2048)

    compute_obj(mesh, intersector, res=1024)

def generate_volume_dataset(filepath, output_filepath, num_surface, epsilon):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 2048)

    print("Computing near surface points...")
    surface_points = compute_near_surface_points(mesh, intersector, num_surface, epsilon)

    print("Computing volume points...")
    volume_points = compute_volume_points(intersector, num_surface)

    all_points = np.concatenate([surface_points, volume_points], 0)
    np.save(output_filepath, all_points)

def generate_border_occupancy_dataset(filepath, output_filepath, count, wall_thickness=0.0025):
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    
    surface_points, _ = trimesh.sample.sample_surface(mesh, 10000000)
    kd_tree = KDTree(surface_points)
    
    volume_points = np.random.randn(count, 3) * 2 - 1
    dist, _ = kd_tree.query(volume_points, k=1)
    volume_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    near_surface_points = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(count, 3) * EPSILON
    dist, _ = kd_tree.query(near_surface_points, k=1)
    near_surface_occupancy = np.where(dist < wall_thickness, np.ones_like(dist), np.zeros_like(dist))
    
    points = np.concatenate([volume_points, near_surface_points], 0)
    occ = np.concatenate([volume_occupancy, near_surface_occupancy], 0).reshape(-1, 1)
    
    dataset = np.concatenate([points, occ], -1)
    np.save(output_filepath, dataset)

def compute_sdf(pts_with_occ, kd_tree):
    pts_occ_rt = torch.from_numpy(pts_with_occ).float().cuda()
    pts = pts_with_occ[:, :3]
    dist, _ = kd_tree.query(pts, k=1)
    mask = pts_occ_rt[:, 3]
    outside_mask = torch.where(mask > 0.5)
    dist_rt = torch.from_numpy(dist).float().cuda()

    dist_rt[outside_mask] *= -1
    dist = torch.clamp(dist_rt, -1., 1.)

    res = torch.cat([pts_occ_rt, dist.reshape(-1, 1)], -1)

    return res.cpu().numpy()

def generate_volume_dataset_new(filepath, output_filepath, num_surface, epsilon):
    mesh = filepath
    normalize_mesh(mesh)  # [-1,1]

    mesh_surface_points, _ = trimesh.sample.sample_surface(mesh, 10000000)

    kd_tree = KDTree(mesh_surface_points)

    intersector = inside_mesh.MeshIntersector(mesh, 1024)

    surface_points_with_occ = compute_near_surface_points(mesh, intersector, num_surface, epsilon)
    near_samples = compute_sdf(surface_points_with_occ, kd_tree)

    volume_points_with_occ = compute_volume_points(intersector, num_surface)
    vol_samples = compute_sdf(volume_points_with_occ, kd_tree)

    all_points = np.concatenate([near_samples, vol_samples], 0)
    np.save(output_filepath, all_points)

def generate_surface_dataset(filepath, output_filepath, num_surface):
    mesh = filepath
    normalize_mesh(mesh)  # [-1,1]
    mesh_surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)
    np.save(output_filepath, mesh_surface_points)

EPSILON = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--type', type=str, choices=['volume', 'border', 'gt', 'surface'], default='volume')
    parser.add_argument('--count', type=int, default=100000)

    args = parser.parse_args()
    
    if args.type == 'volume':
        generate_volume_dataset_new(args.input, args.output, args.count, EPSILON)
    elif args.type == 'border':
        generate_border_occupancy_dataset(args.input, args.output, args.count)
    elif args.type == "gt":
        generate_gt_obj(args.input)
    elif args.tupe == "surface":
        generate_surface_dataset(args.input, args.output, args.count)
