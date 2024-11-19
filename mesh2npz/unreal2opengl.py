import trimesh
import numpy as np
import torch

# 加载点云数据
path = "/data/model/objaverse_npys_100w/000074a334c541878360457c672b6c2e.npy"
path_glb = '/data/model/objaverse_glbs/000074a334c541878360457c672b6c2e.glb'
pointcloud = np.load(path)

# 使用传递的设备索引 i 来选择 CUDA 设备
xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float()
occupancy = torch.from_numpy(pointcloud[:, 3]).contiguous().float()
sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float()

# 获取前1000000个点
xyz_np_first_half = xyz[:1000000].cpu().numpy()

# 使用 trimesh 创建前1000000个点的点云
point_cloud_first_half = trimesh.points.PointCloud(xyz_np_first_half)

# 保存为 obj 格式
output_path_first_half = '/home/gjy/jingyu/InstantMesh/pointcloud_first_half.obj'
point_cloud_first_half.export(output_path_first_half)

# 获取后1000000个点
xyz_np_second_half = xyz[1000000:].cpu().numpy()

# 使用 trimesh 创建后1000000个点的点云
point_cloud_second_half = trimesh.points.PointCloud(xyz_np_second_half)

# 保存为 obj 格式
output_path_second_half = '/home/gjy/jingyu/InstantMesh/pointcloud_second_half.obj'
point_cloud_second_half.export(output_path_second_half)

# 加载 .glb 文件并转换为 .obj 格式
mesh = trimesh.load(path_glb)

# 保存为 obj 格式
output_path_glb_to_obj = '/home/gjy/jingyu/InstantMesh/mesh_from_glb.obj'
mesh.export(output_path_glb_to_obj)

print(f"First half of point cloud saved to {output_path_first_half}")
print(f"Second half of point cloud saved to {output_path_second_half}")
print(f"Mesh from GLB saved to {output_path_glb_to_obj}")
