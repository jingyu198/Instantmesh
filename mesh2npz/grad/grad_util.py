import torch
import numpy as np
import os
import json
from scipy.interpolate import griddata, RegularGridInterpolator

# 加载点云数据函数
def load_xyz_occ_sdf(path, i):
    pointcloud = np.load(path)

    # 使用传递的设备索引 i 来选择 CUDA 设备
    xyz = torch.from_numpy(pointcloud[:, :3]).contiguous().float().to(f'cuda:{i}')
    occupancy = torch.from_numpy(pointcloud[:, 3]).contiguous().float().to(f'cuda:{i}')
    sdf = torch.from_numpy(pointcloud[:, 4]).contiguous().float().to(f'cuda:{i}')

    # 打印 xyz 的最大最小值
    # print("XYZ min values:", xyz.min(dim=0)[0])
    # print("XYZ max values:", xyz.max(dim=0)[0])

    return xyz, occupancy, sdf


def grid_gradient(occ_path, out_path, i):
    # 使用传递的 i 作为设备参数
    xyz, occupancy, sdf = load_xyz_occ_sdf(occ_path, i)

    # chunks字典中包含每个分块的点云数据，可以用来进一步处理
    all_grad = []
    # 1. 创建规则网格
    min_vals = xyz.min(dim=0)[0].min()  # 获取每个维度的最小值
    max_vals = xyz.max(dim=0)[0].max()  # 获取每个维度的最大值
    l = max(abs(min_vals), max_vals)
    res=128
    
    x_lin = torch.linspace(-l, l, res).to(f'cuda:{i}')  # 使用传递的 i
    y_lin = torch.linspace(-l, l, res).to(f'cuda:{i}')
    z_lin = torch.linspace(-l, l, res).to(f'cuda:{i}')
    grid_x, grid_y, grid_z = torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

    grid_points = torch.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    # 2. 使用 griddata 将非规则的点云插值到规则网格上
    sdf_grid_values = griddata(xyz.cpu().numpy(), sdf.cpu().numpy(), grid_points.cpu().numpy(), method='nearest')

    sdf_grid_values = torch.tensor(sdf_grid_values, device=f'cuda:{i}').reshape((res, res, res))
    # 3. 计算梯度
    # 使用一个常数步长（假设为均匀网格），可以通过 (x_lin[1] - x_lin[0]) 来计算步长
    spacing = (x_lin[1] - x_lin[0]).item()  # 使用步长的大小（假设每个轴上的步长相同）
    grad_x, grad_y, grad_z = torch.gradient(sdf_grid_values, spacing=spacing)


    # 4. 在原始点云上插值梯度
    grad_interp_x = RegularGridInterpolator((x_lin.cpu().numpy(), y_lin.cpu().numpy(), z_lin.cpu().numpy()), grad_x.cpu().numpy())
    grad_interp_y = RegularGridInterpolator((x_lin.cpu().numpy(), y_lin.cpu().numpy(), z_lin.cpu().numpy()), grad_y.cpu().numpy())
    grad_interp_z = RegularGridInterpolator((x_lin.cpu().numpy(), y_lin.cpu().numpy(), z_lin.cpu().numpy()), grad_z.cpu().numpy())

    # 获取每个点云点的梯度
    grad_x_vals = grad_interp_x(xyz.cpu().numpy())
    grad_y_vals = grad_interp_y(xyz.cpu().numpy())
    grad_z_vals = grad_interp_z(xyz.cpu().numpy())

    # 5. 计算梯度模场
    gradients = np.stack([grad_x_vals, grad_y_vals, grad_z_vals], axis=1)  # (N, 3)
    gradients = torch.tensor(gradients, device=f'cuda:{i}')

    combined_data = torch.cat([xyz, occupancy.unsqueeze(1), sdf.unsqueeze(1), gradients], dim=1)

    assert list(combined_data.shape) == [2000000,8]
    # Save to file
    #print(combined_data[:5,:])
    grad_magnitude = torch.norm(gradients, dim=1)  # 计算每个点的梯度模长
    # 计算并打印梯度模长的均值
    grad_magnitude_mean = torch.mean(grad_magnitude)
    #print(f"梯度模长的均值: {grad_magnitude_mean.item():.4f}")
    np.save(out_path, combined_data.cpu().numpy().astype(np.float32))  # Move data to CPU for saving



if __name__ == "__main__":
    import time
    occ_path = "/data/model/objaverse_npys_100w/000074a334c541878360457c672b6c2e.npy"
    out_path = "/data/model/objaverse_npys_100w_w_grad/000074a334c541878360457c672b6c2e.npy"

    test_path = "/data/model/objaverse_npys_100w_w_grad/2cdf0cbcebee4a7cab7d556c00d3f633.npy"
    arr = np.load(test_path)
    print(arr.size)
    # start_time = time.time()  # 记录开始时间
    # grid_gradient(occ_path, out_path, 0)
    # end_time = time.time()  # 记录结束时间
    # print(f"运行时间: {end_time - start_time:.2f} 秒")

    # file_size = os.path.getsize(out_path)  # 获取文件大小，单位是字节
    # print(f"文件大小: {file_size / (1024 ** 2):.2f} MB")  # 转换为 MB 并打印

    