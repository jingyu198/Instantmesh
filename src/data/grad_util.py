import torch
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator

def get_sdf_grad(query_xyz, sdf_gt, device='cuda'):
    batch_size = query_xyz.shape[0]
    gradients_list = []

    for b in range(batch_size):
        xyz = query_xyz[b].to(device).float()  # 确保转换为float32
        sdf = sdf_gt[b].to(device).float()  # 确保转换为float32

        min_vals = xyz.min(dim=0)[0]  # 获取每个维度的最小值
        max_vals = xyz.max(dim=0)[0]  # 获取每个维度的最大值

        res = 50

        x_lin = torch.linspace(min_vals[0], max_vals[0], res).to(device).float()  # 使用float32
        y_lin = torch.linspace(min_vals[1], max_vals[1], res).to(device).float()
        z_lin = torch.linspace(min_vals[2], max_vals[2], res).to(device).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        grid_points = torch.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        # 2. 使用 griddata 将非规则的点云插值到规则网格上
        sdf_grid_values = griddata(xyz.cpu().numpy(), sdf.cpu().numpy(), grid_points.cpu().numpy(), method='linear')
        sdf_grid_values = torch.tensor(sdf_grid_values, dtype=torch.float32).to(device).reshape((res, res, res))  # 转换为float32

        # 3. 计算梯度
        spacing = (x_lin[1] - x_lin[0]).item()  # 使用步长的大小
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

        gradients_list.append(torch.tensor(gradients, device=device, dtype=torch.float32))  # 确保转换为float32

    # 合并所有batch的梯度
    gradients_all = torch.stack(gradients_list)  # (b, 200000, 3)
    return gradients_all
