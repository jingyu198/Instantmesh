import numpy as np

# 加载点云数据
pc = np.load('/data/model/objaverse_npys_100w_surface/c601af58d0874673ab9d2032863fa30a.npy')
print(f"Point cloud shape: {pc.shape}")

# 截取点云的特定部分（这里是截取第 1,000,000 以后的点）
# 定义保存的 .obj 文件路径
surf_path = '/home/gjy/jingyu/InstantMesh/mesh2npz/grad/1.obj'

# 将点云数据保存为 .obj 文件
with open(surf_path, 'w') as f:
    for point in pc:
        # OBJ 格式中的顶点信息以 "v x y z" 表示
        f.write(f"v {point[0]} {point[1]} {point[2]}\n")

print(f"Saved to {surf_path}")
