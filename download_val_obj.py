import numpy as np
import os, torch
npy_paths=["objavers_v1_cam32/rendering_output_objavers_cam32/230d900e77ea4505b8cb2447c4a7400e",
        "objavers_v1_cam32/rendering_output_objavers_cam32/2a82b25aee2044908d459a9160a6b00f",
        "objavers_v1_cam32/rendering_output_objavers_cam32/a53a2477a11243be96462e9061f8315e",
        "objavers_v1_cam32/rendering_output_objavers_cam32/7e4696fef0bd4f81ad7eb840d452380b",
        "objavers_v1_cam32/rendering_output_objavers_cam32/c601af58d0874673ab9d2032863fa30a",
        "objavers_v1_cam32/rendering_output_objavers_cam32/cb9d503f2a274d799cd2cf2c60c3fae5",
        "objavers_v1_cam32/rendering_output_objavers_cam32/1725e3264af442189e080894d9c5e74c",
        "objavers_v1_cam32/rendering_output_objavers_cam32/9bfa5bd092bf4d84a4e5a563e235ddc3",]

for path in npy_paths:
    # load point cloud and occupancies
    obj_id = path.split('/')[-1]
    occ_path = os.path.join("/data/model/objaverse_npys_10w/", obj_id + '.npy')
    pointcloud = np.load(occ_path)

    # 分解点云数据，假设 pointcloud 是 n * 5 的数组
    xyz = torch.tensor(pointcloud[:, :3], dtype=torch.float)  # x, y, z 坐标
    transform_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=torch.float)

    #xyz = xyz @ transform_matrix
    occupancy = pointcloud[:, 3]  # 占据值
    sdf = pointcloud[:, 4]  # sdf 值

    print("一共有多少点：", xyz.shape)
    # 过滤 occupancy 为 1 的点
    filtered_points = xyz[occupancy == 1]

    # 创建一个 OBJ 文件并写入数据
    with open(f'/home/gjy/jingyu/InstantMesh/examples/val/{obj_id}0.obj', 'w') as f:
        # 写入顶点坐标
        for point in filtered_points:
            f.write(f'v {point[0]} {point[1]} {point[2]}\n')
