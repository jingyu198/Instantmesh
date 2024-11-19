import torch
xyz = torch.tensor([1,2,3])
transform_matrix = torch.tensor([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])
xyz = xyz @ transform_matrix
        

print(xyz)


def unreal_to_opengl(x, y, z):
    x_opengl = x
    y_opengl = -z
    z_opengl = y
    return [x_opengl, y_opengl, z_opengl]

# 示例
xyz_unreal = [1.0, 2.0, 3.0]
xyz_opengl = unreal_to_opengl(*xyz_unreal)

print(xyz_opengl)