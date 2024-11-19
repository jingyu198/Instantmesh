import trimesh
import torch
import torchcumesh2sdf

import numpy as np
import diso
import mcubes


def load_and_preprocess(p, band):
    # 加载 glb 文件为场景对象
    mesh_scene = trimesh.load(p, force='scene')
    
    # 提取场景中的 Trimesh 子对象并合并
    meshes = [m for m in mesh_scene.dump() if isinstance(m, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("No Trimesh objects found in the scene.")
    
    # 合并为单个 Trimesh 对象
    mesh = trimesh.util.concatenate(meshes)

    # 获取三角形数据
    tris = np.array(mesh.triangles, dtype=np.float32)
    # 归一化
    tris = tris - tris.min(0).min(0)
    tris = (tris / tris.max() + band) / (1 + band * 2)
    
    return torch.tensor(tris, dtype=torch.float32, device='cuda:0')
    
def get_watertight_mesh(mesh_dir, out_dir, res = 256, batch_size=10_000):
    band = 8/res
    tris = load_and_preprocess(mesh_dir,band)
    sdf = torchcumesh2sdf.get_sdf(tris, res, band, batch_size)-2/res
    v, f = diso.DiffMC().cuda().forward(sdf) # todo: how to smooth?
    # v, f, _, _ = skimage.measure.marching_cubes(sdf.cpu().numpy(), 2/res)
    # v,f = mcubes.marching_cubes(sdf.cpu().numpy(), 2/res)
    # to [0,1]
    v_01 = v/res
    # to (-1,1)
    new_v = (v_01 *2 - 1.0)*0.9
    #mcubes.export_obj(new_v.cpu().numpy(), f.cpu().numpy(), out_dir)
    mesh = trimesh.Trimesh(vertices=new_v.cpu().numpy(), faces=f.cpu().numpy(), process=False)
    return mesh

if __name__ == '__main__':
    mesh_dir = '/home/gjy/jingyu/InstantMesh/mesh2npz/data/11.glb'
    out_dir =  '/home/gjy/jingyu/InstantMesh/mesh2npz/data/11_wt.obj'
    get_watertight_mesh(mesh_dir, out_dir)


# mesh = trimesh.load(mesh_dir, force='mesh')
# is_watertight = mesh.is_watertight
# print(f"水密么? {is_watertight}")

# mesh = trimesh.load(out_dir, force='mesh')
# is_watertight = mesh.is_watertight
# print(f"水密么? {is_watertight}")
