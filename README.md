Instantmesh original version：
https://github.com/TencentARC/InstantMesh

mesh preprocess (including watertight mesh transferring and volume/surface sampling)：

Cpu version: mesh2sdf
reference: https://github.com/wang-ps/mesh2sdf
Low efficiency for high resolution 

GPU version: cumesh2sdf
Origin repo: https://github.com/eliphatfs/cumesh2sdf
Refined-version repo:https://github.com/CvHadesSun/cumesh2sdf
- Pre-install git-repo
- Refine for thin mesh structure get sdf, hy-param:
  - band = 8/res
  - Sdf -=2/res
Sample scripts:
https://github.com/CvHadesSun/MeshSample
