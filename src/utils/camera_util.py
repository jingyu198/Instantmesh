import torch
import torch.nn.functional as F
import numpy as np


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
    elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)
    
    c2ws = spherical_camera_pose(azimuths, elevations, radius)
    c2ws = c2ws.float().flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(6, 1, 1).float().flatten(-2)

    extrinsics = c2ws[:, :12]
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    return cameras.unsqueeze(0).repeat(batch_size, 1, 1)

def get_zero123plus_c2ws(cam_distance=4.0):
    azimuths = np.array([30, 90, 150, 210, 270, 330])
    elevations = np.array([20, -10, 20, -10, 20, -10])
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    x = cam_distance * np.cos(elevations) * np.cos(azimuths)
    y = cam_distance * np.cos(elevations) * np.sin(azimuths)
    z = cam_distance * np.sin(elevations)

    cam_locations = np.stack([x, y, z], axis=-1)  # (6, 3)
    cam_locations = torch.from_numpy(cam_locations).float()
    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws.float()

def rotate_z_clockwise(loc, angle_degrees):
        # 角度转弧度
        angle_radians = np.radians(angle_degrees)
        # 绕Z轴的旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_radians), np.sin(angle_radians), 0],
            [-np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
        # 原始坐标
        loc = np.array(loc)
        # 计算旋转后的坐标
        rotated_loc = np.dot(rotation_matrix, loc)
        return rotated_loc    
    
#########################################################
#########################################################
#########################################################

def calculate_angles_and_length(v):
    x,y,z = v
    length = np.sqrt(x**2 + y**2 + z**2)
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    return yaw_deg, pitch_deg, length

def calculate_vector(yaw_deg, pitch_deg, distance):
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    x = distance * np.cos(pitch_rad) * np.cos(yaw_rad)
    y = distance * np.cos(pitch_rad) * np.sin(yaw_rad)
    z = distance * np.sin(pitch_rad)   
    return [x, y, z]

def load_c2ws_gt(camera_path):
    azimuths = []
    elevations = []
    with open(camera_path, 'r') as file:
        lines = file.readlines()
        current_view = None
        for line in lines:
            if line.startswith("CameraPos"):
                parts = line.strip().split()
                loc = [float(parts[1].split('=')[1]), 
                                                    -float(parts[2].split('=')[1]), 
                                                    float(parts[3].split('=')[1])]
                azimuth, elevation, radiu = calculate_angles_and_length(loc)
                azimuths.append(azimuth)
                elevations.append(elevation)
    
    azimuths = np.array(azimuths)
    elevations = np.array(elevations)
    c2ws = spherical_camera_pose(azimuths, elevations, 250/40)
    return c2ws


def load_c2ws_mv(camera_path, pick_index):
    scale=100
    cam_locations= []
    with open(camera_path, 'r') as file:
        lines = file.readlines()
        current_view = None
        index=-1
        for line in lines:
            if line.startswith("CameraPos"):
                index += 1
                if index == pick_index:
                    parts = line.strip().split()
                    loc = [float(parts[1].split('=')[1]), 
                                                        -float(parts[2].split('=')[1]), 
                                                        float(parts[3].split('=')[1])]
    
    distance = 250
    offsets_a = [30, 90, 150, 210, 270, 330]
    offsets_b = [20, -10, 20, -10, 20, -10]
    yaw_deg, pitch_deg, length = calculate_angles_and_length(loc)
    cam_locations = [calculate_vector(a+yaw_deg, b, distance) for a, b in zip(offsets_a, offsets_b)]
    cam_locations=np.array(cam_locations)/scale
    cam_locations = torch.from_numpy(cam_locations).float()
    c2ws = center_looking_at_camera_pose(cam_locations).float()
    return c2ws

