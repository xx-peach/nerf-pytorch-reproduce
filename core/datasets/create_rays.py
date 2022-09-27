import numpy as np


def get_rays(H, W, K, c2w):
    """ Function for Ray Generation, Matrix Multiplicaion, 2x Faster than Original Implementation
    Args:
        H   - the height of the input image
        W   - the width of the input image
        K   - the camera intrinsic matrix
        c2w - shape of (N, 3, 4), all images' corresponding poses
    Returns:
        rays_o - (N, H, W, 3), duplication of (3, ) translation function that move the camera origin to the world origin
        rays_d - (N, H, W, 3), equation for the light rays to the camera origin in world coordinates
    """
    # generate the 2D image pixel coordinate, each of shape (H, W)
    p_i, p_j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # compute rays' direction in camera coordinate using K^T·2D, shape (3, H*W)
    p_ij = np.concatenate([p_i[..., None], p_j[..., None], np.ones_like(p_i)[..., None]], axis=-1).reshape(-1, 3).T
    c_ij = np.linalg.inv(K) @ p_ij                                      # (3, H*W)
    # colmap 相机坐标系 x, y, z -> [右，下，屏幕朝内]，opengl 相机坐标系 x, y, z -> [右，上，屏幕朝外指向人]
    # https://github.com/bmild/nerf/issues/46
    c_ij = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ c_ij         # (3, H*W)
    # compute rays' direction in world coordinate, using R^T·3D
    rays_d = np.einsum("bij, jk -> bik", c2w[:, :3, :3], c_ij)          # (N, 3, H*W)
    rays_d = rays_d.transpose(0, 2, 1).reshape(-1, H, W, 3)             # (N, H, W, 3)
    # get rays' translation, namely camera's origin in world coordinate
    rays_o = np.expand_dims(c2w[:, :3, -1], axis=1).repeat(H, axis=1)   # (N, H, 3)
    rays_o = np.expand_dims(rays_o, axis=2).repeat(W, axis=2)           # (N, H, W, 3)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """ Shift Ray Origins to Near Plane
    Args:
        H, W   - height and width of the image
        focal  - camera's focal length
        near   - default 1, z value of near plane, scalar
        rays_o - (N, H, W, 3), duplication of (3, ) translation function that move the camera origin to the world origin
        rays_d - (N, H, W, 3), equation for the light rays to the camera origin in world coordinates
    Returns:
        rays_o - (N, H, W, 3), camera origin to the world origin after shifting
        rays_d - (N, H, W, 3), equation for the light rays to the camera origin in world coordinates after shifting
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    # projection
    o0 = -1. / (W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    
    rays_o = np.stack([o0, o1, o2], -1)
    rays_d = np.stack([d0, d1, d2], -1)
    
    return rays_o, rays_d


def get_rays_yenchenlin(H, W, K, c2w):
    """ Function for Random Ray Batching
    Args:
        H   - the height of the input image
        W   - the width of the input image
        K   - the camera intrinsic matrix
        c2w - shape of (3, 4), one image's corresponding poses
    Returns:
        rays_o - (H, W, 3), duplication of (3, ) translation function that move the camera origin to the world origin
        rays_d - (H, W, 3), equation for the light rays to the camera origin in world coordinates
    """
    # i, j both of shape (H, W), 合起来就是 (H, W) 的 2D 图像坐标
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # [i, j, 1] = K·[x_c, y_c, 1]^T = [f_x·x_c+c_x, f_y·y_c+c_y, 1], 通过 2D 图像坐标求 3D 相机坐标(z深度不知道)
    # https://github.com/bmild/nerf/issues/46
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)  # (H, W, 3)
    # rotate ray directions from camera frame to the world frame by dot product equals to [c2w.dot(dir) for dir in dirs]
    # (H, W, 1, 3) * (3, 3) -mul-> (H, W, 3, 3) -sum-> (H, W, 3)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)                         # (H, W, 3)
    # translate camera frame's origin to the world frame, it is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


trans_t = lambda t : np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,           0,            0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi),  np.cos(phi), 0],
    [0,           0,            0, 1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [         0, 1,           0, 0],
    [np.sin(th), 0,  np.cos(th), 0],
    [         0, 0,           0, 1]], dtype=np.float32)


def pose_spherical_blender(theta, phi, radius):
    """ Generate the virtual poses(camera2world matrix) from euler angles, no heading, and the
        trnasfer order is: trans -> rot_x -> rot_y -> (-row1, row3, row2, row4) permutation
    Args:
        theta  - euler angle theta, the rotation angle around the y axis, default -30.0 degree
        phi    - euler angle phi, the rotation angle around the x axis, default 4.0 degree
        radius - a fixed offset on the z axis
    Returns:
        c2w - the generated camera2world transformation matrix
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w
