import os
import numpy as np
import imageio 
import json
import cv2
from .create_rays import pose_spherical_blender

def load_blender_data(basedir, half_res=False, testskip=1, split='train', render_factor=0):
    """ Loading blender data, eg. lego, ...
    Args:
        basedir  - the input data directory
        half_res - whether to load blender synthetic data at 400x400 instead of 800x800
        testskip - will load 1/N images from test/val sets, useful for large datasets like deepvoxels
        split    - ['train', 'val', 'test', 'fake']
    Returns:
        imgs  - shape of (N, H, W, 3), all the loaded images
        poses - shape of (N, 4, 4), all the loaded poses
        near  - shape of (1, ), z value of near plane
        far   - shape of (1, ), z value of the far plane
        H     - height of the image
        W     - width of the image
        K     - camera's intrinsic matrix
    """
    # read the original json files(each frame's image path and pose) for dataset 'split'
    raw_data = None
    if split == 'fake': fn = 'test'     # if split is 'fake', load 'test' instead
    else: fn = split                    # if split in ['train', 'val', 'test'], load it
    with open(os.path.join(basedir, 'transforms_{}.json'.format(fn)), 'r') as fp:
        raw_data = json.load(fp)

    imgs, poses = [], []
    # determine the period of an actual loading
    if split == 'train' or testskip == 0: skip = 1
    else: skip = testskip
    # load the image and corresponding pose every skip times using the json file information
    for frame in raw_data['frames'][::skip]:
        img_name = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(imageio.imread(img_name))
        poses.append(np.array(frame['transform_matrix']))
    # divide the image matrix by 255 and cast it into float32, and keep all 4 channels (RGBA)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    # cat the type of all the loaded poses to np.float32
    poses = np.array(poses).astype(np.float32)

    # compute the camera intrinsic matrix
    H, W = int(imgs[0].shape[0]), int(imgs[0].shape[1])             # image's height and width
    camera_angle_x = float(raw_data['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)                    # camera's focal length, pinhole

    # reduce the resolution if we got the 'half_res' signal
    if half_res:
        H, W, focal = H // 2, W // 2, focal / 2.    # half height, width and focal length
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    # re-compute the camera intrinsic
    if (split == 'test' or split == 'fake') and render_factor > 0:
        H, W, focal = H // render_factor, W // render_factor, focal / render_factor
    K = np.array([[focal, 0, 0.5*W], [0, focal, 0.5*H], [0, 0, 1]]) # camera's intrinsic matrix

    if split == 'fake':
        poses = np.stack([pose_spherical_blender(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    # near, far plane for blender data
    near, far = 2, 6
    near, far = np.array(near), np.array(far)

    # return imgs, poses, H, W, and K
    return imgs, poses, near, far, H, W, K
