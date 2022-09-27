import numpy as np
from torch.utils.data import Dataset

from .load_llff import load_llff_data
from .load_blender import load_blender_data
from .create_rays import get_rays, ndc_rays, get_rays_yenchenlin



class NeRFDataSet(Dataset):
    """ Dataset for NeRF Data
    Args:
        args  - arguments
        split - ['train', 'val', 'test', 'fake']
    Member:
        imgs     - shape of (N, H, W, 3), all the loaded images
        poses    - shape of (N, 4, 4), all the loaded poses
        H, W     - height and width of the image
        K        - camera's intrinsic matrix
        near     - shape of (1, ), distance of near plane, scalar
        far      - shape of (1, ), distance of far plane, scalar
        rays_o   - (N, H, W, 3), duplication of (3, ) translation function that move the camera origin to the world origin
        rays_d   - (N, H, W, 3), equation for the light rays to the camera origin in world coordinates
        viewdirs - (N, H, W, 3), normalized rays directions
    """
    def __init__(self, args, split='train') -> None:
        self.split = split
        # load raw data and do some most basic pre-processing
        self.imgs, self.poses, self.near, self.far, self.H, self.W, self.K = self.load_data(args, split)

        self.modules_0, self.modules_1 = self.H * self.W * 3, self.W * 3
        
        # my own implementation of rays generation, matrix multiplication version
        self.rays_o, self.rays_d = get_rays(self.H, self.W, self.K, self.poses)
        
        # # (N, H, W, 3): generate rays of each image's each pixel
        # rays = np.stack([get_rays_yenchenlin(self.H, self.W, self.K, p) for p in self.poses[:, :3, :4]], 0)
        # self.rays_o, self.rays_d = rays[:, 0, :, :, :], rays[:, 1, :, :, :]

        # ndc(used for forward facing data) only good for 'llff' data
        if args.dataset_type == 'llff' and not args.no_ndc:
            self.rays_o, self.rays_d = ndc_rays(self.H, self.W, self.K[0, 0], 1., self.rays_o, self.rays_d)

        # (N, H, W, 3): create view directions
        self.viewdirs = self.rays_d / np.linalg.norm(self.rays_d, axis=-1, keepdims=True)


    def __len__(self):
        return self.imgs.shape[0] * self.imgs.shape[1] * self.imgs.shape[2]


    def __getitem__(self, index):
        # compute the indices of each dimension
        indice_0 = index // self.modules_0
        residual = index - indice_0 * self.modules_0
        indice_1 = residual // self.modules_1
        indice_2 = (residual - indice_1 * self.modules_1) // 3
        # return the corresponding data
        tgt_images = self.imgs[indice_0, indice_1, indice_2]
        src_viewdirs = self.viewdirs[indice_0, indice_1, indice_2]
        src_rays_o, src_rays_d = self.rays_o[indice_0, indice_1, indice_2], self.rays_d[indice_0, indice_1, indice_2]
        return tgt_images, src_rays_o, src_rays_d, src_viewdirs, self.near, self.far


    def load_data(self, args, split):
        if args.dataset_type == 'blender':
            imgs, poses, near, far, H, W, K = load_blender_data(
                args.datadir, args.half_res, args.testskip, split, args.render_factor
            )
            imgs = self.trans_white_bkgd(imgs) if args.white_bkgd else imgs[..., :3]
        
        if args.dataset_type == 'llff':
            imgs, poses, near, far, H, W, K = load_llff_data(
                args.datadir, args.factor, recenter=True, bd_factor=0.75, spherify=args.spherify, 
                llffhold=args.llffhold, split=split, no_ndc=args.no_ndc
            )
        
        return imgs, poses, near, far, H, W, K


    def trans_white_bkgd(self, imgs):
        return imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])

