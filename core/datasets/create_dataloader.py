import numpy as np
import torch
from torch.utils.data import DataLoader
np.random.seed(0)

def get_dataloader(args, dataset, shuffle=True):
    if args.no_batching:
        dataloader = NoBatchingDataLoader(args=args, dataset=dataset)
    else:
        # dataloader = DataLoader(dataset, args.N_rand, shuffle=shuffle)
        dataloader = BatchingDataLoader(args=args, dataset=dataset)
    # return the created dataloader
    return dataloader


def get_next_batch(dataloader, iters, no_batching=True):
    """ Return the Next Batch Data
    Args:
        dataloader  - torch.utils.data.DataLoader / NoBatchingDataLoader
        no_batching - rays in this batch comes from one specific image if 1, else arbitrary
    Returns:
        tgt_images   - (batch_size, 3), actual RGB color of the rays in this batch
        src_rays_o   - (batch_size, 3), rays'(camera) origin in world coordinate
        src_rays_d   - (batch_size, 3), rays' directions in world coordinate
        src_viewdirs - (batch_size, 3), normalized src_rays_d corresponding to rays in this batch
        src_near     - (batch_size, 1), distance of near plane, duplication of scalar
        src_far      - (batch_size, 1), distance of far plane, duplication of scalar
    """
    if no_batching:
        tgt_images, src_rays_o, src_rays_d, src_viewdirs, src_near, src_far = dataloader.next_data(iters)
    else:
        # tgt_images, src_rays_o, src_rays_d, src_viewdirs, src_near, src_far = next(iter(dataloader))
        # src_near, src_far = src_near[0], src_far[0]
        tgt_images, src_rays_o, src_rays_d, src_viewdirs, src_near, src_far = dataloader.next_data()
    # return the next batch's data
    return tgt_images.type(torch.float32), src_rays_o.type(torch.float32), src_rays_d.type(torch.float32), \
           src_viewdirs.type(torch.float32), src_near.type(torch.float32), src_far.type(torch.float32)


class NoBatchingDataLoader():
    def __init__(self, args, dataset) -> None:
        self.batch_size = args.N_rand               # batch size
        self.precrop_iters = args.precrop_iters     # number of steps to train on central crops
        self.precrop_frac = args.precrop_frac       # fraction of img taken for central crops
        self.near, self.far = dataset.near, dataset.far
        self.H, self.W = dataset.H, dataset.W
        self.dataset = dataset
        # generate coordinates for each pixel in the 2D cropped image, (2dH*2dW, 2)
        dH, dW = int(self.H//2*self.precrop_frac), int(self.W//2*self.precrop_frac)
        # generate indices for cropped 2D images
        crop_pi, crop_pj = np.meshgrid(
            np.arange(self.H//2 - dH, self.H//2 + dH, dtype=np.float32), np.arange(self.W//2 - dW, self.W//2 + dW, dtype=np.float32),
        )
        self.crop_p_ij = np.stack([crop_pi.T, crop_pj.T], axis=-1).reshape(-1, 2)
        # generate indices for original 2D images
        orig_pi, orig_pj = np.meshgrid(
            np.arange(self.H, dtype=np.float32), np.arange(self.W, dtype=np.float32),
        )
        self.orig_p_ij = np.stack([orig_pi.T, orig_pj.T], axis=-1).reshape(-1, 2)
        
    
    def next_data(self, iters):
        # choose a random image
        idx = np.random.choice(self.dataset.imgs.shape[0])

        # get the target near plane and far plane for this batch
        src_near, src_far = self.near, self.far                     # (batch_size, 1)
        
        # get the whole image, rays_o, rays_d and viewdirs of the chosen
        tgt_images = self.dataset.imgs[idx]                         # (H, W, 3)
        src_rays_o, src_rays_d, src_viewdirs = self.dataset.rays_o[idx], self.dataset.rays_d[idx], self.dataset.viewdirs[idx]   # (H, W, 3)
        
        # generate the coordinates of each pixel in the 2D image plane
        if iters < self.precrop_iters:
            if iters == 1: print("[config] center cropping is enabled until iter {}".format(self.precrop_iters))
            sel_idx = np.random.choice(self.crop_p_ij.shape[0], self.batch_size, replace=False) # (batch_size, )
            sel_idx = self.crop_p_ij[sel_idx].astype(np.longlong)                               # (batch_size, 2)
        else:
            sel_idx = np.random.choice(self.orig_p_ij.shape[0], self.batch_size, replace=False) # (batch_size, )
            sel_idx = self.orig_p_ij[sel_idx].astype(np.longlong)                               # (batch_size, 2)
        
        # randomly choose `batch_size` 2D pixels
        tgt_images = tgt_images[sel_idx[:, 0], sel_idx[:, 1]]       # (batch_size, 3)
        src_rays_o = src_rays_o[sel_idx[:, 0], sel_idx[:, 1]]       # (batch_size, 3)
        src_rays_d = src_rays_d[sel_idx[:, 0], sel_idx[:, 1]]       # (batch_size, 3)
        src_viewdirs = src_viewdirs[sel_idx[:, 0], sel_idx[:, 1]]   # (batch_size, 3)

        return torch.from_numpy(tgt_images), torch.from_numpy(src_rays_o), torch.from_numpy(src_rays_d), \
               torch.from_numpy(src_viewdirs), torch.from_numpy(src_near), torch.from_numpy(src_far)


class BatchingDataLoader():
    def __init__(self, args, dataset) -> None:
        self.batch_size = args.N_rand               # batch size
        self.i_batch = 0

        self.near, self.far = dataset.near, dataset.far
        self.H, self.W = dataset.H, dataset.W
        # concatenate rays_o, rays_d and viewdirs together and shuffle, (N*H*W, 4, 3)
        self.data = np.stack([
            dataset.rays_o.reshape(-1, 3), dataset.rays_d.reshape(-1, 3), 
            dataset.viewdirs.reshape(-1, 3), dataset.imgs.reshape(-1, 3)
        ], axis=1)
        np.random.shuffle(self.data)
        # print("shuffle: ", self.data.shape[0], self.data[0], self.data[-1], self.data[187])
        
    
    def next_data(self):
        # get the training rays origin, coordinate, and view directions for this batch
        src_rays_o   = self.data[self.i_batch:self.i_batch+self.batch_size, 0]
        src_rays_d   = self.data[self.i_batch:self.i_batch+self.batch_size, 1]
        src_viewdirs = self.data[self.i_batch:self.i_batch+self.batch_size, 2]
        # get the target images for this batch
        tgt_images   = self.data[self.i_batch:self.i_batch+self.batch_size, 3]
        # get the target near plane and far plane for this batch
        src_near, src_far = self.near, self.far
        
        # increment i_batch
        self.i_batch = self.i_batch + self.batch_size

        # re-shuffle the data after one epoch
        if self.i_batch >= self.data.shape[0]:
            rand_idxs = np.random.permutation(self.data.shape[0])
            self.data = self.data[rand_idxs]
            self.i_batch = 0
        
        return torch.from_numpy(tgt_images), torch.from_numpy(src_rays_o), torch.from_numpy(src_rays_d), \
               torch.from_numpy(src_viewdirs), torch.from_numpy(src_near), torch.from_numpy(src_far)
