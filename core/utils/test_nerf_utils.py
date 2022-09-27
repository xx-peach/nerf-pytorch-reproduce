import time
import os
import numpy as np
from tqdm import tqdm
import imageio
import torch

from .train_nerf_utils import train_one_iter
from .metrics import to8b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

def render(args, coarse_model, refine_model, src_rays_o, src_rays_d, src_viewdirs,
           src_near, src_far, savedir=None, gt_imgs=None):
    """ Rendering Given Poses
    Args:
        args          - arguments
        coarse_model  - NeRF model for coarse sampling
        refine_model  - NeRF model for refinement
        src_rays_o    - (N, H, W, 3), rays'(camera) origin in world coordinate
        src_rays_d    - (N, H, W, 3), rays' directions in world coordinate
        src_viewdirs  - (N, H, W, 3), normalized src_rays_d corresponding to rays in this batch
        src_near      - int, distance of near plane, duplication of scalar
        src_far       - int, distance of far plane, duplication of scalar
        render_factor - whether to reduce the test image resolution
        savedir       - directory to save the middle results
        gt_imgs       - (N, H, W, 3), ground truth images
    Returns:
        all_rgbs - (N, H, W, 3), RGB color of rays in this batch
        all_dsps - (N, H, W, ), disparity of rays in this batch
    """
    N, H, W = src_rays_d.shape[0], src_rays_d.shape[1], src_rays_d.shape[2]
    # reshape all the input datas
    src_rays_d, src_rays_o, src_viewdirs = src_rays_d.reshape(N, -1, 3), src_rays_o.reshape(N, -1, 3), src_viewdirs.reshape(N, -1, 3)
    # transfer all the datas to gpu
    src_rays_d, src_rays_o = torch.from_numpy(src_rays_d).type(torch.float32).to(device), torch.from_numpy(src_rays_o).type(torch.float32).to(device)
    src_near, src_far = torch.from_numpy(src_near).type(torch.float32).to(device), torch.from_numpy(src_far).type(torch.float32).to(device)
    src_viewdirs = torch.from_numpy(src_viewdirs).type(torch.float32).to(device)

    # start to record the time
    t = time.time()
    
    # start to render
    chunk = 400
    all_rgbs, all_dsps = [], []
    with torch.no_grad():
        for i in tqdm(range(N), desc='test rendering: '):
            # to compute one image
            temp_rgbs, temp_dsps = [], []
            for j in range(0, H * W, chunk):
                # many steps to compute one image
                rgb, dpt, dsp, acc, _, _, _, _ = train_one_iter(
                    coarse_model, refine_model, src_rays_o[i][j:j+chunk], src_rays_d[i][j:j+chunk], 
                    src_viewdirs[i][j:j+chunk], src_near, src_far, 
                    args.N_samples, args.lindisp, False, args.N_importance, 0, args.white_bkgd
                )
                temp_rgbs.append(rgb.cpu().numpy())
                temp_dsps.append(dsp.cpu().numpy())
            
            all_rgbs.append(np.concatenate(temp_rgbs, axis=0).reshape(H, W, -1))
            all_dsps.append(np.concatenate(temp_dsps, axis=0).reshape(H, W, -1))

            if gt_imgs is not None and args.render_factor == 0:
                p = -10. * np.log10(np.mean(np.square(all_rgbs[i] - gt_imgs[i])))
                print('test loss: {:6f}'.format(p))

            if savedir is not None:
                rgb8 = to8b(all_rgbs[i])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)
    
        # prepare for return
        all_rgbs = np.stack(all_rgbs, axis=0)
        all_dsps = np.stack(all_dsps, axis=0)

    return all_rgbs, all_dsps
