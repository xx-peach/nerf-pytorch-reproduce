import os
import numpy as np
import torch
import time
import imageio
from tqdm import tqdm, trange

from core.utils.create_configs import config_parser
from core.datasets.create_dataset import NeRFDataSet
from core.datasets.create_dataloader import get_dataloader, get_next_batch
from core.models.create_model import get_model
from core.utils.metrics import mse2psnr, img2mse, create_logs, to8b
from core.utils.train_nerf_utils import train_one_iter
from core.utils.test_nerf_utils import render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
np.random.seed(0)


def main():
    # create parser and parse arguments
    parser = config_parser()
    args = parser.parse_args()

    # create train dataset and test dataset
    train_dataset = NeRFDataSet(args=args, split='train')
    print('size of training dataset is: ', train_dataset.poses.shape[0])
    test_dataset  = NeRFDataSet(args=args, split='test')
    print('size of testing dataset is: ', test_dataset.poses.shape[0])
    fake_dataset  = NeRFDataSet(args=args, split='fake')
    print('size of fake dataset is: ', fake_dataset.poses.shape[0])

    # create train dataloader
    dataloader = get_dataloader(args=args, dataset=train_dataset, shuffle=True)

    # create test log for this experiment
    basedir, expname = create_logs(args)

    # create NeRF model
    coarse_model, refine_model, optimizer, start = get_model(args=args, basedir=basedir, expname=expname)
    # transfer the model to gpu
    coarse_model, refine_model = coarse_model.to(device), refine_model.to(device)

    # start training
    global_step = start
    start = start + 1
    print('start training step is: {}, total training step is: {}'.format(start, args.iters))
    for i in trange(start, args.iters+1):
        start_time = time.time()        # start to record the training time
        
        # load the pre-processed data of this batch, (B, 3) and transfer them to target device
        tgt_images, src_rays_o, src_rays_d, src_viewdirs, src_near, src_far = get_next_batch(dataloader, i, no_batching=args.no_batching)
        tgt_images, src_rays_o, src_rays_d = tgt_images.to(device), src_rays_o.to(device), src_rays_d.to(device)
        src_viewdirs, src_near, src_far = src_viewdirs.to(device), src_near.to(device), src_far.to(device)
        # train for one step and get output rgb color
        refine_rgb, refine_dpt, refine_dsp, refine_acc, coarse_rgb, coarse_dpt, coarse_dsp, coarse_acc = train_one_iter(
            coarse_model, refine_model, src_rays_o, src_rays_d, src_viewdirs, src_near, src_far, 
            args.N_samples, args.lindisp, args.perturb, args.N_importance, args.raw_noise_std, args.white_bkgd
        )
        
        # clear the optimizer and compute loss
        optimizer.zero_grad()
        refine_loss = img2mse(refine_rgb, tgt_images)
        coarse_loss = img2mse(coarse_rgb, tgt_images)
        loss = refine_loss + coarse_loss
        psnr = mse2psnr(refine_loss)
        
        # perform loss back-propogation
        loss.backward()
        optimizer.step()
        
        # update the learning rate
        decay_rate = 1e-1
        decay_step = args.lrate_decay * 1000
        new_lrates = args.lrate * (decay_rate ** (global_step / decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrates
        
        dt = time.time() - start_time   # compute the time now

        # all logging stuffs
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': coarse_model.state_dict(),
                'network_fine_state_dict': refine_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('saved checkpoints at', path)
        
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, dsps = render(
                    args, coarse_model, refine_model, fake_dataset.rays_o, fake_dataset.rays_d,
                    fake_dataset.viewdirs, fake_dataset.near, fake_dataset.far
                )
            print('done, saving', rgbs.shape, dsps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(dsps / np.max(dsps)), fps=30, quality=8)
        
        # if i % args.i_testset == 0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #     print('test poses shape', test_dataset.poses.shape)
        #     with torch.no_grad():
        #         _, _ = render(
        #             args, coarse_model, refine_model, test_dataset.rays_o, test_dataset.rays_d,
        #             test_dataset.viewdirs, test_dataset.near, test_dataset.far,
        #             savedir=testsavedir, gt_imgs=test_dataset.imgs
        #         )
        #     print('saved test set')
        
        if i % args.i_print == 0:
            tqdm.write(f"[train] iters: {i:5d}, loss: {loss.item():.8f}, PSNR: {psnr.item():2.6f}")
        
        # increment global_step
        global_step += 1



if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
