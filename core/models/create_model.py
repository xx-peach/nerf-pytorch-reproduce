import torch
from .nerf import NeRF
import os
import torch.optim as opt

def get_model(args, basedir, expname):
    # create the coarse NeRF model
    coarse_model = NeRF(
        d=args.netdepth,
        w=args.netwidth,
        log2_max_freq=args.multires,
        log2_max_freq_view=args.multires_views,
        i_embed=args.i_embed,
        skip=[4],
        use_viewdirs=True
    )
    
    # create the refine NeRF model
    refine_model = NeRF(
        d=args.netdepth_fine,
        w=args.netwidth_fine,
        log2_max_freq=args.multires,
        log2_max_freq_view=args.multires_views,
        i_embed=args.i_embed,
        skip=[4],
        use_viewdirs=True
    )
    
    # create optimizer
    if args.N_importance > 0:
        grad_vars = list(coarse_model.parameters()) + list(refine_model.parameters())
    else:
        grad_vars = list(coarse_model.parameters())
    optimizer = opt.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    # load checkpoints if any
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    if len(ckpts) > 0 and not args.no_reload:
        print('found ckpts: ', ckpts[-1])
        ckpt = torch.load(ckpts[-1])
        print('loading from ckpt: ', ckpts[-1])
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        coarse_model.load_state_dict(ckpt['network_fn_state_dict'])
        refine_model.load_state_dict(ckpt['network_fine_state_dict'])

    return coarse_model, refine_model, optimizer, start
