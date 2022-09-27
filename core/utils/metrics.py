import torch
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")


def img2mse(target, output):
    return torch.mean((target - output) ** 2)


def mse2psnr(mse):
    return -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(device))


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def create_logs(args):
    basedir, expname = args.basedir, args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')

    # create a new args file and write all the args into it
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # create a new config file and copy the using config into it
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    return basedir, expname
