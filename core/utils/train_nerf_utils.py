from asyncio import base_tasks
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_iter(coarse_model, refine_model, src_rays_o, src_rays_d, 
                   src_viewdirs, src_near, src_far, N_samples, lindisp, perturb,
                   N_importance, raw_noise_std, white_bkgd):
    """ Train for One Step
    Args:
        coarse_model - NeRF model for coarse sampling
        refine_model - NeRF model for refinement
        src_rays_o   - (B, 3) / (H, W, 3), rays'(camera) origin in world coordinate
        src_rays_d   - (B, 3) / (H, W, 3), rays' directions in world coordinate
        src_viewdirs - (B, 3) / (H, W, 3), normalized src_rays_d corresponding to rays in this batch
        src_near     - (1, ), distance of near plane, duplication of scalar
        src_far      - (1, ), distance of far plane, duplication of scalar
    Returns:
        rgb - (B, 3) / (H, W, 3), RGB color of rays in this batch
        dpt - (B, )  / (H, W, ), depth of rays in this batch
        dsp - (B, )  / (H, W, ), disparity of rays in this batch
        acc - (B, )  / (H, W, ), sum of weights along each ray
    """
    # sample points(3D coordinates) from the pre-processed data
    coarse_pts, z_vals = coarse_sampling(src_rays_o, src_rays_d, src_near, src_far, N_samples, lindisp, perturb)
    # reshape the 3D points coordinates and view directions
    coarse_viewdirs = src_viewdirs[:, None].expand(coarse_pts.shape).reshape(-1, coarse_pts.shape[-1])  # (B*N_samples, 3)
    coarse_pts = coarse_pts.reshape(-1, coarse_pts.shape[-1])                                           # (B*N_samples, 3)
    # go through coarse NeRF model and get raw output
    coarse_raw = coarse_model(coarse_pts, coarse_viewdirs)      # (B*N_samples, 4)
    # parse the raw outputs
    rgb, dpt, dsp, acc, weights = raw2output(coarse_raw, z_vals, src_rays_d, raw_noise_std, white_bkgd)
    
    if N_importance > 0:
        # store the output of the coarse NeRF model
        coarse_rgb, coarse_dpt, coarse_dsp, coarse_acc = rgb, dpt, dsp, acc
        # hierarchical sampling around the original sample points with higher weights
        z_vals_new = refine_sampling(z_vals, weights[..., 1:-1], N_importance, perturb)         # (B, N_importance)
        # 因为 z_vals_new 的计算中用到了 weights 当作输入，而 weights 又恰好是 nerf 的输出 require_grad=True
        z_vals_new = z_vals_new.detach()
        z_vals_all, _ = torch.sort(torch.cat([z_vals, z_vals_new], dim=-1), dim=-1)             # (B, N_samples + N_importance)
        refine_pts = src_rays_o[:, None, :] + src_rays_d[:, None, :] * z_vals_all[:, :, None]   # (B, N_samples + N_importance, 3)
        # reshape the 3D points coordinates and view directions
        refine_viewdirs = src_viewdirs[:, None].expand(refine_pts.shape).reshape(-1, refine_pts.shape[-1])
        refine_pts = refine_pts.reshape(-1, refine_pts.shape[-1])
        # go through refine NeRF model and get raw output
        refine_raw = refine_model(refine_pts, refine_viewdirs)
        # parse the raw outputs
        rgb, dpt, dsp, acc, weights = raw2output(refine_raw, z_vals_all, src_rays_d, raw_noise_std, white_bkgd)

    return rgb, dpt, dsp, acc, coarse_rgb, coarse_dpt, coarse_dsp, coarse_acc



def coarse_sampling(rays_o, rays_d, near, far, N_samples, lindisp, perturb):
    """ Sample Points from Rays in this Batch
    Args:
        rays_o    - (B, 3), rays'(camera) origin in world coordinate
        rays_d    - (B, 3), rays' directions in world coordinate
        near      - (1, ), distance of near plane, duplication of scalar
        far       - (1, ), distance of far plane, duplication of scalar
        N_samples - int, number of points we're going to sample
        lindisp   - whether to sampling linearly in depth or disparity
        perturb   - whether to add some noise to break linearity
    Returns:
        pts    - (B, N_samples, 3), final 3D coordinates of sample points
        z_vals - (B, N_samples), z values of all sample points
    """
    batch_size = rays_d.shape[0]
    # compute z values of sample points linearly, in depth or disparity, (B, N_samples)
    if not lindisp:
        z_vals = torch.linspace(near, far, N_samples).reshape(1, -1).repeat(batch_size, 1).to(device=device)
    else:
        z_vals = 1 / torch.linspace(1/near, 1/far, N_samples).reshape(1, -1).repeat(batch_size, 1).to(device=device)

    # add some perturb if specified
    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])           # (B, N_samples-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)        # (B, N_samples)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)       # (B, N_samples)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)
        # compute all samples z_vals after adding perturb
        z_vals = lower + (upper - lower) * t_rand               # (B, N_samples)
    
    # (B, N_samples, 3), compute the final 3D coordinates of sample points
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]      # (B, N_sample, 3)
    return pts, z_vals



def refine_sampling(z_vals, weights, N_importance, perturb):
    """ Sample Points from Rays in this Batch
    Args:
        z_vals       - (B, N_samples), z values of all sample points
        weights      - (B, N_samples - 2), weights assigned to each sampled color
        N_importance - int, the number of more points we're gonna sample
        perturb      - bool, whether to add some noise to break linearity
    Returns:
        z_vals_new - (B, N_importance), z values of all sample points
    """
    z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])         # (B, N_samples-1)
    
    # get pdf(probability density function) and cdf(cumulative distribution function)
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # (B, N_samples-2)
    cdf = torch.cumsum(pdf, dim=-1)                             # (B, N_samples-2)
    cdf = torch.cat([torch.zeros((weights.shape[0], 1)).to(device), cdf], dim=-1)   # (B, N_samples-1)
    
    # take uniform samples or random samples between [0, 1], 是在 cdf 这张图的纵轴上采样
    if perturb:
        u = torch.rand((weights.shape[0], N_importance))        # (B, N_importance)
    else:
        u = torch.linspace(0., 1., steps=N_importance)          # (N_importance, )
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])     # (B, N_importance)
    
    # torch.contiguous(): 首先拷贝了一份 u 在内存中的地址，然后将地址按照形状改变之后的 Tensor 语义进行排列
    u = u.contiguous().to(device)
    # torch.searchsorted(): 返回一个和 u 一样大小，其中元素是 cdf 中大于等于 u 中值的列索引，cdf[..][i-1] <= u[...][x] < cdf[..][i]
    # inds: cdf 纵轴 [0, 1] 上将要采集的 N_importance 个采样点 -> 所对应的 cdf 横轴 N_samples-1 个采样点的 index
    inds = torch.searchsorted(cdf, u, right=True)                       # (B, N_importance)
    # below: cdf 横轴上除去自己(-1)以外有多少累计 weight 小于当前 inds 对应纵轴的 weight 的 coarse sample 点个数
    below = torch.max(torch.zeros_like(inds-1), inds-1)                 # (B, N_importance)
    # above: cdf 横轴上除去自己(-1)以外有多少累计 weight 小于当前 inds 对应纵轴的 weight 的 coarse sample 点个数
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # (B, N_importance)
    inds_g = torch.stack([below, above], -1)                            # (B, N_importance, 2)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]               # (B, N_importance, N_samples-1)
    # torch.gather(src, dim, idx), 根据 idx 取 src 中的指定数据，取的数据的 index 是除了 dim 维的 index 是 idx 中的数，别的都是正常排列
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)         # (B, N_importance, 2)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g) # (B, N_importance, 2)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])                             # (B, N_importance)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_new = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return z_vals_new



def raw2output(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """ Transform Raw NeRF Output to RGB Color and Weight
    Args:
        raw    - (B*N_samples, 4), raw rgb color and volume density of each sample point
        z_vals - (B, N_samples), z values of all sample points
        rays_d - (B, 3), directions of each ray in this batch
    Returns:
        color_map - (B, 3), RGB color of rays in this batch
        depth_map - (B, ), depth of rays in this batch
        dispa_map - (B, ), disparity of rays in this batch
        accum_map - (B, ), sum of weights along each ray
        weights   - (B, N_samples), weights assigned to each sampled color
    """
    # use sigmoid to squeeze the raw color output to [0, 1]
    raw = raw.reshape((z_vals.shape[0], z_vals.shape[1], -1))   # (B, N_samples, 4)
    rgb_color = torch.sigmoid(raw[:, :, :3])                    # (B, N_samples, 3)

    # use raw volume density sigma to compute volume transparency
    t_vals = z_vals[..., 1:] - z_vals[..., :-1]                                             # (B, N_samples-1)
    t_vals = torch.cat([t_vals, 1e10 * torch.ones(t_vals.shape[0], 1).to(device)], dim=-1)  # (B, N_samples)
    t_vals = t_vals * torch.norm(rays_d[:, None, :], dim=-1)
    # generate random noise to raw volume density if raw_noise_std
    noise = torch.randn(t_vals.shape).to(device=device) * raw_noise_std if raw_noise_std > 0 else 0
    alpha = 1. - torch.exp(-F.relu(raw[:, :, -1] + noise) * t_vals)                 # (B, N_samples)

    # use volume transparency alpha to compute volume weight (B, N_samples)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha+1e-10], dim=-1), dim=-1)[:, :-1]

    # get the final RGB color, depth map, dirsparity map
    color_map = torch.sum(weights[:, :, None] * rgb_color, dim=-2)  # (B, 3)
    depth_map = torch.sum(weights * z_vals, dim=-1)                 # (B, )
    dispa_map = 1. / torch.max(1e-10*torch.ones_like(depth_map), depth_map/torch.sum(weights, -1))  # (B, )
    accum_map = torch.sum(weights, dim=-1)                          # (B, )
    if white_bkgd: color_map = color_map + (1. - accum_map[:, None])

    return color_map, depth_map, dispa_map, accum_map, weights
