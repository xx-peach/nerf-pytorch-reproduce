import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedder import PositionalEncoding, get_embedder


class NeRF(nn.Module):
    """
    NeRF Model Architecture Described in the Paper with Some Modification
    """
    def __init__(
        self, d=8, w=256,
        log2_max_freq=10,
        log2_max_freq_view=4,
        i_embed=True,
        skip=[4],
        use_viewdirs=True
    ) -> None:
        """ Instantiate all the Layers and Parameters Used in a NeRF Model
        Args:
            d, w               - the depths and widths(channel) of layers in the NeRF MLP
            log2_max_freq      - log2 of max freq for positional encoding (3D location)
            log2_max_freq_view - log2 of max freq for positional encoding (2D direction)
            i_embed            - set 0 for default positional encoding, -1 for none
            skip               - for fc layer i in skips, its channel conf for images is (W+input_ch, W), 就是一个 residual block
            use_viewdirs       - whether to use view directions as input
        """
        super(NeRF, self).__init__()
        # accept all the input configurations
        self.d, self.w = d, w
        self.log2_max_freq = log2_max_freq
        self.log2_max_freq_view = log2_max_freq_view
        self.i_embed = i_embed
        self.skip = skip
        self.use_viewdirs = use_viewdirs
        
        # create embedding layers, official embedder
        # my own implementation, sin altogether and cos altogether, but more fast
        self.coor_embedder, self.view_embedder = PositionalEncoding(log2_max_freq, i_embed), PositionalEncoding(log2_max_freq_view, i_embed)
        self.input_ch, self.input_ch_view = self.coor_embedder.output_channel, self.view_embedder.output_channel
        # # implementation in official code (https://github.com/bmild/nerf/master/run_nerf_helper.py#L104-L105)
        # self.coor_embedder, self.input_ch      = get_embedder(log2_max_freq, i_embed)
        # self.view_embedder, self.input_ch_view = get_embedder(log2_max_freq_view, i_embed)
        
        # create layer list for sample points, namely 3D points coordinates
        self.coor_mlp = nn.ModuleList(
            [nn.Linear(self.input_ch, w)] + [nn.Linear(self.input_ch + w, w) if i in skip else nn.Linear(w, w) for i in range(d-1)]
        )
        
        # create layer list for view directions
        # implementation in official code (https://github.com/bmild/nerf/master/run_nerf_helper.py#L104-L105)
        self.view_mlp = nn.ModuleList([nn.Linear(self.input_ch_view + w, w//2)])
        # # implementation according to the paper
        # self.view_mlp = nn.ModuleList(
        #     [nn.Linear(input_ch_view + w, w//2)] + [nn.Linear(w//2, w//2) for i in range(d//2)]
        # )
        
        # create layer for rgb and volume density prediction
        if use_viewdirs:
            self.feature_linear = nn.Linear(w, w)   # fc layer for a feature map that is gonna to combine with view
            self.sigma_linear = nn.Linear(w, 1)     # fc layer for volume density
            self.rgb_linear = nn.Linear(w//2, 3)    # fc layer for rgb color
        else:
            self.outpu_linear = nn.Linear(w, 4)


    def forward(self, x, d):
        """ NeRF Model's forward() Function
        Args:
            x - (B * N_samples, 3), input sample points' coordinates
            d - (B * N_samples, 3), input view directions
        """
        # perform positional embedding
        x_embed = self.coor_embedder(x)     # (B*N_samples, input_ch), embedded 3D locations
        d_embed = self.view_embedder(d)     # (B*N_samples, input_ch_view), embedded view directions
        # pass x_embed through coor_mlp
        x_opt = x_embed
        for i, fc in enumerate(self.coor_mlp):
            x_opt = F.relu(fc(x_opt))
            if i in self.skip: x_opt = torch.cat([x_opt, x_embed], dim=-1)
        # pass view directions through view_mlp if use_viewdirs
        if self.use_viewdirs:
            # get output volume density
            sigma = self.sigma_linear(x_opt)
            # go through view_mlp fully-connected layers
            feature = self.feature_linear(x_opt)
            x_opt = torch.cat([feature, d_embed], dim=-1)
            for fc in self.view_mlp:
                x_opt = F.relu(fc(x_opt))
            # get output rgb color
            rgb = self.rgb_linear(x_opt)
            output = torch.cat([rgb, sigma], dim=-1)
        else:
            output = self.outpu_linear(x_opt)

        return output
