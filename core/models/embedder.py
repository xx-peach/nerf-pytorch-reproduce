import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")


class PositionalEncoding(nn.Module):
    def __init__(self, log2_max_freq, i_embed, log_sampling=True):
        """ Construct the Positional Encoding Layer
        Args:
            log2_max_freq - log2 of max freq for positional encoding (3D location)
            i_embed       - set 0 for default positional encoding, -1 for none
            log_sampling  - log sampling or not
        """
        super().__init__()
        self.log2_max_freq = log2_max_freq
        self.i_embed = i_embed
        self.log_sampling = log_sampling
        
        # generate different frequencies
        if self.log_sampling:
            freqs = 2.**np.linspace(0, self.log2_max_freq-1, self.log2_max_freq)        # (L, )
        else:
            freqs = np.linspace(2.**0, 2.**(self.log2_max_freq-1), self.log2_max_freq)  # (L, )
        # repeat the frequencies 3 times one by one, (1, 3L)
        self.freqs = torch.from_numpy(freqs.repeat(3).reshape(1, -1)).type(torch.float32).to(device)
        
        # return output embedding dimension
        self.output_channel = 3 if self.i_embed == -1 else 3 + 2*3*self.log2_max_freq

        # # generate indices for sin and cos repectively, (3L, )
        # self.sin_idx = torch.LongTensor([[i, i+1, i+2] for i in range(3, 27, 6)]).ravel()
        # self.cos_idx = torch.LongTensor([[i, i+1, i+2] for i in range(6, 27, 6)]).ravel()


    def forward(self, x):
        """ Element-wise Positional Embeddings to the Input pts or viewdirs.

        Args:
            x - (B*N_samples, 3), original pts or viewdirs
        Returns:
            h - (B*N_samples, 3L+3), the positional encodings
        """
        if self.i_embed == -1:
            return x
        h = x.repeat(1, self.log2_max_freq)         # (B, L)
        h = h * self.freqs
        h = torch.cat([x, torch.sin(h), torch.cos(h)], dim=1)
        return h


def get_embedder(log2_max_freq, i_embed):
    """ Public API for Getting the Positional Embedding
    Args:
        multires - log2 of max freq for positional encoding
        i        - set 0 for default positional encoding, -1 for none
    Returns:
        embedder - the embedded layer, just give it input images, and we can get the positional-encoded images
        out_dims - the output dimension after the positional encoding
    """
    if i_embed == -1:
        return nn.Identity(), 3
    kwargs = {
        "include_input": True,
        "input_dims": 3,
        "log2_max_freq": log2_max_freq-1,
        "num_freqs": log2_max_freq,
        "log_sampling": True,
        "period_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**kwargs)
    embedder = lambda x, em=embedder_obj: em.embed(x)
    return embedder, embedder_obj.out_dims


class Embedder():
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.embedders_list, self.out_dims = self.create_embed()

    def create_embed(self):
        out_dims = 0
        embedders_list = []

        # add the original input using lambda function
        if self.kwargs['input_dims']:
            embedders_list.append(lambda x: x)
            out_dims = out_dims + 3

        # generate all the frequencies
        if self.kwargs['log_sampling']:
            freqs_list = 2.**torch.linspace(0, self.kwargs['log2_max_freq'], self.kwargs['num_freqs'])
        else:
            freqs_list = torch.linspace(2.**0, 2.**self.kwargs['log2_max_freq'], self.kwargs['num_freqs'])

        # add all the freuencies embedder to the embedder list
        for freq in freqs_list:
            for perd in self.kwargs['period_fns']:
                embedders_list.append(lambda x, f=freq, p=perd: p(x*f))
                out_dims = out_dims + 3
        
        return embedders_list, out_dims

    def embed(self, x):
        """ Concatenate the Positional Encoding for each Frequency
        Returns:
            [(batch_size, input_dims), ..., (batch_size, input_dims)] -> (batch_size, out_dim)
        """
        return torch.cat([em(x) for em in self.embedders_list], dim=-1)
