import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

from .utils import *



class ColorTransform(nn.Module):
    def __init__(self, color_downsample=True) -> None:
        super().__init__()
        self.color_downsample = color_downsample
        weight = torch.tensor([
                                [ 0.299000,  0.587000,  0.114000],
                                [-0.168736, -0.331264,  0.500000],
                                [ 0.500000, -0.418688, -0.081312]
                            ])

        bias = torch.tensor([0, 0.5, 0.5])
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("inverse_weight", weight.inverse())

    def forward(self, x):
        # (b, c, h, w) -> (b, c, h, w)
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x):
        x = (x.permute(0, 2, 3, 1) @ self.weight.T.to(x.device) + self.bias.to(x.device)).permute(0, 3, 1, 2)
        if self.color_downsample:
            colors = x[:, 1:, :, :]
            colors = F.interpolate(colors, scale_factor=0.5, mode="bilinear")
            colors = F.interpolate(colors, scale_factor=2, mode="nearest")
        x[:, 1:, :, :] = colors
        return x

    @torch.no_grad()
    def decode(self, x, channel_last=True):
        # (b, c, h, w) -> (b, c, h, w) or (b, h, w, c) if set channel_last
        x = ((x.permute(0, 2, 3, 1) - self.bias.to(x.device)) @ self.inverse_weight.T.to(x.device)).clamp_(0, 255).to(torch.uint8) 
        return x if channel_last else x.permute(0, 3, 1, 2)


class ZigzagTransform(nn.Module):
    def __init__(self, block_size=8) -> None:
        super().__init__()
        self.block_size = block_size
        zigzag_vector = zigzag(block_size).view(-1).to(torch.long)
        weight = F.one_hot(zigzag_vector).to(torch.float)[:, :, None, None]  # (N ** 2, N ** 2, 1, 1)
        self.register_buffer("weight", weight)
    
    def forward(self, x):
        # (b, c, h, w) -> (b, c, h, w) with c dim zigzaged.
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x):
        c = x.size(1)
        if c != self.block_size ** 2:
            x = rearrange(x, "b (n c) h w -> (b n) c h w", n=3)
        x = F.conv2d(x, self.weight.permute(1, 0, 2, 3).to(x.device))
        x = rearrange(x, "(b n) c h w -> b (n c) h w", n=3)
        return x

    @torch.no_grad()
    def decode(self, x):
        # (b, c, h, w) -> (b, c, h, w) with c dim back to normal order.
        c = x.size(1)
        if c != self.block_size ** 2:
            x = rearrange(x, "b (n c) h w -> (b n) c h w", n=3)
        x = F.conv2d(x.to(torch.float), self.weight.to(x.device))
        x = rearrange(x, "(b n) c h w -> b (n c) h w", n=3)
        return x
    
    
class DctEncoder(nn.Module):
    def __init__(self, block_size=8) -> None:
        # https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example
        super().__init__()
        self.block_size = block_size

        axis = torch.arange(block_size)
        alpha = torch.ones(block_size)
        alpha[0] = 1 / math.sqrt(2)

        encode_weight = torch.cos((axis[None, None, None, :] + 0.5) * axis[None, :, None, None] * pi / block_size)
        encode_weight = encode_weight * torch.cos((axis[None, None, :, None] + 0.5) * axis[:, None, None, None] * pi / block_size)
        encode_weight = encode_weight * alpha[:, None, None, None] * alpha[None, :, None, None] / 4    # (N_v, N_u, N_y, N_x)

        self.register_buffer("encode_weight", encode_weight.view(block_size * block_size, 1, block_size, block_size))

        decode_weight = torch.cos((axis[None, :, None, None] + 0.5) * axis[None, None, None, :] * pi / block_size)
        decode_weight = decode_weight * torch.cos((axis[:, None, None, None] + 0.5) * axis[None, None, :, None] * pi / block_size)
        decode_weight = decode_weight * alpha[None, None, :, None] * alpha[None, None, None, :] / 4    # (N_y, N_x, N_v, N_u)

        self.register_buffer("decode_weight", decode_weight.view(block_size * block_size, 1, block_size, block_size))

    def forward(self, x):
        # (b, 1, h, w) -> (b, n * n, h / n, w / n)
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x):
        c = x.size(1)
        if c != 1:
            x = rearrange(x, "b c h w -> (b c) 1 h w")
        x -= 128.   # zero centered
        y = F.conv2d(x, self.encode_weight.to(x.device), stride=self.block_size)    # conv2d has a bug with cpu on xavier jetson device: https://github.com/pytorch/pytorch/issues/59439
        return y

    @torch.no_grad()
    def decode(self, x):
        c = x.size(1)
        if c == 3 * self.block_size ** 2:
            x = rearrange(x, "b (n v u) h w -> (b n) 1 (h v) (w u)", n=3, v=self.block_size, u=self.block_size)
        y = F.conv2d(x, self.decode_weight.to(x.device), stride=self.block_size) + 128.
        # y.clamp_(0, 255)
        y = rearrange(y, "(b n) (v u) h w -> b n (h v) (w u)", n=3, v=self.block_size, u=self.block_size)
        return y

    def patterns(self):
        weight = self.encode_weight * 2 + 0.5
        grid = torchvision.utils.make_grid(weight, nrows=self.block_size, padding=1)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()


class Quantization(nn.Module):
    def __init__(self, block_size=8, q=50) -> None:
        super().__init__()
        self.block_size = block_size
        self.q = q
        quant_mat = torch.cat([Q_luma(q, block_size).view(-1), Q_chroma(block_size).view(-1), Q_chroma(block_size).view(-1)], dim=0)
        self.register_buffer("quant_mat", quant_mat[None, :, None, None])

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x):
        c = x.size(1)
        if c == self.block_size ** 2:
            x = rearrange(x, "(b n) c h w -> b (n c) h w", n=3)
        return torch.round(x / self.quant_mat.to(x.device))

    @torch.no_grad()
    def decode(self, x):
        c = x.size(1)
        if c == self.block_size ** 2:
            x = rearrange(x, "(b n) c h w -> b (n c) h w", n=3)
        x = x * self.quant_mat.to(x.device)
        return x


class SparseEncoder(nn.Module):
    def __init__(self, block_size=8, interleave=True) -> None:
        super().__init__()
        self.block_size = block_size
        self.interleave = interleave

        self.encode_size = None

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x, append_eos=True):
        # returns a list of tensors and each tensor is (3, num_of_non-zero)

        _, c, h, w = x.shape
        self.encode_size = (c, h, w)    # just record the size of dct image for reconstruction

        if self.interleave:
            x = rearrange(x, "b (n c) h w -> b (c n) (h w)", n=3)
        else:
            x = rearrange(x, "b c h w -> b c (h w)")

        sparsed_batch = []

        for ele in x:
            sparse = ele.to_sparse()
            ind = sparse._indices() + 3 # shift index by 3, and 0 is for padding, 1 is for sos, 2 is for eos
            val = sparse._values() + 16 * self.block_size ** 2 + 3  # dct value is in [-16 * N ** 2, 16 * N ** 2]

            sparse = torch.cat([ind, val[None, :].to(torch.long)], dim=0)

            if append_eos:
                sparse = F.pad(sparse, (0, 1), mode="constant", value=2)
            sparsed_batch.append(sparse)
                
        return sparsed_batch

    @torch.no_grad()
    def decode(self, sparsed_batch, decode_size=None):
        decode_size = decode_size or self.encode_size
        assert decode_size, "Decode size cannot be inferred."
        c, h, w = decode_size
        dense_batch = []
        for ele in sparsed_batch:
            mask = (ele[-1] != 2) & (ele[-1] != 0)
            ele = ele[:, mask]  # remove eos and padding
            ind = ele[:2] - 3
            val = ele[-1] - 16 * self.block_size ** 2 - 3
            dense = torch.sparse_coo_tensor(ind, val, (c, h * w)).to_dense().view(c, h, w)
            dense_batch.append(dense)
        x = torch.stack(dense_batch)    # b (c n) h w
        if self.interleave:
            x = rearrange(x, "b (c n) h w -> b (n c) h w", n=3)
        return x
        

class DctCompress(nn.Module):
    def __init__(self, block_size=8, q=50, interleave=True) -> None:
        super().__init__()
        self.block_size = block_size
        self.q = q
        self.interleave = interleave

        self.color_transform = ColorTransform()
        self.dct_encoder = DctEncoder(block_size)
        self.quantization = Quantization(block_size, q)
        self.zigzag = ZigzagTransform(block_size)
        self.sparse_encoder = SparseEncoder(block_size, interleave)

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def encode(self, x):
        # (b, c, h, w) -> list[(chn, pos, val)]
        x = self.color_transform(x)
        x = self.dct_encoder(x)
        x = self.quantization(x)
        x = self.zigzag(x)
        x = self.sparse_encoder(x)
        return x 

    @torch.no_grad()
    def decode(self, x, channel_last=True):
        x = self.sparse_encoder.decode(x)
        x = self.zigzag.decode(x)
        x = self.quantization.decode(x)
        x = self.dct_encoder.decode(x)
        x = self.color_transform.decode(x, channel_last)
        return x

    @torch.no_grad()
    def prior_to_img(self, x):
        x = self.sparse_encoder(x)
        x = self.decode(x)
        return x