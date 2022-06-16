import torch
import torch.nn as nn
from mm_modules.DCN.modules.deform_conv2d import DeformConv2dPack
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

def conv_flops(k, c_in, c_out, stride, padding, resolution, bias=True, dialation=1):
    batch_size = 1
    output_dims = math.floor((resolution + 2*padding - dialation*(k - 1) - 1) / stride + 1)
    kernel_dims = k
    in_channels = c_in
    out_channels = c_out
    groups = 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(kernel_dims**2) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(output_dims**2)

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DTM(nn.Module):
    r""" Deformable Token Merging.

    Link: https://arxiv.org/abs/2105.14217

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.c_in = dim
        self.c_out = dim*2
        self.dconv = DeformConv2dPack(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.norm_layer = nn.BatchNorm2d(dim*2)
        self.act_layer = nn.GELU()

    def forward(self, x, return_offset=False):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x, offset = self.dconv(x, return_offset=False)
        x = self.act_layer(self.norm_layer(x)).flatten(2).transpose(1, 2)
        if return_offset:
            return x, offset
        else:
            return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        offset_flops = conv_flops(self.kernel_size, self.c_in, 12, self.stride, self.padding, H)
        dconv_flops = conv_flops(self.kernel_size, self.c_in, self.c_out, self.stride, self.padding, H)
        deformable_flops = offset_flops + dconv_flops
        # norm layer
        norm_flops = (H // 2) * (W // 2) * 2 * self.dim

        return deformable_flops + norm_flops
