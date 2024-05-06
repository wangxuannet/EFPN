

import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU(),
                 norm_layer=nn.LayerNorm):
        super(AttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))

            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        self.patch_embeds = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        attention_depths = [3, 3, 3, 3]
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            patch_embed = nn.Sequential(
                PatchEmbed(img_size=7*(2**i), patch_size=2**i, in_c=dims[3-i], embed_dim=dims[3-i])
            )

            attention_block = nn.Sequential(
                *[AttentionBlock(dim=dims[3-i], num_heads=12,   mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                                 act_layer=nn.GELU(), norm_layer=nn.LayerNorm)
                  for j in range(attention_depths[i])]
            )

            self.attention_blocks.append(attention_block)
            self.patch_embeds.append(patch_embed)
            self.stages.append(stage)
            cur += depths[i]

        self.norm_max = nn.LayerNorm(dims[-1]+dims[-2]+dims[-3]+dims[-4], eps=1e-6)
        self.norm1 = nn.LayerNorm(dims[3], eps=1e-6)
        self.norm2 = nn.LayerNorm(dims[2], eps=1e-6)
        self.norm3 = nn.LayerNorm(dims[1], eps=1e-6)
        self.norm4 = nn.LayerNorm(dims[0], eps=1e-6)

        self.fc = nn.Linear(in_features=dims[-1], out_features=int(dims[-1]/2))

        self.cls_token3 = nn.Parameter(torch.randn(1, 1, dims[2]))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dims[1]))
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dims[0]))

        self.head1 = nn.Linear(dims[-1], dims[-1])
        self.head2 = nn.Linear(dims[-2], dims[-2])
        self.head3 = nn.Linear(dims[-3], dims[-3])
        self.head3 = nn.Linear(dims[-4], dims[-4])
        self.head = nn.Linear(dims[-1]+dims[-2]+dims[-3]+dims[-4], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor):
        x = self.downsample_layers[0](x)
        B = x.shape[0]
        x1 = self.stages[0](x)
        x2 = self.downsample_layers[1](x1)
        x2 = self.stages[1](x2)
        x3 = self.downsample_layers[2](x2)
        x3 = self.stages[2](x3)
        x4 = self.downsample_layers[3](x3)
        x4 = self.stages[3](x4)
        r4 = self.norm1(x4.mean([-2, -1]))

        r3 = self.patch_embeds[1](x3)
        cls_tokens3 = self.cls_token3.expand(B, -1, -1).cuda()
        r3 = torch.cat((cls_tokens3, r3), dim=1)
        r3 = self.attention_blocks[1](r3)
        r3 = self.norm2(r3)[:, 0]

        r2 = self.patch_embeds[2](x2)
        cls_tokens2 = self.cls_token2.expand(B, -1, -1).cuda()
        r2 = torch.cat((cls_tokens2, r2), dim=1)
        r2 = self.attention_blocks[2](r2)
        r2 = self.norm3(r2)[:, 0]

        r1 = self.patch_embeds[3](x1)
        cls_tokens1 = self.cls_token1.expand(B, -1, -1).cuda()
        r1 = torch.cat((cls_tokens1, r1), dim=1)
        r1 = self.attention_blocks[3](r1)
        r1 = self.norm4(r1)[:, 0]
        # r3 = self.norm2(x3.mean([-2, -1]))
        # r2 = self.norm3(x2.mean([-2, -1]))
        # r1 = self.norm4(x1.mean([-2, -1]))

        return r1, r2, r3, r4

    def forward(self, x: torch.Tensor):
        r1, r2, r3, r4 = self.forward_features(x)
        # x = self.norm_max(torch.cat([r1, r2, r3, r4], dim=-1))
        # x = self.head(x)
        # return x
        return r1, r2, r3, r4


def convnext_tiny(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class CONV_BLOCK(nn.Module):
    def __init__(self, dim=512):
        super(CONV_BLOCK, self).__init__()
        self.proj = Block(dim=dim, drop_rate=0, layer_scale_init_value=1e-6)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim):
        super(TransMIL, self).__init__()
        self.pos_layer = CONV_BLOCK(dim=int(dim/2))
        self._fc1 = nn.Sequential(nn.Linear(dim, int(dim/2)), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, int(dim/2)))
        self.layer1 = TransLayer(dim=int(dim/2))
        self.layer2 = TransLayer(dim=int(dim/2))
        self.norm = nn.LayerNorm(int(dim/2))

    def forward(self, h):
        h = h.unsqueeze(0)
        # h = kwargs['data'].float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)

        return h


class Transformation(nn.Module):
    def __init__(self, dim):
        super(Transformation, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, int(dim)))

    def forward(self, h):
        h = h.unsqueeze(0)
        # h = kwargs['data'].float()  # [B, n, 1024]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        return h


class PyramidMIL(nn.Module):
    def __init__(self, n_classes):
        super(PyramidMIL, self).__init__()
        self.num_classes = n_classes
        self.dims = [96, 192, 384, 768]
        self.fcs = nn.ModuleList()
        self.convnext = convnext_tiny(n_classes)
        self.ppeg = PPEG(dim=self.dims[0]+self.dims[1]+self.dims[2])
        self.transmil1 = TransMIL(self.dims[-1])
        self.transmil2 = TransMIL(self.dims[-2])
        self.transmil3 = TransMIL(self.dims[-3])
        self.transmil4 = TransMIL(self.dims[-4])

        self.transformation1 = Transformation(self.dims[0])
        self.transformation2 = Transformation(self.dims[1])
        self.transformation3 = Transformation(self.dims[2])

        self._head = nn.Linear(self.dims[-2] + self.dims[-3] + self.dims[-4], n_classes)
        self._head1 = nn.Linear(self.dims[-2], self.dims[-2])
        self._head2 = nn.Linear(self.dims[-3], self.dims[-3])
        self._head3 = nn.Linear(self.dims[-4], self.dims[-4])

        self.norm1 = nn.LayerNorm(self.dims[3], eps=1e-6)
        self.norm2 = nn.LayerNorm(self.dims[2], eps=1e-6)
        self.norm3 = nn.LayerNorm(self.dims[1], eps=1e-6)
        self.norm4 = nn.LayerNorm(self.dims[0], eps=1e-6)
        for i in range(3):
            fc = nn.Sequential(
                nn.Linear(in_features=self.dims[3 - i], out_features=(self.dims[2 - i]))
            )
            self.fcs.append(fc)

        self.M = self.dims[0]+self.dims[1]+self.dims[2]+self.dims[2]
        self.L = self.dims[0]+self.dims[1]+self.dims[2]+self.dims[2]
        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, n_classes), )

    def forward(self, h):
        h = h.squeeze(0)
        x1, x2, x3, x4 = self.convnext(h)
        for param in self.convnext.parameters():
            param.requires_grad = False
        x1 = self.transformation1(x1)
        x2 = self.transformation2(x2)
        x3 = self.transformation3(x3)

        y0 = self.transmil1(x4)  # 768-384
        y0 = y0[:, 1:]

        y1 = torch.cat([y0, x3], dim=-1)  # 384-768
        y1 = y1.squeeze(0)
        y1 = self.transmil1(y1)  # 768-384
        r1 = y1[:, 0]
        y1 = y1[:, 1:]
        r1 = self._head1(r1)

        y2 = self.fcs[1](y1)   # 384-192
        y2 = torch.cat([y2, x2], dim=-1)  # 192-384
        y2 = y2.squeeze(0)
        y2 = self.transmil2(y2)  # 384-192
        r2 = y2[:, 0]
        y2 = y2[:, 1:]
        r2 = self._head2(r2)

        y3 = self.fcs[2](y2)
        y3 = torch.cat([y3, x1], dim=-1)
        y3 = y3.squeeze(0)
        y3 = self.transmil3(y3)
        r3 = y3[:, 0]
        y3 = y3[:, 1:]
        r3 = self._head3(r3)

        out = torch.cat([y0, y1, y2, y3], dim=-1)
        out = out.squeeze(0)

        # out = self.gamma * out + x

        A = self.attention(out)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, out)
        # ---->predict
        logits = self.classifier(Z)  # [B, n_classes]
        return logits
