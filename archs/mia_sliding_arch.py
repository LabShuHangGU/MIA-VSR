import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from functools import lru_cache
from einops import rearrange, repeat
from .functions import batch_index_fill, batch_index_select
class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):#true
        if not self.training:  # no Gumbel noise during inference
            return (torch.sigmoid(x / gumbel_temp) >= 0.5).float()

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        #assert not torch.any(torch.isnan(hard))
        return hard
class AdaptiveMask(nn.Module):
    def __init__(self, dim):
        super(AdaptiveMask, self).__init__()
        self.rnn = nn.Linear(dim, 1)#1
        self.gumbel = Gumbel()

    def forward(self, x):
        soft = self.rnn(x)
        hard = self.gumbel(soft, gumbel_temp=2/3)
        if self.training: 
            maskloss = (hard.sum()/hard.numel()).float()  
            return hard, maskloss
        else:
            score= hard.reshape(1,-1)
            num_keep_node = int(hard.sum())
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return hard, [idx1, idx2]
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  C)
    return windows

def window_single_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[1] * window_size[2], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def window_reverse_single(windows, window_size, B, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, H // window_size[1], W // window_size[2], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def Pred(x):
    score= x.reshape(1,-1)
    num_keep_node = int(x.sum())
    idx = torch.argsort(score, dim=1, descending=True)
    idx1 = idx[:, :num_keep_node]
    idx2 = idx[:, num_keep_node:]
    return [idx1, idx2]



class MaskWindowAttention(nn.Module):
    r""" Mask IIAB Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,use_mask=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = qk_scale or head_dim**-0.5
        self.predmask = AdaptiveMask(dim)
        self.use_mask=use_mask
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0]) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_d = torch.arange(self.window_size[0])
        coords_1 = torch.arange(1)
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords_2 = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_1 = torch.stack(torch.meshgrid(coords_1, coords_h, coords_w))  # 3, 1, Wh, Ww
        coords_flatten_1 = torch.flatten(coords_1, 1)  # 3, 1*Wh*Ww
        coords_flatten_2 = torch.flatten(coords_2, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # 3, Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        #self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,shift_size, quary=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b,t, h,w,c = x.shape
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        # partition windows
        attn = window_partition(x, self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c
        B_, N, C = attn.shape
        pred_q = attn[:, 2*(N//3):N, :].detach()
        pred = torch.ones(B_, N//3, 1).cuda()
        maskloss = None
        self.ratio_k = 1
        if self.use_mask and quary is not None:
            pred = torch.abs(pred_q - quary[0])
            if self.training:
                pred, maskloss = self.predmask(pred)
                self.ratio_k = maskloss
            else:
                pred, mask_idx = self.predmask(pred)
                idx1, idx2 = mask_idx
                self.ratio_k = idx1.numel()/(idx1.numel()+idx2.numel())
        qkv = self.qkv(attn)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        quary_q = qkv[0]
        quary_q = quary_q[:,:,2*(N//3):N,:]
        k = qkv[1]
        v = qkv[2]
        quary_q = quary_q * self.scale
        attn = quary_q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N//3, :N].reshape(
            -1)].reshape(N//3, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N//3, N) + mask[:,:,:,2*(N//3):N,:]
            attn = attn.view(-1, self.num_heads, N//3, N)        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N//3, c)
        if self.training:
            x = self.proj(x)
            x_pred = self.proj_drop(x)
            if self.use_mask and quary is not None:
                x_pred = x_pred*pred + quary[1]*(1-pred)
            # merge windows
            x = torch.cat((x_pred,pred), -1)
            x = x.view(-1, self.window_size[1], self.window_size[2], c+1)
            x = window_reverse_single(x, self.window_size, b, h, w)  # B D' H' W' C+1
        
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x = torch.roll(
                    x, shifts=(shift_size[1], shift_size[2]), dims=(1, 2))
            return x, [pred_q, x_pred.detach()], maskloss
        elif self.use_mask and quary is not None:  
            x = rearrange(x,'b n c-> 1 (b n) c')         
            x1 = batch_index_select(x, idx1)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)
            quary[1] = rearrange(quary[1],'b n c-> 1 (b n) c')
            x2 = batch_index_select(quary[1], idx2)
            x0 = torch.zeros_like(x)
            x_pred = batch_index_fill(x0, x1, x2, idx1, idx2)
            x_pred = rearrange(x_pred,'1 (b n) c-> b n c',n=N//3)
            x = torch.cat((x_pred,pred), -1)
            x = x.view(-1, self.window_size[1], self.window_size[2], c+1)
            x = window_reverse_single(x, self.window_size, b, h, w)  # B D' H' W' C+1
        
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x = torch.roll(
                    x, shifts=(shift_size[1], shift_size[2]), dims=(1, 2))
            return x, [pred_q, x_pred]
        else:
            x = self.proj(x)
            x_pred = self.proj_drop(x)
            # merge windows
            x = torch.cat((x_pred,pred), -1)
            x = x.view(-1, self.window_size[1], self.window_size[2], c+1)
            x = window_reverse_single(x, self.window_size, b, h, w)  # B D' H' W' C+1
        
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x = torch.roll(
                    x, shifts=(shift_size[1], shift_size[2]), dims=(1, 2))
            return x, [pred_q, x_pred]
      

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        flops = 0
        flops += 8*(n//3) * self.dim * self.dim
        if self.use_mask:
            flops += self.num_heads * (n//3) * (self.dim // self.num_heads) * n * (self.ratio_k)
            flops += (n//3) * self.dim * self.dim* (self.ratio_k)
        else:
            flops += self.num_heads * (n//3) * (self.dim // self.num_heads) * n
            flops += (n//3) * self.dim * self.dim
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        
        return flops



        


class MaskSwinTransformerBlock(nn.Module):
    r""" Mask IIAB Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_frames=5,
                 use_mask=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.use_mask = use_mask
        self.norm1 = norm_layer(dim)
        self.attn = MaskWindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_mask=use_mask)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, cond, attn_mask, lastq, recurrent):
        x = torch.cat((cond, x.unsqueeze(1)), dim=1).contiguous()
        b, t, h, w, c = x.shape

        shortcut = x[:,-1,:,:,:].contiguous()

        if recurrent:
            last_quary = lastq.pop(0)
        else:
            last_quary = None

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        
        if self.training:
            if any(i > 0 for i in self.shift_size):
                x, q, maskloss = self.attn(x,self.shift_size, last_quary, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
            else:
                x, q, maskloss = self.attn(x,self.shift_size, last_quary, mask=None)
        
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()
            mask_pred = x[:, :, :, -1].unsqueeze(-1).contiguous()
            x = shortcut + self.drop_path(x[:, :, :, :-1].contiguous())
            if self.use_mask:
                if recurrent:     
                    hidden = self.drop_path(self.mlp(self.norm2(x)))*mask_pred + last_quary[2]*(1-mask_pred)
                    self.ratio_k = maskloss
                else:
                    hidden = self.drop_path(self.mlp(self.norm2(x)))
                    self.ratio_k = 1
            else:
                hidden = self.drop_path(self.mlp(self.norm2(x)))
            x = x + hidden
            q.append(hidden.detach())
            lastq.append(q)
            
            return x, lastq, maskloss
        else:
            if any(i > 0 for i in self.shift_size):
                x, q = self.attn(x,self.shift_size, last_quary, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
            else:
                x, q = self.attn(x,self.shift_size, last_quary, mask=None)
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()
            mask_pred = x[:, :, :, -1].contiguous()
            x = shortcut + self.drop_path(x[:, :, :, :-1].contiguous())
            if self.use_mask:
                if recurrent: 
                    xh = self.norm2(x)
                    idx1, idx2 = Pred(mask_pred)
                    xh = xh.view(1,h*w,c)
                    x1 = batch_index_select(xh, idx1)
                    x1 = self.mlp(x1)
                    last_quary[2] = last_quary[2].view(1,h*w,c)
                    x2 = batch_index_select(last_quary[2], idx2)
                    x0 = torch.zeros_like(xh)
                    hidden = batch_index_fill(x0, x1, x2, idx1, idx2)
                    hidden = hidden.view(1,h,w,c)
                    self.ratio_k = (mask_pred.sum()/mask_pred.numel()).float()  
                else:
                    hidden = self.drop_path(self.mlp(self.norm2(x)))
                    self.ratio_k = 1
            else:
                hidden = self.drop_path(self.mlp(self.norm2(x)))
            x = x + hidden 
            q.append(hidden.detach())
            lastq.append(q)
            return x, lastq, None

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w * self.num_frames
        # W-MSA/SW-MSA
        nw = h * w / self.window_size[1] / self.window_size[2]
        flops += nw * self.attn.flops(self.window_size[1] * self.window_size[2] * self.num_frames)
        print(flops)
        if self.use_mask:
            # mlp
            flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio*self.ratio_k
            # norm2
            flops += self.dim * h * w*self.ratio_k
        else:
            # mlp
            flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio 
            # norm2
            flops += self.dim * h * w
        print(flops)
        return flops




class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_frames=5,
                 use_mask = True
                 ):

        super().__init__()
        self.dim = dim  
        self.input_resolution = input_resolution  #64,64
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_mask = use_mask

        # build blocks
        self.blocks = nn.ModuleList([
            MaskSwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0], window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_frames=num_frames,
                use_mask = self.use_mask) for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x, cond, attn_mask, lastquary, predmask, recurrent=True):
        #x (b,c,t,h,w)
    
        for blk_idx, blk in enumerate(self.blocks):      
            if self.use_checkpoint:
                x, lastquary, pred = checkpoint.checkpoint(blk, x, cond, attn_mask, lastquary, recurrent)
                predmask.append(pred)
            
            else:
                x, lastquary, pred = blk(x, cond, attn_mask, lastquary, recurrent)
                predmask.append(pred)


        if self.downsample is not None:
            x = self.downsample(x)

        x = x.permute(0, 3, 1, 2).contiguous()
          #b,c,h,w
        
        return x, lastquary, predmask

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            #print("swinlayer",blk.flops()/1e9)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Multi-frame Self-attention Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=(1, 1),
                 resi_connection='1conv',
                 num_frames=5,
                 use_mask=True
                ):
        super(RSTB, self).__init__()

        self.dim = dim  
        self.input_resolution = input_resolution  
        self.num_frames=num_frames
        self.use_mask = use_mask
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            num_frames=num_frames,
            use_mask=self.use_mask)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, cond, attn_mask, lastquary, predmask, recurrent=True):
        #n, c, h, w = x.shape
        x_ori = x
        x, lastquary, predmask = self.residual_group(x, cond, attn_mask, lastquary, predmask, recurrent=recurrent)
        x = self.conv(x)
        x = x.permute(0,2,3,1)

        x = x + x_ori
        return x, lastquary, predmask

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.num_frames * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        #flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, num_frames=5, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.num_frames = num_frames

        self.in_chans = in_chans  
        self.embed_dim = embed_dim  

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #x.size = (n,t,embed_dim,h,w)
        n, t, c, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, t, h, w)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim *self.num_frames
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@lru_cache()
def compute_mask(t, x_size, window_size, shift_size, device):
    h, w = x_size
    Dp = int(np.ceil(t / window_size[0])) * window_size[0]
    Hp = int(np.ceil(h / window_size[1])) * window_size[1]
    Wp = int(np.ceil(w / window_size[2])) * window_size[2]
    img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)  # 1 h w 1
    
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    #mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

@lru_cache()
def compute_single_mask(t, x_size, window_size, shift_size, device):
    h, w = x_size
    Hp = int(np.ceil(h / window_size[1])) * window_size[1]
    Wp = int(np.ceil(w / window_size[2])) * window_size[2]
    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)  # 1 h w 1
    cnt = 0
    for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
        for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_single_partition(img_mask, window_size)  # nW, ws[1]*ws[2], 1
    mask_windows = mask_windows.view(-1, window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    attn_mask = attn_mask.repeat(1,3)
    return attn_mask
class SwinIRFM(nn.Module):
    r""" SwinIRFM
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        num_frames: The number of frames processed in the propagation block in PSRT-recurrent
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 num_frames=3,
                 use_mask=True,
                 **kwargs):
        super(SwinIRFM, self).__init__()
        num_in_ch = in_chans  #3
        num_out_ch = in_chans  #3
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.window_size = window_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.num_frames = num_frames

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_feat = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)  
        self.embed_dim = embed_dim  
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim  
        self.mlp_ratio = mlp_ratio  
        self.use_mask=use_mask
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            num_frames=num_frames,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  #64*64
        patches_resolution = self.patch_embed.patches_resolution  #[64,64]
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                num_frames=num_frames,
                use_mask=self.use_mask
                ) 
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, lastquary, predmask, recurrent=True):
        #print("x_size:",x_size)
        x_size = (x.shape[3], x.shape[4])  #180,320
        h, w = x_size
        x = self.patch_embed(x)  #n,embed_dim,t,h,w
    
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        attn_mask = compute_mask(self.num_frames,x_size,tuple(self.window_size),self.shift_size, x.device)
        cond = x[:,:,:-1,:,:].contiguous().permute(0,2,3,4,1)
        x = x[:,:,-1,:,:].contiguous().permute(0,2,3,1)
        for layer in self.layers:    
            x, lastquary, predmask= layer(x.contiguous(), cond, attn_mask, lastquary, predmask, recurrent=recurrent)
            
        featmap = self.norm(x)  # b seq_len c

        x = featmap.permute(0, 3, 1, 2).contiguous()

        return x, lastquary, predmask

    def forward(self, x, lastquary, recurrent=True, ref=None):
        n, t, c, h, w = x.size()
        predmask = []

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            if c == 3:
                x = x.view(-1, c, h, w)
                x = self.conv_first(x)
                x = x.view(n, t, -1, h, w)

            if c == 64:
                x = x.view(-1, c, h, w)
                x = self.conv_first_feat(x)
                x = x.view(n, t, -1, h, w)

            x_center = x[:, -1, :, :, :].contiguous()
            feats, lastquary, predmask = self.forward_features(x, lastquary, predmask, recurrent=recurrent)
            x = self.conv_after_body(feats) + x_center
            if ref:
                x = self.conv_before_upsample(x)

        
        return x, lastquary, predmask
        

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        #flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i,layer in enumerate(self.layers):
            layer_flop=layer.flops()
            flops += layer_flop
            print(i,layer_flop / 1e9)


        flops += h * w * self.num_frames * self.embed_dim
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        #flops += self.upsample.flops()
        return flops





if __name__ == '__main__':
    upscale = 4
    window_size = (2, 8, 8)
    height = (1024 // upscale // window_size[1] + 1) * window_size[1]
    width = (1024 // upscale // window_size[2] + 1) * window_size[2]

    model = SwinIRFM(
        img_size=height,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=window_size,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.,
        upsampler='pixelshuffle',
    )

    print(model)
    #print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 5, 3, height, width))
    x = model(x)
    print(x.shape)
