import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY
from .mia_sliding_arch import SwinIRFM


@ARCH_REGISTRY.register()
class MIAVSR(nn.Module):
    """MIA-VSR network structure.


    Paper:
        Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention

    """

    def __init__(self,
                in_channels=3,
                mid_channels=64,
                embed_dim=120,
                depths=(6, 6, 6, 6, 6, 6),
                num_heads=(6, 6, 6, 6, 6, 6),
                window_size=(3, 8, 8),
                num_frames=3,
                img_size = 64,
                patch_size=1,
                cpu_cache_length=100,
                is_low_res_input=True,
                use_mask=True,
                spynet_path=None):
        super().__init__()
        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.conv_before_upsample = nn.Conv2d(embed_dim, mid_channels, 3, 1, 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.ues_mask = use_mask
        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            #self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
            self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        
        # propagation branches
        self.patch_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.patch_align[module] = SwinIRFM(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_channels,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=2.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint=True,
                    upscale=4,
                    img_range=1.,
                    upsampler='pixelshuffle',
                    resi_connection='1conv',
                    num_frames=num_frames,
                    use_mask=self.ues_mask
                    )


        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propgated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t,_, h, w = flows.size()
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        lastquary = []
        pred = []

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = torch.zeros(n, self.embed_dim, h, w)
        for i, idx in enumerate(frame_idx):
            if module_name == 'backward_1':
                feat_current = feats['spatial'][mapping_idx[idx]]
            if module_name == 'forward_1':
                feat_current = feats['backward_1'][mapping_idx[idx]]
            if module_name == 'backward_2':
                feat_current = feats['forward_1'][mapping_idx[idx]]
            if module_name == 'forward_2':
                feat_current = feats['backward_2'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp_avg_patch(feat_prop, flow_n1)

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = cond_n1

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp_avg_patch(feat_n2, flow_n2)


                cond = torch.stack([cond_n2, cond_n1, feat_current], dim=1)
                # patch alignment
                
                feat_prop,  lastquary, predmask= self.patch_align[module_name](cond, lastquary)#
                pred.append(predmask)
                

            if i == 0:
                cond = torch.stack([feat_current, feat_current, feat_current], dim=1)
                
                feat_prop, lastquary, predmask = self.patch_align[module_name](cond, lastquary, recurrent=False)
                pred.append(predmask)

            
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()


        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        
        return feats, pred
    

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = feats['forward_2'][i]
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.conv_before_upsample(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)


    def forward(self, lqs):

        """Forward function for MIAVSR.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()
        pred_all = []
        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.conv_first(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.conv_first(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
            
        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        shape = [n, t-1,h,w]
        # feature propgation
        for iter_ in [1,2]:#[1, 2]
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats, pred = self.propagate(feats, flows, module)
                pred_all.append(pred)
                if self.cpu_cache:
                    del flows
                    del pred
                    torch.cuda.empty_cache()
        if not self.ues_mask:
            pred_all = None
        return self.upsample(lqs, feats), pred_all

    def flops(self):
        flops = 0
        h,w = self.img_size,self.img_size
        flops += h * w * 3 * self.embed_dim * 9
        for pipl_name,modules in self.patch_align.items():
            modules_flop = modules.flops()
            flops += modules_flop
            print(pipl_name, modules_flop / 1e9)
            print("\n")

        flops += h * w * self.embed_dim * self.mid_channels * 9
        flops += h * w * self.mid_channels * self.mid_channels* 4 * 9
        flops += 2*h * 2*w * self.mid_channels * self.mid_channels * 4 * 9
        flops += 4*h * 4*w * self.mid_channels * self.mid_channels  * 9
        flops += 4*h * 4*w * self.mid_channels * 3  * 9

        return flops





def flow_warp_avg_patch(x, flow, interpolation='nearest', padding_mode='zeros', align_corners=True):
    """Patch Alignment

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, 2,h, w). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    # if x.size()[-2:] != flow.size()[1:3]:
    #     raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
    #                      f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # patch size is set to 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    flow = F.pad(flow, (0, pad_w, 0, pad_h), mode='reflect')
    hp = h + pad_h
    wp = w + pad_w
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, hp), torch.arange(0, wp))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    flow = F.avg_pool2d(flow, 8)
    flow = F.interpolate(flow, scale_factor=8, mode='nearest')
    flow = flow.permute(0, 2, 3, 1)
    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow = grid_flow[:, :h, :w, :]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0  #grid[:,:,:,0]是w
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(), grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output



if __name__ == '__main__':
    #upscale = 4
    window_size = [3, 8, 8]
    img_size=64

    model = MIAVSR(
        mid_channels = 64,
        embed_dim=120,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        window_size=window_size,
        num_frames = 3,
        img_size = img_size,
        patch_size = 1,
        cpu_cache_length = 100,
        is_low_res_input = True,
        spynet_path = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    )

    print(model)
    print("flops",model.flops() / 1e9 + 'G')

    x = torch.randn((1, 5, 3, img_size, img_size))
    x = model(x)
    print(x.shape)
