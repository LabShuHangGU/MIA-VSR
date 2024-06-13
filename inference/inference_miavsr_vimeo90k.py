import cv2
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import argparse
from archs.mia_vsr_arch import MIAVSR
from basicsr.data.data_util import read_img_seq
from basicsr.metrics import psnr_ssim
from basicsr.utils import get_root_logger, imwrite, tensor2img

def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    #parser.add_argument('config', help='test config file path')
    parser.add_argument('--folder_type',type=str,default="vimeo_test" , help='folder_name')
    parser.add_argument('--vimeo', type=str,default="/data0/zhouxingyu/vimeo_septuplet/sep_testlist.txt", help='index corresponds to the first frame of the sequence')

    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    # -------------------- Configurations -------------------- #
    args = parse_args()
    device = torch.device('cuda',args.device)
    save_imgs = False
    # set suitable value to make sure cuda not out of memory
    # for vimeo90K dataset, we load the whole clip at once
    interval = 7
    # which channel is used to evaluate
    test_y_channel = True
    # measure the quality of center frame
    center_frame_only = True
    # flip sequence
    flip_seq = False
    crop_border = 0
    # model
    model_path = '/data1/home/zhouxingyu/zhouxingyu_vsr/MIA-VSR/experiments/pretrained_models/MIAVSR_Vid4_x4.pth'  # noqa E501
    # test data
    test_name = f'Vimeo90K_maskfinetune2-1000'

    lr_folder = '/data0/zhouxingyu/vimeo_septuplet/BIx4_test'
    gt_folder = '/data0/zhouxingyu/vimeo_septuplet/sequences'
    save_folder = f'/data1/home/zhouxingyu/zhouxingyu_vsr/MIA-VSR/inference/results/{test_name}'
    os.makedirs(save_folder, exist_ok=True)

    # set up the models
    model = MIAVSR(mid_channels=64,
                 embed_dim=120,
                 depths=[6, 6, 6,6],
                 num_heads=[6,6,6,6],
                 window_size=[3, 8, 8],
                 num_frames=3,
                 cpu_cache_length=100,
                 is_low_res_input=True,
                 use_mask=True,
                 spynet_path='/data1/home/zhouxingyu/zhouxingyu_vsr/MIA-VSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')

    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # load video names
    vimeo_motion_txt = '/data0/zhouxingyu/vimeo_septuplet/sep_testlist.txt'
    #vimeo_motion_txt=args.vimeo
    subfolder_names = []
    avg_psnr_l = []
    avg_ssim_l = []
    folder_type = args.folder_type
    image_foler='image'
    save_subfolder_root = os.path.join(save_folder, folder_type)
    os.makedirs(save_subfolder_root, exist_ok=True)

    # logger
    log_file = osp.join(save_subfolder_root, f'psnr_test_{folder_type}.log')
    #print(log_file)
    logger = get_root_logger(logger_name='swinvir_recurrent', log_level=logging.INFO, log_file=log_file)
    logger.info(f'Data: {test_name} - {lr_folder}')
    logger.info(f'Model path: {model_path}')

    count = 0
    for line in open(vimeo_motion_txt).readlines():
        count += 1
        video_name = line.strip().split(' ')[0]
        subfolder_names.append(video_name)
        hq_video_path = os.path.join(gt_folder, video_name)
        lq_video_path = os.path.join(lr_folder, video_name)
        imgs_lq, imgnames = read_img_seq(lq_video_path, return_imgname=True)
        external_folder,internal_folder=video_name.split('/')
        center_img=f"{external_folder}_{internal_folder}.png"
        imgs_lq = imgs_lq.unsqueeze(0).to(device)
        n = imgs_lq.size(1)

        # flip seq
        if flip_seq:
            imgs_lq = torch.cat([imgs_lq, imgs_lq.flip(1)], dim=1)

        with torch.no_grad():
            outputs,_ = model(imgs_lq)

        if flip_seq:
            output_1 = outputs[:, :n, :, :, :]
            output_2 = outputs[:, n:, :, :, :].flip(1)
            outputs = 0.5 * (output_1 + output_2)

        if center_frame_only:
            output = outputs[:, n // 2, :, :, :]
            output = tensor2img(output, rgb2bgr=True, min_max=(0, 1))

        img_gt = cv2.imread(osp.join(hq_video_path, 'im4.png'), cv2.IMREAD_UNCHANGED)
        crt_psnr = psnr_ssim.calculate_psnr(output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)
        crt_ssim = psnr_ssim.calculate_ssim(output, img_gt, crop_border=crop_border, test_y_channel=test_y_channel)

        # save
        if save_imgs:
            imwrite(output, osp.join(save_subfolder_root, image_foler, center_img))

        logger.info(f'Folder {video_name} - PSNR: {crt_psnr:.6f} dB. SSIM: {crt_ssim:.6f}')

        avg_psnr_l.append(crt_psnr)
        avg_ssim_l.append(crt_ssim)

    logger.info(f'Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')
    logger.info(f'Average SSIM: {sum(avg_ssim_l) / len(avg_ssim_l):.6f} dB ' f'for {len(subfolder_names)} clips. ')


if __name__ == '__main__':

    main()
