import io
import cv2
import os
import math
import random
import av
import numpy as np
import os.path as osp
# from scipy.io import loadmat
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.degradations import (random_add_gaussian_noise,
                                       random_mixed_kernels)
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class VFHQDataset(data.Dataset):

    def __init__(self, opt):
        super(VFHQDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        
        self.crop_components = False
        
        self.load_latent_gt = False  

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)
        
        # General parameters
        self.scale = opt.get('scale', 4)
        self.num_frame = opt.get('video_frame', 0)
        self.interval_list = opt.get('interval_list', [1, 2, 3, 4, 5, 6])
        self.random_reverse = opt.get('random_reverse', True)
        self.use_flip = opt.get('use_flip', False)
        self.use_rot = opt.get('use_rot', False)

        # Degradation parameters
        self.blur_kernel_size = opt.get('blur_kernel_size', 21)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.7, 0.3])
        self.blur_x_sigma = opt.get('blur_x_sigma', [2, 10])
        self.blur_y_sigma = opt.get('blur_y_sigma', [2, 10])
        self.noise_range = opt.get('noise_range', [0, 10])
        self.resize_prob = opt.get('resize_prob', [0.3, 0.3, 0.4])
        self.crf_range = opt.get('crf_range', [25, 45])
        self.vcodec = opt.get('vcodec', ['libx264', 'h264', 'mpeg4'])
        self.vcodec_prob = opt.get('vcodec_prob', [0, 1, 0])

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.gt_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.gt_size - components_bbox['right_eye'][0]
            components_bbox['nose'][0] = self.gt_size - components_bbox['nose'][0]
            components_bbox['mouth'][0] = self.gt_size - components_bbox['mouth'][0]
        
        locations_gt = {}
        locations_in = {}
        for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            elif part == 'nose':
                half_len *= self.nose_enlarge_ratio
            elif part == 'mouth':
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc/(self.gt_size//self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in


    def noresize__getitem__(self, index):
        frames_cnt = 5

        video_path=self.paths[index]

        all_list = []
        for files in os.listdir(video_path):
            all_list.append(files)

        all_list.sort()

        rnd_range=len(all_list)-frames_cnt
        start=random.randrange(0,rnd_range)
        selected=all_list[start:start + frames_cnt]

        img_gts = []

        for sel_frame in selected:
            img_gt_path = os.path.join(video_path, sel_frame)
            img_gt = cv2.imread(img_gt_path) / 255.0
            img_gts.append(img_gt)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        # ------------- generate LQ frames --------------#
        # add blur
        kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, self.blur_kernel_size, self.blur_x_sigma,
                                      self.blur_y_sigma)
        img_lqs = [cv2.filter2D(v, -1, kernel) for v in img_gts]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]
        # downsample
        original_height, original_width = img_gts[0].shape[0:2]
        resize_type = random.choices(
            [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC], self.resize_prob)[0]
        resized_height, resized_width = int(
            original_height // self.scale), int(original_width // self.scale)
        # ensure the resized_height and resized_width are even numbers
        img_lqs = [cv2.resize(v, (resized_width, resized_height),
                              interpolation=resize_type) for v in img_lqs]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]

        # ffmpeg
        crf = np.random.randint(self.crf_range[0], self.crf_range[1])
        codec = random.choices(self.vcodec, self.vcodec_prob)[0]

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = resized_height
            stream.width = resized_width
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': str(crf)}

            for img_lq in img_lqs:
                img_lq = np.clip(img_lq * 255, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img_lq, format='rgb24')
                frame.pict_type = 0  # Changed from 'NONE' to 0
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        img_lqs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    img_lqs.append(frame.to_rgb().to_ndarray() / 255.)

        assert len(img_lqs) == len(img_gts), 'Wrong length'

        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        self.normalize = True
        if self.normalize:
            normalize(img_lqs, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gts, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts}




        """
        return_dict = {'in': img_in, 'in_large_de': img_in_large, 'gt': img_gt, 'gt_path': gt_path}

        if self.crop_components:
            return_dict['locations_in'] = locations_in
            return_dict['locations_gt'] = locations_gt

        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt
        """
        return_dict = 0
        return return_dict
    

    def __getitem__(self, index):
        frames_cnt = self.num_frame

        video_path=self.paths[index]

        with av.open(video_path) as container:
            frames = []
            for frame in container.decode(video=0):
                frame = frame.to_rgb().to_ndarray()[:, :, ::-1] / 255.0

                h, w, _ = frame.shape

                refer_length = min(h, w) - min(h, w) % 2
                center_h = h // 2
                center_w = w // 2
                new_x1 = max(center_w - (refer_length // 2), 0)
                new_x2 = min(center_w + (refer_length // 2), w)
                new_y1 = max(center_h - (refer_length // 2), 0)
                new_y2 = min(center_h + (refer_length // 2), h)

                cropped_img = frame.copy()[new_y1:new_y2, new_x1:new_x2, :]
                frame = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_LINEAR)

                frames.append(frame)

        rnd_range=len(frames)-frames_cnt
        start=random.randrange(0,rnd_range)

        img_gts=frames[start:start + frames_cnt]

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        # ------------- generate LQ frames --------------#
        # add blur
        kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, self.blur_kernel_size, self.blur_x_sigma,
                                    self.blur_y_sigma)
        img_lqs = [cv2.filter2D(v, -1, kernel) for v in img_gts]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]
        # downsample
        original_height, original_width = img_gts[0].shape[0:2]
        resize_type = random.choices(
            [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC], self.resize_prob)[0]
        resized_height, resized_width = int(
            original_height // self.scale), int(original_width // self.scale)
        # ensure the resized_height and resized_width are even numbers
        img_lqs = [cv2.resize(v, (resized_width, resized_height),
                            interpolation=resize_type) for v in img_lqs]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]

        # ffmpeg
        crf = np.random.randint(self.crf_range[0], self.crf_range[1])
        codec = random.choices(self.vcodec, self.vcodec_prob)[0]

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = resized_height
            stream.width = resized_width
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': str(crf)}

            for img_lq in img_lqs:
                img_lq = np.clip(img_lq * 255, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img_lq, format='rgb24')
                frame.pict_type = 0  # Changed from 'NONE' to 0
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        img_lqs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    img_lqs.append(frame.to_rgb().to_ndarray() / 255.)

        assert len(img_lqs) == len(img_gts), 'Wrong length'

        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        self.normalize = True
        if self.normalize:
            normalize(img_lqs, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gts, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'stat': True}



    def __len__(self):
        return len(self.paths)