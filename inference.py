import math
import os
import av
import argparse
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
from basicsr.utils.misc import get_device

from basicsr.utils.registry import ARCH_REGISTRY

# CILP
import clip
import torchvision.transforms as transforms

from basicsr.utils.clip_util import VisionTransformer
clip.model.VisionTransformer = VisionTransformer

import basicsr.models.lcm_pipecall as lcm_pipecall

# LCM
from diffusers import UNet2DConditionModel, ControlNetModel

from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter

from diffusers.utils import export_to_video

@torch.no_grad()
def main():
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs', 
            help='Input image, video or folder. Default: inputs')
    parser.add_argument('-o', '--output_path', type=str, default="./results",
            help='Output folder. Default: results')
    # LCM
    parser.add_argument('-t', '--step', type=int, default=4, help='T for lcm')
    parser.add_argument('-f', '--frame_unitsize', type=int, default=6, help='frames count of once process')
    parser.add_argument('--visual_path', type=str,
                        default='./weights/inter_video_lcm/visual_encoder_4step_6frames.pth',
                        help='visual_encoder checkpoint')
    parser.add_argument('--spatial_path', type=str,
                        default='./weights/inter_video_lcm/spatial_encoder_4step_6frames.pth',
                        help='spatial_encoder checkpoint')
    
    # post
    parser.add_argument('--stack', type=int, default=1,
                        help='')

    args = parser.parse_args()
    print(args)
    
    frame_cnt = args.frame_unitsize
    stack = args.stack

    # init models
    # CLIPImageEncoder
    clip_model, clip_preprocess = clip.load('ViT-B/16', device=device)
    preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                    clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                    clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

    # Visual Encoder
    visual_encoder = ARCH_REGISTRY.get('VisualEncoder')(nf=64, emb_dim=197, video_frame = frame_cnt,  ch_mult=[2,4,8], res_blocks=2, img_size=512).to(device)
    checkpoint_ve = torch.load(args.visual_path)['params_ema']
    visual_encoder.load_state_dict(checkpoint_ve)
    visual_encoder.eval()

    # Spatial Encoder
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path="emilianJR/epiCRealism", subfolder="unet")
    spatial_encoder = ControlNetModel.from_unet(unet).to(device)
    checkpoint_c = torch.load(args.spatial_path)['params_ema']
    spatial_encoder.load_state_dict(checkpoint_c)
    spatial_encoder.eval()

    # lcm
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    pipe.se = spatial_encoder

    # read video
    for filename in os.listdir(args.input_path):
        if filename.endswith(".mp4"):
            print(f"Proccessing video {filename} ...")

            file_path = os.path.join(args.input_path, filename)

            with av.open(file_path) as container:
                frames = []
                for frame in container.decode(video=0):
                    frames.append(frame.to_rgb().to_ndarray()[:, :, ::-1] / 255.0)

            frames = img2tensor(frames)
            frames = torch.stack(frames, dim=0)

            normalize(frames, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

            # frames = torch.tensor(T, C, H, W)

            total_frames = frames.shape[0]
            num_units = math.ceil((total_frames - stack) / (frame_cnt - stack))
            if num_units < 1:
                num_units = 1
            padded_total_frames = num_units * (frame_cnt - stack) + stack

            if padded_total_frames > total_frames:
                last_frame = frames[-1:]
                padding_needed = padded_total_frames - total_frames
                padding_frames = last_frame.repeat(padding_needed, 1, 1, 1)
                frames = torch.cat((frames, padding_frames), dim=0)
            
            frames = frames.to(device)

            all_video = torch.empty(1, 3, 0, 512, 512).to(device)

            for i in range(num_units):
                print(f"Proccessing unit {i + 1}/{num_units} in video {filename}")
                start_index = i * (frame_cnt - stack)
                end_index = start_index + frame_cnt
                current_unit_frames = frames[start_index:end_index]
                current_unit_frames = torch.unsqueeze(current_unit_frames, dim = 0)

                embed_stacked = torch.empty(1,0,768).to(device)

                for j in range(frame_cnt):
                    input_std = preprocess(current_unit_frames[:, j])
                    embed = clip_model.encode_image(input_std).to(torch.float)
                    embed_stacked = torch.cat((embed_stacked, embed), dim = 1)

                visual_feat = visual_encoder(embed_stacked)  # output of Visual Encoder

                torch.cuda.empty_cache()

                latent_code = pipe.vae.encode(current_unit_frames.to(torch.float)[0])['latent_dist'].mean
                
                latent_code = latent_code * 0.18215
                with torch.no_grad():
                    output = lcm_pipecall.call(pipe,
                                            latent_code,
                                            prompt="A video of a human face",
                                            negative_prompt="",
                                            num_frames=frame_cnt,
                                            guidance_scale=1.0,
                                            num_inference_steps=args.step,
                                            generator=torch.Generator("cpu").manual_seed(0),
                                            input_embed = visual_feat.to(torch.float),
                                            lq_input=current_unit_frames.to(torch.float)[0]
                                            ).to(device)
                    
                # B C T H W
                if i == 0:
                    all_video = torch.cat((all_video, output), dim = 2)
                else:
                    prev_video_non_overlap = all_video[:, :, :-stack, :, :]
                    prev_video_overlap_part = all_video[:, :, -stack:, :, :]
                    new_video_overlap_part = output[:, :, :stack, :, :]
                    averaged_overlap = (prev_video_overlap_part + new_video_overlap_part) / 2.0
                    new_video_non_overlap = output[:, :, stack:, :, :]
                    all_video = torch.cat(
                        (prev_video_non_overlap, averaged_overlap, new_video_non_overlap), 
                        dim=2
                    )
                
            all_video = all_video[:, :, :total_frames]
            output_save = pipe.video_processor.postprocess_video(video=all_video, output_type="pil")
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            print(f"Saving video {filename}")
            export_to_video(output_save[0], f'{args.output_path}/{filename}', fps = 24)
        
    return 1

if __name__ == '__main__':
    main()