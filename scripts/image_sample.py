"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.utils import make_grid, save_image


def main():
    args = create_argparser().parse_args()

    # taohu
    if True:
        args.use_ddim = True
        args.model_path = "lsun_bedroom.pt"
        args.batch_size = 8
        args.num_samples = 16
        args.timestep_respacing = "ddim250"

        args.attention_resolutions = "32, 16, 8"
        args.class_cond = False
        args.image_size = 256
        args.dropout = 0.0
        args.learn_sigma = True
        args.noise_schedule = "linear"
        args.diffusion_steps = 1000
        args.num_channels = 256
        args.num_head_channels = 64

        args.num_res_blocks = 2
        args.resblock_updown = True
        args.use_fp16 = False
        args.use_scale_shift_norm = True

    ################

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        if True:

            def model_fn(x, t, y=None):
                return model(x, t, y)

            # taohu
            sample = (sample + 1) * 0.5
            save_image(sample, "sample.png")
            sample = (sample - 0.5) * 2.0
            noise_z = diffusion.ddim_sample_reverse_loop(
                model_fn,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                noise=sample,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=dist_util.dev(),
                eta=0.0,
            )
            print(noise_z.shape)
            print(noise_z.min(), noise_z.max(), noise_z.mean(), noise_z.std())
            recovered_sample = diffusion.ddim_sample_loop_by_ode(
                model_fn,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                noise=noise_z,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=dist_util.dev(),
                eta=0.0,
            )
            recovered_sample = (recovered_sample + 1) * 0.5
            save_image(recovered_sample, "recovered_sample.png")
            print("done dddd")
            exit(0)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
