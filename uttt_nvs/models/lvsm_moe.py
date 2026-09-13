import copy
import os

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
from PIL import Image
from omegaconf import ListConfig

from .layers import Block, _init_weights
from .ttt_minimal_v2 import parallel_ttt_config
from .class_name import get_obj_by_name


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.lpips_loss_weight = config.training.get("lpips_loss_weight", 0.0)
        self.lpips_loss_weight_backup = self.lpips_loss_weight

        if self.lpips_loss_weight > 0.0:
            self.lpips_loss_module = lpips.LPIPS(net="vgg").cuda().eval()
        
    def calculate_loss(
        self,
        rendering,
        target,
        lb_loss,
    ):
        """
        rendering: [b, v, 3, h, w]; in range (0, 1)
        target: [b, v, 3, h, w]; in range (0, 1)
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)

        # Clipping target to avoid 
        l2_diff = (rendering - target.clip(min=0, max=1)) ** 2
        l2_loss = l2_diff.mean()

        # compute the per-view per-batch psnr, then take the mean
        # [bv, c, h, w] -> [bv] -> [1]
        with torch.no_grad():
            psnr_values = -10. * torch.log10(l2_diff.mean(dim=(1, 2, 3)))
            psnr_loc = psnr_values.reshape(b, v).mean(dim=0)
            psnr = psnr_loc.mean()

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.lpips_loss_weight > 0.0:
            lpips_loss = self.lpips_loss_module(rendering, target, normalize=True).mean()

        loss = l2_loss + self.lpips_loss_weight * lpips_loss + lb_loss

        loss_metrics = {
            "loss": loss,
            "lb_loss": lb_loss,
            "l2_loss": l2_loss,
            "psnr_loc": psnr_loc,
            "psnr": psnr,
            "lpips_loss": lpips_loss,
        }
        return loss_metrics


def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d



class Images2latent3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(config.model.image_size, int):
            self.image_size = (config.model.image_size, config.model.image_size)
        else:
            self.image_size = config.model.image_size
        self.patch_size = config.model.patch_size
        self.dim = config.model.dim

        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]

        self.input_dim = len(self.posed_image_keys) * 3
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            ),
            nn.Linear(self.input_dim * (self.patch_size**2), self.dim, bias=False),
        )

        self.transformer_input_layernorm = nn.LayerNorm(self.dim, bias=False)
        CLASS = get_obj_by_name(config.model.block_config.type)
        self.transformer_blocks = CLASS(
            layers=config.model.layers,
            dim=self.dim,
            **config.model.block_config.params,
        )

        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * 3, bias=False),
            Rearrange(
                "b (v h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                h=self.image_size[0] // self.patch_size,
                w=self.image_size[1] // self.patch_size,
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Sigmoid(),
        )

        self.apply(_init_weights)

        self.loss_computer = LossComputer(config)

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        overview = edict(
            image_tokenizer=count_train_params(self.image_tokenizer),
            transformer=count_train_params(self.transformer_blocks),
            image_token_decoder=count_train_params(self.image_token_decoder),
        )
        return overview

    def forward(self, data_batch):
        """
        image (torch.tensor): [b, v, c, h, w]
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
        """
        # Do not autocast during the data processing
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_total_views, _, h, w = data_batch["image"].size()

            # Get rays_o, rays_d, and the pluckr coordinates from the camera information
            fxfycxcy = data_batch["fxfycxcy"]
            c2w = data_batch["c2w"]
            data_batch["ray_o"], data_batch["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            data_batch["o_cross_d"] = torch.cross(data_batch["ray_o"], data_batch["ray_d"], dim=2)
            data_batch["normalized_image"] = data_batch["image"] * 2.0 - 1.0

            # Compile the information for posed-image input, and pose-only input.
            data_batch["posed_image"] = torch.concat(
                [data_batch[key] for key in self.posed_image_keys], dim=2
            )
            data_batch["pose_only"] = torch.concat(
                [data_batch[key] for key in self.pose_keys], dim=2
            )

            # For evaluation, we evaluate all views in an autoregressive manner.
            transformer_input = data_batch["image"].new_zeros(batch_size, 2 * (num_total_views - 1), self.input_dim, h, w)
            input = {key: value[:, :-1] for key, value in data_batch.items()}
            target = {key: value[:, 1:] for key, value in data_batch.items()}
            transformer_input[:, ::2, :, :, :] = input["posed_image"]
            transformer_input[:, 1::2, :, :, :] = self.pad_pose_only(target["pose_only"])

        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_total_tokens = 2 * (num_total_views - 1) * num_img_tokens
        ttt_config = parallel_ttt_config(
            update_minibatch=num_img_tokens,
            apply_only_minibatch=num_img_tokens,
            length=num_total_tokens,
        )
        info_dict = {
            "num_img_tokens": num_img_tokens,
            "ttt_config": ttt_config, 
        }

        x = self.image_tokenizer(transformer_input)  # [b, v * n_patches, d]
        x = self.transformer_input_layernorm(x)
        x, lb_loss, infos = self.transformer_blocks(x, info_dict)


        rendered_images = self.image_token_decoder(x)
        target_rendered_images = rendered_images[:, 1::2]  # [b, v, 3, h, w]
        loss_metrics = self.loss_computer.calculate_loss(target_rendered_images, target["image"], lb_loss)

        if isinstance(infos[-1], dict):
            if "apply_max_violation_rate" in infos[-1]:
                loss_metrics.update(infos[-1])
            infos.pop()

        result = {
            "input":input,
            "target":target,
            "loss_metrics": loss_metrics,
            "rendering": target_rendered_images,
            "op2block2info": infos,
        }
        return result

    def pad_pose_only(self, x):
        # Pad pose-only input with zeros (for the RGB channel)
        b, v, c, h, w = x.size()
        zero_c = self.input_dim - c
        zeros = x.new_zeros(b, v, zero_c, h, w)
        x = torch.cat([x, zeros], dim=2)
        return x

    @torch.no_grad()
    def save_visuals(self, out_dir, result, data_batch):
        os.makedirs(out_dir, exist_ok=True)

        # All in [b, v, c, h, w]
        input, target, rendering = result["input"], result["target"], result["rendering"]

        visual = torch.cat((target["image"], rendering), dim=4).detach().cpu()
        visual = rearrange(visual, "b v c h (m w) -> (b h) (v m w) c", m=2)
        visual = (visual.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

        uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]
        uid_firstlast = f"{uids[0]:08}_{uids[-1]:08}"

        # Target + Rendering image
        supervision_image = Image.fromarray(visual)
        supervision_image.save(
            os.path.join(out_dir, f"supervision_{uid_firstlast}.jpg")
        )
        with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
            uids = "_".join([f"{uid:08}" for uid in uids])
            f.write(uids)

        # Input image
        input_image = rearrange(input["image"], "b v c h w -> (b h) (v w) c")
        input_image = (
            (input_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        )
        input_image = Image.fromarray(input_image[..., :3])
        input_image.save(
            os.path.join(out_dir, f"input_{uid_firstlast}.jpg")
        )

        # Draw figure to show the auto-regressive PSNR results.
        import matplotlib.pyplot as plt
        from io import BytesIO

        psnr_loc = result["loss_metrics"]["psnr_loc"].cpu().numpy()   # [v]
        v = psnr_loc.shape[0]
        fig, ax = plt.subplots()
        ax.plot(range(1, v+1), psnr_loc, marker='o')
        ax.set_xlabel('View (1->v)')
        ax.set_ylabel('PSNR')
        ax.set_title('Per-view PSNR')
        ax.grid(True)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        from PIL import Image as PILImage
        psnr_loc_image = PILImage.open(buf).convert("RGB")
        buf.close()
        plt.close(fig)

        return {
            "supervision": supervision_image,
            "input": input_image,
            "psnr_loc": psnr_loc_image,
        }
    