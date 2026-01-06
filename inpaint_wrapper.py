import os
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from PIL import Image

from relighting.argument import CONTROLNET_MODELS, SD_MODELS
from relighting.ball_processor import get_ideal_normal_ball
from relighting.image_processor import pil_square_image
from relighting.inpainter_2lora import BallInpainter
from relighting.mask_utils import MaskGenerator
from relighting.utils import name2hash
import relighting.dist_utils as dist_util

ImageInput = Union[str, os.PathLike, Image.Image]


class DiffusionLightInpaintWrapper:
    def __init__(
        self,
        model_option: str = "sdxl",
        use_controlnet: bool = True,
        use_lora: bool = True,
        lora_scale: float = 0.75,
        exposure_lora_path: str = "models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500",
        exposure_lora_scale: float = 0.75,
        turbo_lora_path: str = "models/rev3/Flickr2K/Flickr2kPlus_extended/checkpoint-230000",
        turbo_lora_scale: float = 1.0,
        img_width: int = 1024,
        img_height: int = 1024,
        force_square: bool = True,
        device: Optional[torch.device] = None,
        is_cpu: bool = False,
        offload: bool = False,
        use_torch_compile: bool = False,
    ):
        self.model_option = self._resolve_model_option(model_option)
        self.use_controlnet = use_controlnet
        self.use_lora = use_lora
        self.lora_scale = lora_scale
        self.exposure_lora_path = exposure_lora_path
        self.exposure_lora_scale = exposure_lora_scale
        self.turbo_lora_path = turbo_lora_path
        self.turbo_lora_scale = turbo_lora_scale
        self.img_width = img_width
        self.img_height = img_height
        self.force_square = force_square
        self.device = device or (torch.device("cpu") if is_cpu else dist_util.dev())
        self.torch_dtype = torch.float32 if self.device.type == "cpu" else torch.float16

        self.pipe = self._create_pipe(offload=offload)
        self.mask_generator = MaskGenerator()
        self._active_lora = None

        if self.use_lora:
            self._load_exposure_lora()

        if use_torch_compile:
            try:
                self.pipe.pipeline.unet = torch.compile(
                    self.pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True
                )
            except Exception:
                pass

    def inpaint(
        self,
        image: ImageInput,
        prompt: str = "a perfect mirrored reflective chrome ball sphere",
        prompt_dark: str = "a perfect black dark mirrored reflective chrome ball sphere",
        negative_prompt: str = "matte, diffuse, flat, dull",
        ball_size: int = 256,
        ball_dilate: int = 20,
        seed: Union[int, str] = 0,
        seed_key: Optional[str] = None,
        denoising_step: int = 30,
        control_scale: float = 0.5,
        algorithm: str = "turbo_swapping",
        strength: Optional[float] = None,
        switch_lora_timestep: int = 800,
        num_iteration: int = 2,
        ball_per_iteration: int = 30,
        agg_mode: str = "median",
        save_intermediate: bool = False,
        cache_dir: str = "./temp_inpaint_iterative",
        enable_acceleration: bool = False,
        ev: Union[str, Iterable[float]] = "0",
        max_negative_ev: float = -5.0,
        guidance_scale: float = 5.0,
        return_square: bool = False,
        sdedit_timestep: Optional[Union[int, float]] = None,
        sdedit_timestep_is_index: bool = False,
    ):
        if ball_dilate % 2 != 0:
            raise ValueError("ball_dilate must be an even number")

        algorithm = self._normalize_algorithm(algorithm)
        if sdedit_timestep is not None and algorithm not in ["normal", "turbo_swapping"]:
            raise ValueError(
                "sdedit_timestep is only supported with algorithm='normal' or 'turbo_swapping'"
            )
        input_image, image_id = self._prepare_image(image)
        seed = self._resolve_seed(seed, seed_key=seed_key or image_id)

        x, y, r = self._get_ball_location(
            input_image.size, ball_size=ball_size, ball_dilate=ball_dilate
        )
        normal_ball, mask_ball = get_ideal_normal_ball(size=ball_size + ball_dilate)
        mask = self.mask_generator.generate_single(
            input_image,
            mask_ball,
            x - (ball_dilate // 2),
            y - (ball_dilate // 2),
            r + ball_dilate,
        )

        embeddings = self._interpolate_embeddings(
            prompt=prompt,
            prompt_dark=prompt_dark,
            ev=ev,
            max_negative_ev=max_negative_ev,
        )

        outputs = {}
        for ev_value, (prompt_embeds, pooled_prompt_embeds) in embeddings.items():
            if algorithm in ["turbo_sdedit", "turbo_pred", "turbo_swapping"]:
                self._load_turbo_lora()
            elif self.use_lora:
                self._load_exposure_lora()

            base_strength = 1.0
            if sdedit_timestep is not None:
                base_strength = self._strength_from_timestep(
                    sdedit_timestep,
                    denoising_step,
                    timestep_is_index=sdedit_timestep_is_index,
                )
            elif algorithm in ["normal", "turbo_swapping"] and strength is not None:
                base_strength = strength

            generator = torch.Generator().manual_seed(seed)
            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt": negative_prompt,
                "num_inference_steps": denoising_step,
                "generator": generator,
                "image": input_image,
                "mask_image": mask,
                "strength": base_strength,
                "current_seed": seed,
                "controlnet_conditioning_scale": control_scale,
                "height": input_image.size[1],
                "width": input_image.size[0],
                "normal_ball": normal_ball,
                "mask_ball": mask_ball,
                "x": x,
                "y": y,
                "r": r,
                "guidance_scale": guidance_scale,
            }

            if self.use_lora:
                kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}

            if algorithm == "normal":
                output_image = self.pipe.inpaint(**kwargs).images[0]
            elif algorithm == "iterative":
                iter_strength = 0.8 if strength is None else strength
                kwargs.update(
                    {
                        "strength": iter_strength,
                        "num_iteration": num_iteration,
                        "ball_per_iteration": ball_per_iteration,
                        "agg_mode": agg_mode,
                        "save_intermediate": save_intermediate,
                        "cache_dir": cache_dir,
                    }
                )
                output_image = self.pipe.inpaint_iterative(**kwargs)
            elif algorithm in ["turbo_sdedit", "turbo_pred"]:
                if algorithm == "turbo_pred":
                    enable_acceleration = True
                iter_strength = 0.8 if strength is None else strength
                kwargs.update(
                    {
                        "strength": iter_strength,
                        "num_iteration": num_iteration,
                        "ball_per_iteration": ball_per_iteration,
                        "agg_mode": agg_mode,
                        "save_intermediate": save_intermediate,
                        "cache_dir": cache_dir,
                        "enable_acceleration": enable_acceleration,
                        "exposure_lora_path": self.exposure_lora_path,
                        "exposure_lora_scale": self.exposure_lora_scale,
                    }
                )
                output_image = self.pipe.inpaint_turbo_sdedit(**kwargs)
                self._active_lora = None
            elif algorithm == "turbo_swapping":
                kwargs.update(
                    {
                        "switch_lora_timestep": switch_lora_timestep,
                        "exposure_lora_path": self.exposure_lora_path,
                        "exposure_lora_scale": self.exposure_lora_scale,
                    }
                )
                output_image = self.pipe.inpaint_turbo_swapping(**kwargs).images[0]
                self._active_lora = None
            else:
                raise NotImplementedError(f"Unknown algorithm {algorithm}")

            if return_square:
                square_image = output_image.crop((x, y, x + r, y + r))
                outputs[ev_value] = (output_image, square_image)
            else:
                outputs[ev_value] = output_image

        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

    def _create_pipe(self, offload: bool):
        if self.model_option not in SD_MODELS:
            raise ValueError(f"Unknown model option {self.model_option}")

        if self._is_sdxl_model(self.model_option):
            model = SD_MODELS[self.model_option]
            controlnet = CONTROLNET_MODELS[self.model_option] if self.use_controlnet else None
            return BallInpainter.from_sdxl(
                model=model,
                controlnet=controlnet,
                device=self.device,
                torch_dtype=self.torch_dtype,
                offload=offload,
            )

        model = SD_MODELS[self.model_option]
        controlnet = CONTROLNET_MODELS[self.model_option] if self.use_controlnet else None
        return BallInpainter.from_sd(
            model=model,
            controlnet=controlnet,
            device=self.device,
            torch_dtype=self.torch_dtype,
            offload=offload,
        )

    def _resolve_model_option(self, model_option: str) -> str:
        if model_option == "sdxl_fast":
            return "sdxl"
        return model_option

    def _normalize_algorithm(self, algorithm: str) -> str:
        if algorithm in ["special", "turbo"]:
            return "turbo_swapping"
        return algorithm

    def _is_sdxl_model(self, model_option: str) -> bool:
        return model_option.startswith("sdxl") or model_option.startswith("hypersd")

    def _load_exposure_lora(self):
        if self._active_lora == "exposure":
            return
        self._clear_lora()
        self.pipe.pipeline.load_lora_weights(self.exposure_lora_path)
        self.pipe.pipeline.fuse_lora(lora_scale=self.exposure_lora_scale)
        self._active_lora = "exposure"

    def _load_turbo_lora(self):
        if self._active_lora == "turbo":
            return
        self._clear_lora()
        self.pipe.pipeline.load_lora_weights(self.turbo_lora_path)
        self.pipe.pipeline.fuse_lora(lora_scale=self.turbo_lora_scale)
        self._active_lora = "turbo"

    def _clear_lora(self):
        try:
            self.pipe.pipeline.unfuse_lora()
            self.pipe.pipeline.unload_lora_weights()
        except Exception:
            pass

    def _prepare_image(self, image: ImageInput) -> Tuple[Image.Image, Optional[str]]:
        image_id = None
        if isinstance(image, (str, os.PathLike)):
            image_id = os.fspath(image)
            image = Image.open(image_id).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a path or PIL.Image")

        target_width = self.img_width or image.size[0]
        target_height = self.img_height or image.size[1]
        target_size = (target_width, target_height)

        if self.force_square:
            image = pil_square_image(image, desired_size=target_size)
        elif target_size != image.size:
            image = image.resize(target_size)

        return image, image_id

    def _get_ball_location(
        self, image_size: Tuple[int, int], ball_size: int, ball_dilate: int
    ) -> Tuple[int, int, int]:
        img_width, img_height = image_size
        x = (img_width // 2) - (ball_size // 2)
        y = (img_height // 2) - (ball_size // 2)
        r = ball_size

        half_dilate = ball_dilate // 2
        if x - half_dilate < 0:
            x = half_dilate
        if y - half_dilate < 0:
            y = half_dilate
        if x + r + half_dilate > img_width:
            x = img_width - r - half_dilate
        if y + r + half_dilate > img_height:
            y = img_height - r - half_dilate

        return x, y, r

    def _resolve_seed(self, seed: Union[int, str], seed_key: Optional[str] = None) -> int:
        if seed == "auto":
            if seed_key:
                return name2hash(seed_key)
            return 0
        return int(seed)

    def _strength_from_timestep(
        self,
        noise_timestep: Union[int, float],
        num_inference_steps: int,
        timestep_is_index: bool = False,
    ) -> float:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        scheduler = self.pipe.pipeline.scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps.detach().cpu()

        if timestep_is_index:
            idx = int(noise_timestep)
            if idx < 0 or idx >= len(timesteps):
                raise ValueError("noise_timestep index is out of range")
        else:
            target = float(noise_timestep)
            idx = int((timesteps - target).abs().argmin().item())

        init_timestep = len(timesteps) - idx
        strength = init_timestep / len(timesteps)
        return min(max(strength, 0.0), 1.0)

    def _encode_prompt(self, prompt: str):
        try:
            encoded = self.pipe.pipeline.encode_prompt(prompt)
        except TypeError:
            encoded = self.pipe.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                lora_scale=None,
            )
        if isinstance(encoded, tuple):
            if len(encoded) >= 4:
                return encoded[0], encoded[2]
            if len(encoded) >= 2:
                return encoded[0], None
        return encoded, None

    def _parse_ev(self, ev: Union[str, Iterable[float]]) -> Tuple[float, ...]:
        if isinstance(ev, str):
            if not ev:
                return (0.0,)
            return tuple(float(x) for x in ev.split(","))
        if isinstance(ev, (int, float)):
            return (float(ev),)
        return tuple(float(x) for x in ev)

    def _interpolate_embeddings(
        self,
        prompt: str,
        prompt_dark: str,
        ev: Union[str, Iterable[float]],
        max_negative_ev: float,
    ) -> Dict[float, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        ev_list = self._parse_ev(ev)
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        if prompt_dark is None:
            return {ev_value: (prompt_embeds, pooled_prompt_embeds) for ev_value in ev_list}

        if max_negative_ev == 0 or (len(ev_list) == 1 and ev_list[0] == 0):
            return {ev_list[0]: (prompt_embeds, pooled_prompt_embeds)}

        dark_embeds, dark_pooled_embeds = self._encode_prompt(prompt_dark)
        embeddings = {}
        for ev_value in ev_list:
            t = ev_value / max_negative_ev
            interp_prompt = prompt_embeds + t * (dark_embeds - prompt_embeds)
            if pooled_prompt_embeds is not None and dark_pooled_embeds is not None:
                interp_pooled = pooled_prompt_embeds + t * (
                    dark_pooled_embeds - pooled_prompt_embeds
                )
            else:
                interp_pooled = pooled_prompt_embeds
            embeddings[ev_value] = (interp_prompt, interp_pooled)

        return embeddings
