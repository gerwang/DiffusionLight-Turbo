from pathlib import Path
import argparse

import numpy as np
import torch
from irtk.io import read_image, write_image

from diffusionlight_turbo.inpaint_wrapper import DiffusionLightInpaintWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffusionLight inpainting on one image.")
    parser.add_argument(
        "--input",
        default="example/Hammer_B000FK3VZ6_Wood_Lighting002/im_0007.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        default="output/im_0007_inpaint.png",
        help="Path to save the inpainted image.",
    )
    parser.add_argument(
        "--algorithm",
        default="turbo_swapping",
        help="Inpainting algorithm (normal, iterative, turbo_sdedit, turbo_pred, turbo_swapping).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Strength for SDEdit-style noise; when unset uses the algorithm default.",
    )
    parser.add_argument(
        "--sdedit_timestep",
        type=float,
        default=None,
        help="Optional diffusion timestep to add noise at before denoising.",
    )
    parser.add_argument(
        "--sdedit_timestep_is_index",
        action="store_true",
        help="Interpret sdedit_timestep as a scheduler step index instead of a raw timestep.",
    )
    parser.add_argument(
        "--ev",
        default="0,-2.5,-5",
        help="Comma-separated EV values to render (e.g. \"0,-2.5,-5\").",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the input tensor and wrapper (cuda or cpu).",
    )
    return parser.parse_args()

def parse_evs(ev_arg: str):
    ev_values = []
    for item in ev_arg.split(","):
        item = item.strip()
        if not item:
            continue
        ev_values.append(float(item))
    return ev_values

def ev_suffix(ev_value: float) -> str:
    if ev_value == 0:
        ev_str = "-00"
    else:
        ev_str = str(ev_value).replace(".", "")
    return f"_ev{ev_str}"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    evs = parse_evs(args.ev)
    if output_path.suffix:
        output_dir = output_path.parent
        output_stem = output_path.stem
        output_ext = output_path.suffix
    else:
        output_dir = output_path
        output_stem = f"{input_path.stem}_inpaint"
        output_ext = ".png"
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = DiffusionLightInpaintWrapper(
        model_option="sdxl",
        img_width=800,
        img_height=800,
        force_square=True,
        device=torch.device(args.device),
        is_cpu=args.device == "cpu",
    )
    image_np = read_image(input_path, is_srgb=False)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(args.device)

    output_image = wrapper.inpaint(
        image_tensor,
        algorithm=args.algorithm,
        strength=args.strength,
        sdedit_timestep=args.sdedit_timestep,
        sdedit_timestep_is_index=args.sdedit_timestep_is_index,
        ev=evs,
    )
    if isinstance(output_image, dict):
        for ev_value, image in output_image.items():
            out_path = output_dir / f"{output_stem}{ev_suffix(ev_value)}{output_ext}"
            image_np = np.asarray(image).astype(np.float32) / 255.0
            write_image(out_path, image_np, is_srgb=False)
            print(f"Saved {out_path}")
    else:
        out_path = output_dir / f"{output_stem}{output_ext}"
        image_np = np.asarray(output_image).astype(np.float32) / 255.0
        write_image(out_path, image_np, is_srgb=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
