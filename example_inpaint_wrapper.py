from pathlib import Path
import argparse

from inpaint_wrapper import DiffusionLightInpaintWrapper


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
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = DiffusionLightInpaintWrapper(
        model_option="sdxl",
        img_width=800,
        img_height=800,
        force_square=True,
    )
    output_image = wrapper.inpaint(str(input_path), algorithm="turbo_swapping")
    output_image.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
