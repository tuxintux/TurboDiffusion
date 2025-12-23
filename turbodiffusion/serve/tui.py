"""
TurboDiffusion TUI Server Mode

A persistent GPU server that loads models once and provides an interactive
text-based interface for video generation.

Supports both T2V (text-to-video) and I2V (image-to-video) modes.

Usage:
    # T2V mode
    PYTHONPATH=turbodiffusion python -m serve --mode t2v --dit_path checkpoints/model.pth

    # I2V mode
    PYTHONPATH=turbodiffusion python -m serve --mode i2v \
        --high_noise_model_path checkpoints/high.pth \
        --low_noise_model_path checkpoints/low.pth
"""

import argparse
import math
import os
import sys

# Add inference directory to path for modify_model import
_inference_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference")
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as T

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# Runtime-adjustable parameters and their types/validators
RUNTIME_PARAMS = {
    "num_steps": {"type": int, "choices": [1, 2, 3, 4]},
    "num_samples": {"type": int, "min": 1},
    "num_frames": {"type": int, "min": 1},
    "sigma_max": {"type": float, "min": 0.1},
}

# Immutable launch-only parameters
LAUNCH_ONLY_PARAMS = [
    "mode", "model", "dit_path", "high_noise_model_path", "low_noise_model_path",
    "resolution", "aspect_ratio", "attention_type", "sla_topk",
    "quant_linear", "default_norm", "vae_path", "text_encoder_path",
    "boundary", "adaptive_resolution", "ode", "seed",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for TUI server mode."""
    parser = argparse.ArgumentParser(
        description="TurboDiffusion TUI Server - Interactive video generation"
    )

    # Mode selection
    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)")

    # T2V model path
    parser.add_argument("--dit_path", type=str, default=None,
                        help="Path to DiT checkpoint (required for t2v mode)")

    # I2V model paths
    parser.add_argument("--high_noise_model_path", type=str, default=None,
                        help="Path to high-noise model (required for i2v mode)")
    parser.add_argument("--low_noise_model_path", type=str, default=None,
                        help="Path to low-noise model (required for i2v mode)")
    parser.add_argument("--boundary", type=float, default=0.9,
                        help="Timestep boundary for model switching (i2v only)")

    # Model configuration
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"],
                        default=None, help="Model architecture (auto-detected from mode if not set)")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")

    # Resolution
    parser.add_argument("--resolution", default=None, type=str,
                        help="Resolution (default: 480p for t2v, 720p for i2v)")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true",
                        help="Adapt resolution to input image aspect ratio (i2v only)")

    # Attention/quantization
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla", help="Attention mechanism type")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Use quantized linear layers")
    parser.add_argument("--default_norm", action="store_true",
                        help="Use default LayerNorm/RMSNorm (not optimized)")

    # Sampling options
    parser.add_argument("--ode", action="store_true",
                        help="Use ODE sampling (sharper but less robust than SDE, i2v only)")

    # Runtime-adjustable parameters (defaults)
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of inference steps (1-4)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="Initial sigma (default: 80 for t2v, 200 for i2v)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate arguments based on mode."""
    # Set mode-dependent defaults
    if args.model is None:
        args.model = "Wan2.1-1.3B" if args.mode == "t2v" else "Wan2.2-A14B"

    if args.resolution is None:
        args.resolution = "480p" if args.mode == "t2v" else "720p"

    if args.sigma_max is None:
        args.sigma_max = 80 if args.mode == "t2v" else 200

    # Validate mode-specific requirements
    if args.mode == "t2v":
        if args.dit_path is None:
            log.error("--dit_path is required for t2v mode")
            sys.exit(1)
    else:  # i2v
        if args.high_noise_model_path is None or args.low_noise_model_path is None:
            log.error("--high_noise_model_path and --low_noise_model_path are required for i2v mode")
            sys.exit(1)

    # Validate resolution
    if args.resolution not in VIDEO_RES_SIZE_INFO:
        log.error(f"Invalid resolution: {args.resolution}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO.keys())}")
        sys.exit(1)

    if args.aspect_ratio not in VIDEO_RES_SIZE_INFO[args.resolution]:
        log.error(f"Invalid aspect ratio: {args.aspect_ratio}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO[args.resolution].keys())}")
        sys.exit(1)


def load_models_t2v(args: argparse.Namespace):
    """Load T2V model."""
    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args)
    net.cuda().eval()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT model.")
    return {"net": net}


def load_models_i2v(args: argparse.Namespace):
    """Load I2V models (high noise and low noise)."""
    log.info("Loading DiT models for I2V...")

    log.info(f"Loading high-noise model from {args.high_noise_model_path}")
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args)
    high_noise_model.cpu().eval()
    torch.cuda.empty_cache()

    log.info(f"Loading low-noise model from {args.low_noise_model_path}")
    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args)
    low_noise_model.cpu().eval()
    torch.cuda.empty_cache()

    log.success("Successfully loaded DiT models.")
    return {"high_noise_model": high_noise_model, "low_noise_model": low_noise_model}


def load_models(args: argparse.Namespace):
    """Load models based on mode."""
    log.info(f"Loading VAE from {args.vae_path}")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    log.success("Successfully loaded VAE.")

    if args.mode == "t2v":
        models = load_models_t2v(args)
    else:
        models = load_models_i2v(args)

    models["tokenizer"] = tokenizer
    return models


def generate_t2v(models: dict, args: argparse.Namespace, prompt: str, output_path: str) -> str:
    """Generate video from text prompt (T2V mode)."""
    net = models["net"]
    tokenizer = models["tokenizer"]

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    # Get text embedding
    log.info("Computing text embedding...")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)

    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs),
            "b l d -> (k b) l d",
            k=args.num_samples
        )
    }

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    for i, (t_cur, t_next) in enumerate(tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=total_steps
    )):
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

    samples = x.float()

    log.info("Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        output_path,
        fps=16
    )

    return output_path


def generate_i2v(models: dict, args: argparse.Namespace, prompt: str,
                 image_path: str, output_path: str) -> str:
    """Generate video from image and text prompt (I2V mode)."""
    high_noise_model = models["high_noise_model"]
    low_noise_model = models["low_noise_model"]
    tokenizer = models["tokenizer"]

    # Get text embedding
    log.info("Computing text embedding...")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)

    # Load and preprocess image
    log.info(f"Loading image from {image_path}")
    input_image = Image.open(image_path).convert("RGB")

    if args.adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
        log.info(f"Adaptive resolution: {w}x{h}")
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    # Image transforms
    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(input_image).unsqueeze(0).to(
        device=tensor_kwargs["device"], dtype=torch.float32
    )

    # Encode image
    log.info("Encoding image...")
    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2),
             torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)],
            dim=2
        )
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()

    # Create mask
    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w,
                      device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs),
            "b l d -> (k b) l d",
            k=args.num_samples
        ),
        "y_B_C_T_H_W": y
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    # Move high noise model to GPU
    high_noise_model.cuda()
    net = high_noise_model
    switched = False

    for i, (t_cur, t_next) in enumerate(tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=total_steps
    )):
        # Switch models at boundary
        if t_cur.item() < args.boundary and not switched:
            high_noise_model.cpu()
            torch.cuda.empty_cache()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")

        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

    samples = x.float()

    # Move models back to CPU
    if switched:
        low_noise_model.cpu()
    else:
        high_noise_model.cpu()
    torch.cuda.empty_cache()

    log.info("Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        output_path,
        fps=16
    )

    return output_path


def get_multiline_prompt() -> str:
    """Read multi-line prompt from user. Empty line finishes input."""
    lines = []
    while True:
        try:
            line = input("> " if lines else "")
            if line == "":
                break
            lines.append(line)
        except EOFError:
            if not lines:
                return None
            break
    return "\n".join(lines)


def print_help(mode: str):
    """Print help for slash commands."""
    print(f"""
Commands:
  /help              Show this help message
  /show              Show current configuration
  /set <param> <val> Set a runtime parameter
  /reset             Reset runtime parameters to defaults
  /quit              Exit the server

Runtime parameters (adjustable with /set):
  num_steps   Number of inference steps (1-4)
  num_samples Number of samples per generation
  seed        Random seed
  num_frames  Number of video frames
  sigma_max   Initial sigma for rCM

Current mode: {mode}
""")


def print_config(args: argparse.Namespace, defaults: dict):
    """Print current configuration."""
    print("\n=== Launch Configuration (immutable) ===")
    print(f"  mode:            {args.mode}")
    print(f"  model:           {args.model}")

    if args.mode == "t2v":
        print(f"  dit_path:        {args.dit_path}")
    else:
        print(f"  high_noise_model_path: {args.high_noise_model_path}")
        print(f"  low_noise_model_path:  {args.low_noise_model_path}")
        print(f"  boundary:        {args.boundary}")
        print(f"  adaptive_resolution: {args.adaptive_resolution}")
        print(f"  ode:             {args.ode}")

    print(f"  resolution:      {args.resolution}")
    print(f"  aspect_ratio:    {args.aspect_ratio}")
    print(f"  attention_type:  {args.attention_type}")
    print(f"  sla_topk:        {args.sla_topk}")
    print(f"  quant_linear:    {args.quant_linear}")
    print(f"  default_norm:    {args.default_norm}")
    print(f"  seed:            {args.seed}")

    print("\n=== Runtime Configuration (adjustable) ===")
    for param in RUNTIME_PARAMS:
        val = getattr(args, param)
        default = defaults[param]
        marker = "" if val == default else " *"
        print(f"  {param}: {val}{marker}")
    print()


def handle_set_command(args: argparse.Namespace, param: str, value: str) -> bool:
    """Handle /set command. Returns True if successful."""
    if param not in RUNTIME_PARAMS:
        print(f"Error: '{param}' is not a runtime parameter.")
        print(f"Adjustable parameters: {', '.join(RUNTIME_PARAMS.keys())}")
        return False

    spec = RUNTIME_PARAMS[param]
    try:
        typed_value = spec["type"](value)
    except ValueError:
        print(f"Error: Invalid value '{value}' for {param}")
        return False

    if "choices" in spec and typed_value not in spec["choices"]:
        print(f"Error: {param} must be one of {spec['choices']}")
        return False
    if "min" in spec and typed_value < spec["min"]:
        print(f"Error: {param} must be >= {spec['min']}")
        return False

    setattr(args, param, typed_value)
    print(f"{param} = {typed_value}")
    return True


def handle_command(cmd: str, args: argparse.Namespace, defaults: dict) -> bool:
    """Handle slash command. Returns False if should quit."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/quit":
        return False
    elif command == "/help":
        print_help(args.mode)
    elif command == "/show":
        print_config(args, defaults)
    elif command == "/set":
        if len(parts) != 3:
            print("Usage: /set <param> <value>")
        else:
            handle_set_command(args, parts[1], parts[2])
    elif command == "/reset":
        for param, default in defaults.items():
            setattr(args, param, default)
        print("Runtime parameters reset to defaults.")
    else:
        print(f"Unknown command: {command}")
        print("Type /help for available commands.")

    return True


def print_header(args: argparse.Namespace):
    """Print server header with current config."""
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    mode_str = "T2V (text-to-video)" if args.mode == "t2v" else "I2V (image-to-video)"
    print(f"""
TurboDiffusion TUI Server
=========================
Mode: {mode_str} | Model: {args.model}
Resolution: {args.resolution} ({w}x{h}) | Steps: {args.num_steps}
Type /help for commands, or enter a prompt to generate.
""")


def run_tui(models: dict, args: argparse.Namespace):
    """Main TUI loop."""
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    last_output_path = "output/generated_video.mp4"
    last_image_path = None

    print_header(args)

    while True:
        print("Prompt (empty line to generate):")
        prompt = get_multiline_prompt()

        if prompt is None:
            print("\nGoodbye!")
            break

        prompt = prompt.strip()

        if not prompt:
            continue

        if prompt.startswith("/"):
            if not handle_command(prompt, args, defaults):
                print("Goodbye!")
                break
            continue

        # For I2V mode, get image path
        image_path = None
        if args.mode == "i2v":
            try:
                default_hint = f" [{last_image_path}]" if last_image_path else ""
                user_image = input(f"Image path{default_hint}: ").strip()
                if not user_image and last_image_path:
                    image_path = last_image_path
                elif user_image:
                    image_path = user_image
                else:
                    print("Error: Image path is required for I2V mode.")
                    continue

                if not os.path.isfile(image_path):
                    print(f"Error: Image file not found: {image_path}")
                    continue

                last_image_path = image_path
            except EOFError:
                print("\nGoodbye!")
                break

        # Get output path
        try:
            user_path = input(f"Output path [{last_output_path}]: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        output_path = user_path if user_path else last_output_path

        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        # Generate
        try:
            if args.mode == "t2v":
                result_path = generate_t2v(models, args, prompt, output_path)
            else:
                result_path = generate_i2v(models, args, prompt, image_path, output_path)

            log.success(f"Generated: {result_path}")
            last_output_path = result_path
        except Exception as e:
            log.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()

        print()


def main(passed_args: argparse.Namespace = None):
    """Main entry point for TUI server."""
    args = passed_args if passed_args is not None else parse_arguments()

    validate_args(args)

    models = load_models(args)

    try:
        run_tui(models, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
