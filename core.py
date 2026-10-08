"""Core pipeline: text -> image (SDXL-Turbo) -> depth (Depth Anything) -> parallax video."""
import time
import cv2
import numpy as np
import torch
import imageio
from PIL import Image
from diffusers import AutoPipelineForText2Image
from transformers import pipeline as hf_pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_t2i = None
_depth = None


def get_t2i():
    global _t2i
    if _t2i is None:
        _t2i = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=DTYPE,
            variant="fp16" if DEVICE == "cuda" else None,
        ).to(DEVICE)
    return _t2i


def get_depth():
    global _depth
    if _depth is None:
        _depth = hf_pipeline(
            "depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=0 if DEVICE == "cuda" else -1,
        )
    return _depth


def generate_image(prompt, seed=0, steps=2, size=512):
    """SDXL-Turbo wants guidance_scale=0 and 1-4 steps."""
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    t0 = time.time()
    img = get_t2i()(
        prompt=prompt, num_inference_steps=int(steps), guidance_scale=0.0,
        width=size, height=size, generator=gen,
    ).images[0]
    return img, time.time() - t0


def estimate_depth(img: Image.Image):
    """Returns float32 array in [0,1], where 1 = nearest (Depth Anything is inverse depth)."""
    out = get_depth()(img)["depth"].resize(img.size, Image.BICUBIC)
    d = np.asarray(out).astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    # Smooth slightly so the warp doesn't tear at noisy depth edges
    d = cv2.GaussianBlur(d, (0, 0), 2.0)
    return d


def _camera_offset(motion, t):
    """t in [0,1] -> (dx, dy, zoom) in normalized units."""
    s = np.sin(2 * np.pi * t)  # loops seamlessly
    c = np.cos(2 * np.pi * t)
    if motion == "Orbit":
        return 0.5 * c, 0.3 * s, 0.0
    if motion == "Pan (left-right)":
        return s, 0.0, 0.0
    if motion == "Dolly zoom":
        return 0.0, 0.0, 0.5 * (1 - c) / 2
    return 0.0, 0.0, 0.0


def render_parallax(img, depth, motion="Orbit", strength=28, seconds=3, fps=24, out_path="parallax.mp4"):
    """Backward-warp each frame: pixels shift in proportion to depth (near moves more)."""
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cx, cy = w / 2, h / 2
    centered = depth - 0.5
    n = int(seconds * fps)
    frames = []
    for i in range(n):
        dx, dy, z = _camera_offset(motion, i / n)
        map_x = xs - dx * strength * centered
        map_y = ys - dy * strength * centered
        if z:
            scale = 1.0 + z * (0.3 + depth)  # near pixels scale more -> dolly feel
            map_x = cx + (map_x - cx) / scale
            map_y = cy + (map_y - cy) / scale
        map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
        frame = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(frame)
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return out_path


def depth_to_image(depth):
    return Image.fromarray((depth * 255).astype(np.uint8))
