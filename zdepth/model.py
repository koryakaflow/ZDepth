"""Depth Anything V2 through transformers.

Weights come from the original Hugging Face repositories, pinned to a revision, into the
Hugging Face cache. Small is Apache-2.0; Base and Large are CC-BY-NC-4.0 (non-commercial).
"""

import os
import sys
import threading

import numpy as np

# name -> (repository, revision)
MODELS = {
    "Small": ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"),
    "Base": ("depth-anything/Depth-Anything-V2-Base-hf", "b1958afc87fb45a9e3746cb387596094de553ed8"),
    "Large": ("depth-anything/Depth-Anything-V2-Large-hf", "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"),
}
NONCOMMERCIAL = {"Base", "Large"}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH = 14

_lock = threading.Lock()
_loaded = {}  # name -> model; one at a time, VRAM is scarce


def device():
    import torch

    want = os.environ.get("ZDEPTH_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():  # NVIDIA, and AMD on ROCm builds
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def precision(dev):
    """Whether to run in fp16 (autocast): on CUDA unless ZDEPTH_PRECISION=fp32, except GTX 16xx.
    Those (Turing without tensor cores) measured 3-4x slower in fp16 than fp32 with these
    networks, and cuDNN's default fp16 convolutions there return NaNs. ZDEPTH_PRECISION=fp16
    forces it (benchmark mode then picks convolutions that mostly don't); a result that still
    comes out non-finite is redone in fp32."""
    import torch

    want = os.environ.get("ZDEPTH_PRECISION", "auto")
    if dev.type != "cuda" or want == "fp32":
        return False
    gtx16 = "GTX 16" in torch.cuda.get_device_name(dev)
    if gtx16:
        torch.backends.cudnn.benchmark = True
    return want == "fp16" or not gtx16


def download(name):
    from huggingface_hub import snapshot_download

    repo, rev = MODELS[name]
    return snapshot_download(repo, revision=rev)


def _load(name):
    with _lock:
        if name in _loaded:
            return _loaded[name]
        import torch
        from transformers import AutoModelForDepthEstimation

        from . import video

        _loaded.clear()
        video.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        repo, rev = MODELS[name]
        dev = device()
        print(f"zdepth: loading {repo} on {dev}", file=sys.stderr)
        model = AutoModelForDepthEstimation.from_pretrained(repo, revision=rev).to(dev).eval()
        _loaded[name] = model
        return model


def unload():
    with _lock:
        _loaded.clear()


def release():
    """Unload every model and hand the GPU memory back. True when something was loaded."""
    from . import video

    had = bool(_loaded or video._loaded)
    unload()
    video.unload()
    if had and "torch" in sys.modules:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return had


def _input_size(h, w, short_side):
    """Shorter side to `short_side`, aspect kept, both sides multiples of the patch size."""
    s = short_side / min(h, w)
    return (max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH))


def predict(rgb, name, short_side=518):
    """Relative inverse depth (larger = nearer) for straight RGB in 0..1, HxWx3 -> HxW float32."""
    import torch
    import torch.nn.functional as F

    model = _load(name)
    dev = next(model.parameters()).device
    half = precision(dev)
    h, w = rgb.shape[:2]
    with torch.inference_mode():
        x = torch.from_numpy(np.clip(rgb, 0.0, 1.0)).to(dev).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=_input_size(h, w, short_side), mode="bicubic", align_corners=False)
        mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast(device_type=dev.type, enabled=half):
            d = model(pixel_values=x).predicted_depth.float()
        if half and not torch.isfinite(d).all():
            print("zdepth: fp16 gave non-finite depth, redoing the frame in fp32", file=sys.stderr)
            d = model(pixel_values=x).predicted_depth.float()
        d = F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)
        return d[0, 0].cpu().numpy()


def setup(names):
    from . import video

    for name in names:
        name = name.replace("-", " ").replace("_", " ").title()  # "video-small" -> "Video Small"
        if name in MODELS:
            path = download(name)
        elif name in video.MODELS:
            path = video.download(name)
        else:
            print(f"unknown model {name!r}; choose from {', '.join([*MODELS, *video.MODELS])}", file=sys.stderr)
            return 2
        if name in NONCOMMERCIAL | video.NONCOMMERCIAL:
            print(f"note: {name} is CC-BY-NC-4.0, for non-commercial use only", file=sys.stderr)
        print(f"{name}: {path}", file=sys.stderr)
    print(f"device: {device()}", file=sys.stderr)
    return 0
