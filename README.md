# ZDepth

AI depth maps for [EffectCraft](https://github.com/storytold/effectcraft), in the spirit of
DepthScanner: a Z-depth pass from footage with Depth Anything, including flicker-free depth
over a whole clip with Video Depth Anything. It runs as an
[external effect helper](https://github.com/storytold/effectcraft/blob/main/docs/external-helpers.md),
so the model works outside EffectCraft's process: if it fails, the effect shows ⚠ and
EffectCraft carries on.

**Platforms:** Linux and macOS. Windows isn't supported yet (EffectCraft looks for an `.exe`
helper there). GPUs: NVIDIA (CUDA), AMD (ROCm build of PyTorch) and Apple Silicon (MPS); CPU
works, slowly. Tested on Linux with an NVIDIA GTX 1660 Super; ROCm and MPS are untested.

## Install

You need [`uv`](https://docs.astral.sh/uv/) (or `python3` with `venv` and `pip`), and about
5 GB of disk for PyTorch and the models.

```sh
curl -fsSLO https://github.com/koryakaflow/ZDepth/releases/latest/download/install.sh
sh install.sh
```

The script is short; read it first if you like. It downloads the latest release, checks its
SHA-256, puts it in EffectCraft's helpers folder as `zdepth/`, and runs
`effectcraft-helper-zdepth --setup Small Video-Small`: PyTorch (a few GB, the CUDA, ROCm or CPU
build for your machine) goes into `~/.local/share/zdepth/venv`, the models into the Hugging
Face cache. Options: `--models "Small Video-Small Large"`, `--version v0.1.0`, `--no-setup`;
`sh install.sh --help` lists them. **Run it again to update.**

Then in EffectCraft: **Effect ▸ Scan for Effect Helpers**. ZDepth appears under
**3D Channel ▸ ZDepth** in Effects & Presets.

To start helpers with EffectCraft every time, launch it with `EFFECTCRAFT_HELPERS=1` (for
example in the `Exec=` line of its `.desktop` file: `Exec=env EFFECTCRAFT_HELPERS=1 …`). Only
then does ZDepth also appear in the Effect menu and Quick Apply, which EffectCraft builds once
at start.

<details>
<summary>By hand</summary>

Download `zdepth.tar.gz` from the [latest release](https://github.com/koryakaflow/ZDepth/releases/latest)
and extract it into EffectCraft's helpers folder (`~/.config/effectcraft/helpers` on Linux,
`~/Library/Application Support/EffectCraft/helpers` on macOS), then run
`helpers/zdepth/effectcraft-helper-zdepth --setup Small Video-Small`. Until `--setup` has run,
the helper exits at once with a message instead of installing anything, so it never holds up
EffectCraft's start.

From a git checkout (development): run `./effectcraft-helper-zdepth --setup` in it and link
it into the helpers folder: `ln -s "$PWD/effectcraft-helper-zdepth" ~/.config/effectcraft/helpers/`.
Only one ZDepth may be in the helpers folder; `install.sh` stops if it finds another.
</details>

## Use

| Parameter | |
|---|---|
| Model | Small, Base, Large: Depth Anything V2, frame by frame. Video Small, Video Large: Video Depth Anything, one consistent depth for the whole clip. |
| Quality | The model's input size (shorter side): Draft 364, Fast 518, Detail 770, Max 1036. |
| Normalize | Per Frame maps each frame's own depth range to black–white; Per Clip uses one range for the whole clip, so the map doesn't pump when something near enters the frame. |
| Invert, Black Point, White Point | White is near by default. |
| Output | Depth (gray); Depth (Color) (Turbo colormap, for checking); Normals (from depth, with Normal Strength); Depth Matte (the layer kept between Matte Far and Matte Near, with Matte Softness, in the same 0–100 depth as the output). |

The video models go through the layer's whole source range on their first render, which takes
a while on long clips (EffectCraft's log shows the progress); after that every frame is
instant. Per Clip with the frame-by-frame models samples up to 48 frames once.

| Model | License |
|---|---|
| Small, Video Small | Apache-2.0 |
| Base, Large, Video Large | CC-BY-NC-4.0: non-commercial use only |

Weights come from the authors' Hugging Face repositories, pinned to a revision.

## GPU memory and speed

The models' own speed and memory are in their authors' READMEs
([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything));
`tools/bench.py` measures every downloaded model on your machine. What ZDepth adds:

- Video Small at Fast (518) peaks at about 2.7 GB of GPU memory (ZDepth frees the network's
  intermediates early; same output as the original code), so it runs next to EffectCraft on a
  6 GB card. Higher Quality needs more.
- The video pass also waits for EffectCraft to decode every frame of the clip.
- CUDA runs in fp16 except on GTX 16xx, where fp16 measured several times slower than fp32 and
  cuDNN's fp16 convolutions return NaNs. A result that comes out non-finite in fp16 is redone
  in fp32.
- Models leave the GPU after 60 s without a render; depth already computed stays cached, so
  only new frames load the model again.

## Troubleshooting

- **Frames look wrong after changing Model or Quality** (old results, flicker every other
  frame): EffectCraft's preview cache can keep frames from before the change. Edit ▸ Purge ▸
  All Memory & Disk Cache.
- **⚠ not enough GPU memory:** pick a lower Quality (Draft), Video Small instead of Video
  Large, or close other apps using the GPU.
- **⚠ EffectCraft could not pass frames on:** during a video pass every frame stays in
  EffectCraft's shared memory (`/dev/shm`, RAM) until the pass ends, so a long clip at a large
  preview size can fill it. Lower the preview resolution or trim the layer.
- **Nothing happens / ZDepth missing:** run `sh install.sh` (or `--setup`), then Effect ▸ Scan
  for Effect Helpers.
  The helper's messages are in EffectCraft's log (`~/.config/effectcraft/Logs/effectcraft.log`);
  `touch ~/.local/share/zdepth/debug` adds a line per render.

## Settings

Environment variables, set where EffectCraft is started:

| Variable | |
|---|---|
| `ZDEPTH_DEVICE` | `cuda`, `mps` or `cpu` instead of the automatic choice |
| `ZDEPTH_PRECISION` | `fp16` or `fp32` instead of the automatic choice |
| `ZDEPTH_IDLE_UNLOAD` | seconds without a render before models leave the GPU (default 60; 0 keeps them) |
| `ZDEPTH_TORCH_BACKEND` | for `--setup`: the PyTorch build (`auto`, `cpu`, `cu128`, `rocm6.4`, …) |
| `ZDEPTH_HOME` | where the environment lives (default `~/.local/share/zdepth`) |

## Development

```sh
python3 tests/test_protocol.py    # the helper protocol, no model (any Python with numpy)
python3 tests/test_outputs.py     # output modes, no model
~/.local/share/zdepth/venv/bin/python tools/bench.py   # speed and memory per model
```

Releases: bump `__version__` in `zdepth/__init__.py` and `version` in the effect manifest
(`zdepth/effect.py`), add `release-notes/vX.Y.Z.md`, push a `vX.Y.Z` tag. The release workflow
runs the tests, builds `zdepth.tar.gz` with `tools/make_release.sh` and publishes it with
`install.sh`.

## Uninstall

Quit EffectCraft, then run the `install.sh` that ZDepth keeps in its folder (Linux path; on
macOS `~/Library/Application Support/EffectCraft/helpers/zdepth/install.sh`):

```sh
sh ~/.config/effectcraft/helpers/zdepth/install.sh --uninstall                 # ZDepth and its PyTorch environment
sh ~/.config/effectcraft/helpers/zdepth/install.sh --uninstall --purge-models  # also its models in the Hugging Face cache
```

Models stay by default because other apps may share the Hugging Face cache. By hand: delete
`helpers/zdepth` in EffectCraft's settings folder, `~/.local/share/zdepth`, and, if you like,
`~/.cache/huggingface/hub/models--depth-anything--*`.

## License

ZDepth is licensed under the [Apache License 2.0](LICENSE). It includes modified code from
Video Depth Anything (Apache-2.0); model weights are downloaded separately under their own
licenses. See [NOTICE](NOTICE).
