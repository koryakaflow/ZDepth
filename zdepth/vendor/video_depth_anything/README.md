Model code from [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
(commit 4f5ae23172ba60fd7bc11ef671cca678842c7072), Copyright (2025) Bytedance Ltd. and/or its
affiliates, Apache-2.0 (see LICENSE). Only the network is vendored, unchanged; the windowed
inference and alignment of `video_depth.py` are reimplemented as a stream in `zdepth/video.py`.

Local change: `dinov2_layers/attention.py` computes attention with
`torch.nn.functional.scaled_dot_product_attention` instead of an explicit `q @ k^T` (marked
"ZDepth"), so a 32-frame window fits a 6 GB GPU.

More local changes, all marked "ZDepth", none changing results (checked against the
unchanged code paths: max difference 4e-6 in fp32):

- `motion_module/motion_module.py`: the temporal transformer blocks run a slice of spatial
  positions at a time (they attend across frames only).
- `motion_module/attention.py`: fused attention when no mask or fp32 softmax is asked for.
- `dpt_temporal.py`: encoder features and the projected layers are freed as soon as they are
  used.
