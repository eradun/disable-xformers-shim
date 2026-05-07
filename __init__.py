"""
ComfyUI custom node shim that monkey-patches xformers.ops.memory_efficient_attention
to use torch's SDPA so EVA-CLIP / PuLID don't crash on cu130 containers where
xformers 0.0.35 has no working CUDA backend.

Approach (revised after pip-uninstall + sys.modules-shadow approaches failed):
  pip uninstall doesn't help - modules already imported retain their references.
  sys.modules shadow doesn't help - modules that did `from xformers.ops import ...`
    captured the original reference at import time.
  **Monkey-patching the function on the existing xformers.ops module DOES work**
  because EVA-CLIP calls `xops.memory_efficient_attention(...)` (attribute lookup
  at call time, not at import time). Reassigning the attribute makes new calls
  use our replacement.

Side effects on other engines:
  - Wan family: doesn't use xformers (sageattention path; the boot log shows
    `sageattention will not be available`, so Wan was already on torch SDPA).
  - LTX 2.3: doesn't use xformers.
  - Qwen-Image-Edit: doesn't use xformers.

Usage:
  Add this repo URL to your ComfyDeploy machine's custom-node list. Save & Build.
  The patch runs once at custom-node import time.
"""
import torch


def _torch_sdpa_replacement(q, k, v, attn_bias=None, scale=None, p=0.0, **_unused):
    """Drop-in replacement for xformers.ops.memory_efficient_attention.

    EVA-CLIP feeds (B, M, H, D); torch SDPA expects (B, H, M, D).
    Transpose in, run SDPA, transpose out.
    """
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out_t = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=attn_bias,
        dropout_p=p,
        scale=scale,
    )
    return out_t.transpose(1, 2)


def _apply_patch():
    try:
        import xformers
        import xformers.ops as xops
    except ImportError:
        print('[disable-xformers-shim] xformers not present - nothing to patch')
        return

    if getattr(xops, '_vs_patched', False):
        return
    xops._vs_original_mea = xops.memory_efficient_attention
    xops.memory_efficient_attention = _torch_sdpa_replacement
    xops._vs_patched = True
    print('[disable-xformers-shim] xformers.ops.memory_efficient_attention patched -> torch SDPA')


_apply_patch()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
