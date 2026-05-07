"""
ComfyUI custom node shim that uninstalls xformers at boot.

Why this exists:
  ComfyDeploy machine builder v4 ships pytorch 2.11.0+cu130 with xformers
  0.0.35 — but xformers 0.0.35 has no CUDA backend for cu130. Modules
  like EVA-CLIP (used by PuLID-Flux II) call xformers.ops.memory_efficient_
  attention unconditionally and crash with `NotImplementedError: No operator
  found for memory_efficient_attention_forward`. Their fallback path only
  triggers when `import xformers` itself fails.

  Solution: uninstall xformers at ComfyUI startup, BEFORE any other custom
  node imports it. EVA-CLIP's `try: import xformers` then fails cleanly,
  setting `XFORMERS_IS_AVAILBLE = False`, which routes attention through
  torch's built-in scaled_dot_product_attention (SDPA) — fast on cu130.

  No user-facing nodes; this module exists purely for its side effect at
  import time.

Side effects on other engines:
  - Wan 2.2 family (s2v, animate, a14b, 5b): use sage-attention, not
    xformers. Boot log already shows "sageattention will not be available"
    on builder v4, so Wan was relying on torch SDPA anyway. No change.
  - LTX 2.3: torch SDPA. No change.
  - Qwen-Image-Edit: torch SDPA. No change.

Usage:
  Add `https://github.com/<your-user>/disable-xformers-shim` to your
  ComfyDeploy machine's custom-node list. Save & Build. Done.
"""
import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'xformers'],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print('[disable-xformers-shim] pip uninstall xformers ->', result.returncode)
    if result.stdout:
        print('[disable-xformers-shim]', result.stdout.strip()[:200])
except Exception as e:
    print('[disable-xformers-shim] uninstall failed (non-fatal):', e)

# Belt-and-suspenders: even if uninstall didn't take effect (cached wheels,
# read-only fs, etc), force-fail any future `import xformers`.
class _XformersShim:
    def __getattr__(self, name):
        raise ImportError("xformers disabled by disable-xformers-shim")

# Only shadow if not already disabled by uninstall
try:
    import xformers  # type: ignore
    # If we got here, uninstall didn't take. Shadow the module.
    sys.modules['xformers'] = _XformersShim()  # type: ignore
    sys.modules['xformers.ops'] = _XformersShim()  # type: ignore
    print('[disable-xformers-shim] xformers shadowed')
except ImportError:
    print('[disable-xformers-shim] xformers cleanly removed')

# ComfyUI custom-node interface — empty maps, no user-facing nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
