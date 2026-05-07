# disable-xformers-shim

Tiny ComfyUI custom node that uninstalls `xformers` at ComfyUI startup so
modules with broken xformers backends (e.g. PuLID-Flux II's EVA-CLIP on
ComfyDeploy machine builder v4 with `torch 2.11.0+cu130`) fall back to
PyTorch's native `scaled_dot_product_attention`.

No user-facing nodes — this module exists purely for its side effect at
import time.

## When you need this

You see errors like:

```
NotImplementedError: No operator found for `memory_efficient_attention_forward`
xFormers wasn't build with CUDA support
```

…on a ComfyDeploy machine where `torch.cuda.is_available()` returns True
and the GPU is otherwise functional. The container has xformers installed
but its CUDA kernels don't match the runtime's torch+cuda version.

## Install

Add the repo URL to your ComfyDeploy machine's custom-node list. Save &
Build. The shim runs once on container start.

## Side effects

`xformers` is removed from the container's Python environment. If any
other engine on your machine genuinely requires xformers (Wan, LTX, SDXL,
Flux, etc. — most do not on builder v4 since sage-attention/SDPA is the
default), it will fall back to torch SDPA. No correctness change; small
perf delta in some cases.

## License

Apache 2.0.
