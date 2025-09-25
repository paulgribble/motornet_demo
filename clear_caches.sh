#!/bin/bash

# macOS temp (where the failing PCH lives)
rm -rf "$TMPDIR"/torchinductor_*

# User caches
rm -rf ~/.cache/torch/inductor
rm -rf ~/.cache/torch/extensions
rm -rf ~/.cache/torch/compile 2>/dev/null || true

# (Optional) Triton/other build caches if present
# rm -rf ~/.cache/triton ~/.triton 2>/dev/null || true

# (Optional) Python package cache
# pip cache purge
