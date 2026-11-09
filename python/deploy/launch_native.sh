#!/bin/bash
# Convenience launcher for RKNN 2.5 models that use GPU ops (GridSample).
#
# It makes the OpenCL shim self-healing and then runs the given python
# script/args with LD_LIBRARY_PATH set, so callers never have to remember it.
#
#   ./launch_native.sh board_eval.py --models foo.rknn --images *.jpg
#   PYTHON=/usr/bin/python3 ./launch_native.sh -c "print('hello')"
set -u

SHIM=${SHIM:-/home/cat/opencl_shim}
PY=${PYTHON:-/home/cat/software/miniconda3/envs/rknn38/bin/python}

BLOB=$(ls /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-*g610*.so 2>/dev/null | head -1)
if [ -z "$BLOB" ]; then
    BLOB=$(ls /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-*.so 2>/dev/null | head -1)
fi

if [ -z "$BLOB" ]; then
    echo "[launch] no Mali blob found under /usr/lib/aarch64-linux-gnu/libmali-x11/" >&2
    exit 1
fi

mkdir -p "$SHIM"
if [ "$(readlink -f "$SHIM/libOpenCL.so" 2>/dev/null)" != "$(readlink -f "$BLOB")" ]; then
    ln -sf "$BLOB" "$SHIM/libOpenCL.so"
    echo "[launch] refreshed shim -> $BLOB"
fi

if ! nm -D "$SHIM/libOpenCL.so" 2>/dev/null | grep -q clImportMemoryARM; then
    echo "[launch] $SHIM/libOpenCL.so does not export clImportMemoryARM" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$SHIM:${LD_LIBRARY_PATH:-}"
exec "$PY" "$@"
