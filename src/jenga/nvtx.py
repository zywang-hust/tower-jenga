"""NVTX ranges for `nsys` timelines of the serving path (docs/throughput.md §8.2). Loads libnvToolsExt from the
nvidia-nvtx wheel that jaxlib pulls in (or the system library) and is a no-op without one, so the server runs
unchanged outside a profile: a push/pop pair costs about 1 us when no tool is attached."""
from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from pathlib import Path


def _load():
    names = [Path(p) / "nvidia/nvtx/lib/libnvToolsExt.so.1" for p in sys.path] + ["libnvToolsExt.so.1"]
    for name in names:
        try:
            lib = ctypes.CDLL(str(name))
        except OSError:
            continue
        lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        lib.nvtxRangePushA.restype = ctypes.c_int
        lib.nvtxRangePop.restype = ctypes.c_int
        return lib
    return None


_LIB = _load()


def push(name: str) -> None:
    if _LIB is not None:
        _LIB.nvtxRangePushA(name.encode())


def pop() -> None:
    if _LIB is not None:
        _LIB.nvtxRangePop()


@contextmanager
def range(name: str):  # noqa: A001
    push(name)
    try:
        yield
    finally:
        pop()
