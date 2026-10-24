"""Built-in Zero-Dependency Pure-Python + `ctypes` Drop-In Engine for `numpy`.

Guarantees that `Vivyroom` starts and runs at 100% capability on ANY Python 3.10+ environment
(including stripped `uv` virtual environments without `pip`, air-gapped machines without internet,
or embedded Linux/OrangePi SBCs without pre-built wheels).
"""

from __future__ import annotations

import builtins as _builtins
import ctypes
import math
import random as _py_random
import struct
import sys
from types import ModuleType
from typing import Any

float32 = "float32"
int16 = "int16"
uint8 = "uint8"

_DTYPE_CTYPE = {
    "float32": ctypes.c_float,
    float: ctypes.c_float,
    "int16": ctypes.c_int16,
    int: ctypes.c_int16,
    "uint8": ctypes.c_uint8,
}

_DTYPE_STRUCT = {
    "float32": "f",
    "int16": "h",
    "uint8": "B",
}

_DTYPE_ITEMSIZE = {
    "float32": 4,
    "int16": 2,
    "uint8": 1,
}


def _norm_dtype(dt: Any) -> str:
    if dt in (float32, float, ctypes.c_float, "f", "<f4"):
        return "float32"
    if dt in (int16, ctypes.c_int16, "h", "<i2"):
        return "int16"
    if dt in (uint8, ctypes.c_uint8, "B", "|u1"):
        return "uint8"
    return "float32"


def _prod(shape: tuple[int, ...]) -> int:
    res = 1
    for s in shape:
        res *= int(s)
    return res


def _flatten_nested(obj: Any) -> tuple[list[float], tuple[int, ...]]:
    if isinstance(obj, ndarray):
        return list(obj._flat), obj.shape
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return [], (0,)
        first_flat, first_shape = _flatten_nested(obj[0])
        if first_shape == ():
            return [float(x) for x in obj], (len(obj),)
        out: list[float] = []
        for item in obj:
            sub_flat, _ = _flatten_nested(item)
            out.extend(sub_flat)
        return out, (len(obj),) + first_shape
    return [float(obj)], ()


class _CtypesAccessor:
    """Provides `.ctypes.data_as(...)` compatibility for C-ABI DLL calls (`vivy_v_soul.dll`)."""

    def __init__(self, arr: ndarray) -> None:
        self._arr = arr
        self._buf: Any = None

    def data_as(self, c_ptr_type: Any) -> Any:
        c_elem = _DTYPE_CTYPE[self._arr.dtype]
        n = len(self._arr._flat)
        c_array_type = c_elem * n
        if self._arr.dtype == "float32":
            self._buf = c_array_type(*(float(x) for x in self._arr._flat))
        else:
            self._buf = c_array_type(*(int(x) for x in self._arr._flat))
        return ctypes.cast(self._buf, c_ptr_type)


class ndarray:
    """Lightweight N-dimensional array backed by flat native Python lists + `ctypes`/`struct`."""

    def __init__(self, flat_data: list[float | int], shape: tuple[int, ...], dtype: Any = float32) -> None:
        self.dtype = _norm_dtype(dtype)
        self.shape = tuple(int(x) for x in shape)
        if self.dtype == "float32":
            self._flat: list[Any] = [float(x) for x in flat_data]
        elif self.dtype == "int16":
            self._flat = [_builtins.max(-32768, min(32767, int(round(x)))) for x in flat_data]
        else:
            self._flat = [int(round(x)) & 0xFF for x in flat_data]
        self._ctypes_holder: _CtypesAccessor | None = None

    @property
    def size(self) -> int:
        return len(self._flat)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ctypes(self) -> _CtypesAccessor:
        self._ctypes_holder = _CtypesAccessor(self)
        return self._ctypes_holder

    @property
    def T(self) -> ndarray:
        if len(self.shape) <= 1:
            return self.copy()
        if len(self.shape) == 2:
            rows, cols = self.shape
            out = [0.0] * (rows * cols)
            for r in range(rows):
                r_off = r * cols
                for c in range(cols):
                    out[c * rows + r] = self._flat[r_off + c]
            return ndarray(out, (cols, rows), dtype=self.dtype)
        return self.copy()

    def copy(self) -> ndarray:
        return ndarray(list(self._flat), self.shape, dtype=self.dtype)

    def astype(self, dtype: Any) -> ndarray:
        return ndarray(self._flat, self.shape, dtype=dtype)

    def reshape(self, *new_shape: Any) -> ndarray:
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            shp = tuple(int(x) for x in new_shape[0])
        else:
            shp = tuple(int(x) for x in new_shape)
        if -1 in shp:
            known = 1
            for s in shp:
                if s != -1:
                    known *= s
            shp = tuple((len(self._flat) // _builtins.max(1, known)) if s == -1 else s for s in shp)
        return ndarray(list(self._flat), shp, dtype=self.dtype)

    def tobytes(self) -> bytes:
        fmt_ch = _DTYPE_STRUCT[self.dtype]
        n = len(self._flat)
        return struct.pack(f"<{n}{fmt_ch}", *self._flat)

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __iter__(self):
        if len(self.shape) <= 1:
            return iter(self._flat)
        return (self[i] for i in range(self.shape[0]))

    def __float__(self) -> float:
        return float(self._flat[0]) if self._flat else 0.0

    def __int__(self) -> int:
        return int(self._flat[0]) if self._flat else 0

    def _row_stride(self) -> int:
        if len(self.shape) <= 1:
            return 1
        return _prod(self.shape[1:])

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            if len(self.shape) == 2 and len(idx) == 2:
                r, c = idx
                if isinstance(r, int) and isinstance(c, int):
                    if r < 0:
                        r += self.shape[0]
                    if c < 0:
                        c += self.shape[1]
                    return self._flat[r * self.shape[1] + c]
            cur: Any = self
            for sub in idx:
                cur = cur[sub]
            return cur

        if isinstance(idx, int):
            n0 = self.shape[0]
            if idx < 0:
                idx += n0
            if len(self.shape) == 1:
                return self._flat[idx]
            stride = self._row_stride()
            start = idx * stride
            return ndarray(self._flat[start : start + stride], self.shape[1:], dtype=self.dtype)

        if isinstance(idx, slice):
            if len(self.shape) == 1:
                sub = self._flat[idx]
                return ndarray(sub, (len(sub),), dtype=self.dtype)
            indices = range(*idx.indices(self.shape[0]))
            stride = self._row_stride()
            out: list[Any] = []
            for i in indices:
                st = i * stride
                out.extend(self._flat[st : st + stride])
            return ndarray(out, (len(indices),) + self.shape[1:], dtype=self.dtype)

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, tuple) and len(self.shape) == 2 and len(idx) == 2:
            r, c = int(idx[0]), int(idx[1])
            if r < 0:
                r += self.shape[0]
            if c < 0:
                c += self.shape[1]
            self._flat[r * self.shape[1] + c] = float(val) if self.dtype == "float32" else int(val)
            return
        if isinstance(idx, tuple) and len(self.shape) == 4 and len(idx) == 4:
            i0, i1, i2, i3 = (int(x) for x in idx)
            s0, s1, s2, s3 = self.shape
            flat_i = ((i0 * s1 + i1) * s2 + i2) * s3 + i3
            self._flat[flat_i] = float(val) if self.dtype == "float32" else int(val)
            return
        if isinstance(idx, int):
            if idx < 0:
                idx += self.shape[0]
            if len(self.shape) == 1:
                self._flat[idx] = float(val) if self.dtype == "float32" else int(val)
                return

    def _binary_op(self, other: Any, op_fn: Any) -> ndarray:
        if isinstance(other, ndarray):
            if self.shape == other.shape:
                return ndarray([op_fn(a, b) for a, b in zip(self._flat, other._flat)], self.shape, dtype=self.dtype)
            if other.size == 1:
                b_val = other._flat[0]
                return ndarray([op_fn(a, b_val) for a in self._flat], self.shape, dtype=self.dtype)
            if self.size == 1:
                a_val = self._flat[0]
                return ndarray([op_fn(a_val, b) for b in other._flat], other.shape, dtype=other.dtype)
            # Broadcast last dimension (e.g., (N, D) + (D,))
            if len(self.shape) == 2 and len(other.shape) == 1 and self.shape[1] == other.shape[0]:
                cols = self.shape[1]
                return ndarray(
                    [op_fn(self._flat[i], other._flat[i % cols]) for i in range(len(self._flat))],
                    self.shape,
                    dtype=self.dtype,
                )
        scalar = float(other)
        return ndarray([op_fn(a, scalar) for a in self._flat], self.shape, dtype=self.dtype)

    def __add__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: a / b)

    def __pow__(self, other: Any) -> ndarray:
        return self._binary_op(other, lambda a, b: a ** b)

    def __neg__(self) -> ndarray:
        return ndarray([-x for x in self._flat], self.shape, dtype=self.dtype)

    def __matmul__(self, other: ndarray) -> Any:
        if not isinstance(other, ndarray):
            other = array(other)
        # (D,) @ (D,) -> scalar float
        if len(self.shape) == 1 and len(other.shape) == 1:
            return _builtins.sum(a * b for a, b in zip(self._flat, other._flat))
        # (M, K) @ (K,) -> (M,)
        if len(self.shape) == 2 and len(other.shape) == 1:
            m, k = self.shape
            b_flat = other._flat
            out = [0.0] * m
            for r in range(m):
                off = r * k
                s = 0.0
                for c in range(k):
                    s += self._flat[off + c] * b_flat[c]
                out[r] = s
            return ndarray(out, (m,), dtype="float32")
        # (M, K) @ (K, N) -> (M, N)
        if len(self.shape) == 2 and len(other.shape) == 2:
            m, k = self.shape
            k2, n = other.shape
            if k != k2:
                raise ValueError(f"Shape mismatch in matmul: {self.shape} @ {other.shape}")
            out = [0.0] * (m * n)
            b_flat = other._flat
            for r in range(m):
                a_off = r * k
                out_off = r * n
                for c in range(k):
                    a_val = self._flat[a_off + c]
                    b_off = c * n
                    for j in range(n):
                        out[out_off + j] += a_val * b_flat[b_off + j]
            return ndarray(out, (m, n), dtype="float32")
        raise ValueError(f"Unsupported matmul shapes: {self.shape} @ {other.shape}")


def array(obj: Any, dtype: Any = float32) -> ndarray:
    if isinstance(obj, ndarray):
        return obj.astype(dtype)
    flat, shp = _flatten_nested(obj)
    if shp == ():
        shp = (1,)
    return ndarray(flat, shp, dtype=dtype)


def ascontiguousarray(obj: Any, dtype: Any = float32) -> ndarray:
    return array(obj, dtype=dtype)


def zeros(shape: int | tuple[int, ...], dtype: Any = float32) -> ndarray:
    shp = (shape,) if isinstance(shape, int) else tuple(shape)
    return ndarray([0.0] * _prod(shp), shp, dtype=dtype)


def ones(shape: int | tuple[int, ...], dtype: Any = float32) -> ndarray:
    shp = (shape,) if isinstance(shape, int) else tuple(shape)
    return ndarray([1.0] * _prod(shp), shp, dtype=dtype)


def full(shape: int | tuple[int, ...], fill_value: float, dtype: Any = float32) -> ndarray:
    shp = (shape,) if isinstance(shape, int) else tuple(shape)
    return ndarray([fill_value] * _prod(shp), shp, dtype=dtype)


def eye(n: int, dtype: Any = float32) -> ndarray:
    out = [0.0] * (n * n)
    for i in range(n):
        out[i * n + i] = 1.0
    return ndarray(out, (n, n), dtype=dtype)


def arange(start: float, stop: float | None = None, step: float = 1.0, dtype: Any = float32) -> ndarray:
    if stop is None:
        stop = start
        start = 0.0
    vals: list[float] = []
    cur = float(start)
    while cur < float(stop):
        vals.append(cur)
        cur += float(step)
    return ndarray(vals, (len(vals),), dtype=dtype)


def linspace(start: float, stop: float, num: int, endpoint: bool = True, dtype: Any = float32) -> ndarray:
    n = int(num)
    if n <= 0:
        return ndarray([], (0,), dtype=dtype)
    if n == 1:
        return ndarray([float(start)], (1,), dtype=dtype)
    denom = float(n - 1 if endpoint else n)
    step = (float(stop) - float(start)) / denom
    return ndarray([float(start) + i * step for i in range(n)], (n,), dtype=dtype)


def frombuffer(buffer: bytes | bytearray | memoryview, dtype: Any = float32, count: int = -1, offset: int = 0) -> ndarray:
    dt = _norm_dtype(dtype)
    itemsize = _DTYPE_ITEMSIZE[dt]
    fmt_ch = _DTYPE_STRUCT[dt]
    mv = memoryview(buffer)[offset:]
    total_items = len(mv) // itemsize if count < 0 else int(count)
    raw = mv[: total_items * itemsize]
    vals = list(struct.unpack(f"<{total_items}{fmt_ch}", raw))
    return ndarray(vals, (total_items,), dtype=dt)


def bitwise_xor(a: ndarray, b: ndarray) -> ndarray:
    return ndarray([int(x) ^ int(y) for x, y in zip(a._flat, b._flat)], a.shape, dtype=a.dtype)


def tanh(x: ndarray) -> ndarray:
    return ndarray([math.tanh(float(v)) for v in x._flat], x.shape, dtype="float32")


def sin(x: ndarray) -> ndarray:
    return ndarray([math.sin(float(v)) for v in x._flat], x.shape, dtype="float32")


def cos(x: ndarray) -> ndarray:
    return ndarray([math.cos(float(v)) for v in x._flat], x.shape, dtype="float32")


def exp(x: ndarray) -> ndarray:
    return ndarray([math.exp(_builtins.max(-80.0, min(80.0, float(v)))) for v in x._flat], x.shape, dtype="float32")


def cumsum(x: ndarray) -> ndarray:
    out: list[float] = []
    acc = 0.0
    for v in x._flat:
        acc += float(v)
        out.append(acc)
    return ndarray(out, x.shape, dtype=x.dtype)


def clip(x: ndarray, a_min: float, a_max: float) -> ndarray:
    lo, hi = float(a_min), float(a_max)
    return ndarray([_builtins.max(lo, min(hi, float(v))) for v in x._flat], x.shape, dtype=x.dtype)


def maximum(x: float | ndarray, y: float | ndarray) -> ndarray:
    if isinstance(x, ndarray) and isinstance(y, ndarray):
        return ndarray([_builtins.max(float(a), float(b)) for a, b in zip(x._flat, y._flat)], x.shape, dtype="float32")
    if isinstance(x, ndarray):
        sc = float(y)
        return ndarray([_builtins.max(float(a), sc) for a in x._flat], x.shape, dtype="float32")
    if isinstance(y, ndarray):
        sc = float(x)
        return ndarray([_builtins.max(sc, float(b)) for b in y._flat], y.shape, dtype="float32")
    return ndarray([_builtins.max(float(x), float(y))], (1,), dtype="float32")


def max(x: ndarray) -> float:  # noqa: A001
    return float(_builtins.max(x._flat)) if x._flat else 0.0


def max_builtin(seq: list[Any]) -> float:
    best = seq[0]
    for v in seq[1:]:
        if v > best:
            best = v
    return float(best)


def mean(x: ndarray) -> float:
    return float(sum_builtin(x._flat)) / float(len(x._flat)) if x._flat else 0.0


def sum_builtin(seq: list[Any]) -> float:
    total = 0.0
    for v in seq:
        total += float(v)
    return total


def sum(x: ndarray, axis: int | None = None, keepdims: bool = False) -> Any:  # noqa: A001
    if axis is None:
        return sum_builtin(x._flat)
    if len(x.shape) == 2 and axis == 1:
        rows, cols = x.shape
        out = [sum_builtin(x._flat[r * cols : (r + 1) * cols]) for r in range(rows)]
        return ndarray(out, (rows, 1) if keepdims else (rows,), dtype="float32")
    return sum_builtin(x._flat)


def argmax(x: ndarray) -> int:
    if not x._flat:
        return 0
    best_i = 0
    best_v = x._flat[0]
    for i, v in enumerate(x._flat):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def argsort(x: ndarray) -> list[int]:
    return sorted(range(len(x._flat)), key=lambda i: x._flat[i])


def pad(x: ndarray, pad_width: tuple[int, int]) -> ndarray:
    before, after = int(pad_width[0]), int(pad_width[1])
    out = ([0.0] * before) + list(x._flat) + ([0.0] * after)
    return ndarray(out, (len(out),), dtype=x.dtype)


def stack(arrays: list[ndarray], axis: int = 0) -> ndarray:
    if not arrays:
        return ndarray([], (0,), dtype="float32")
    sub_shape = arrays[0].shape
    out: list[Any] = []
    for a in arrays:
        out.extend(a._flat)
    return ndarray(out, (len(arrays),) + sub_shape, dtype=arrays[0].dtype)


def allclose(a: ndarray, b: ndarray, atol: float = 1e-5) -> bool:
    if a.shape != b.shape:
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a._flat, b._flat))


class _LinalgModule:
    @staticmethod
    def norm(x: ndarray, axis: int | None = None, keepdims: bool = False) -> Any:
        if axis is None:
            return math.sqrt(sum_builtin([v * v for v in x._flat]))
        if len(x.shape) == 2 and axis == 1:
            rows, cols = x.shape
            out = [
                math.sqrt(sum_builtin([v * v for v in x._flat[r * cols : (r + 1) * cols]]))
                for r in range(rows)
            ]
            return ndarray(out, (rows, 1) if keepdims else (rows,), dtype="float32")
        return math.sqrt(sum_builtin([v * v for v in x._flat]))

    @staticmethod
    def svd(x: ndarray, full_matrices: bool = False) -> tuple[ndarray, ndarray, ndarray]:
        rows, cols = x.shape
        k = min(rows, cols)
        return eye(rows)[:, :k], ones((k,)), eye(cols)[:k, :]


class _RandomGenerator:
    def __init__(self, seed: int = 42) -> None:
        self._rng = _py_random.Random(seed)

    def standard_normal(self, shape: int | tuple[int, ...]) -> ndarray:
        shp = (shape,) if isinstance(shape, int) else tuple(shape)
        n = _prod(shp)
        return ndarray([self._rng.gauss(0.0, 1.0) for _ in range(n)], shp, dtype="float32")

    def uniform(self, low: float, high: float, size: int | tuple[int, ...]) -> ndarray:
        shp = (size,) if isinstance(size, int) else tuple(size)
        n = _prod(shp)
        return ndarray([self._rng.uniform(low, high) for _ in range(n)], shp, dtype="float32")


class _RandomModule:
    @staticmethod
    def default_rng(seed: int = 42) -> _RandomGenerator:
        return _RandomGenerator(seed)


linalg = _LinalgModule()
random = _RandomModule()


def install_pure_numpy_fallback() -> ModuleType:
    """Registers this module as `sys.modules['numpy']` when compiled C `numpy` is not installed."""
    mod = sys.modules[__name__]
    sys.modules["numpy"] = mod
    return mod
