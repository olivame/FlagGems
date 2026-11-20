# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.runtime import torch_device_fn")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_add_kernel(
    rank: int,
    dim: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("out_shape_dim,")
            code.writeline("index_stride,")
            code.writeline("alpha,")

            src_stride_args = ", ".join(f"src_stride_{i}: tl.int64" for i in range(rank))
            code.writeline(f"{src_stride_args}, # stride for src")

            out_stride_args = ", ".join(f"out_stride_{i}: tl.int64" for i in range(rank))
            code.writeline(f"{out_stride_args}, # stride for out")

            shape_args = ", ".join(f"src_shape_{i}: tl.int64" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

        with code.indent():
            code.writeline("pid = tl.program_id(axis=0).to(tl.int64)")
            code.writeline(
                "offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)"
            )
            code.writeline("mask = offsets < N")
            code.writeline("rem = offsets")

            for i in range(rank - 1, -1, -1):
                code.writeline(f"coord_{i} = rem % src_shape_{i}")
                code.writeline(f"rem = rem // src_shape_{i}")
            code.newline()

            src_comp = [f"coord_{i} * src_stride_{i}" for i in range(rank)]
            code.writeline(f"src_offset = {' + '.join(src_comp)}")

            code.writeline(f"index_offset = coord_{dim} * index_stride")
            code.writeline(
                "dst_dim_idx = tl.load(index + index_offset, mask=mask, other=0).to(tl.int64)"
            )

            out_comp = [
                f"dst_dim_idx * out_stride_{i}" if i == dim else f"coord_{i} * out_stride_{i}"
                for i in range(rank)
            ]
            code.writeline(f"out_offset = {' + '.join(out_comp)}")

            code.writeline(
                "valid_mask = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_shape_dim)"
            )
            code.writeline(
                "add_on = tl.load(src + src_offset, mask=valid_mask, other=0) * alpha"
            )
            code.writeline(
                "tl.atomic_add(out + out_offset, add_on, mask=valid_mask, sem='relaxed')"
            )

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = [
        "out",
        "index",
        "src",
        "dim",
        "out_shape_dim",
        "N",
        "alpha",
    ]
    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("if N == 0:")
        with code.indent():
            code.writeline("return out")
        code.writeline("src_strides = list(src.stride())")
        code.writeline("out_strides = list(out.stride())")
        code.writeline("src_shapes = list(src.shape)")
        code.writeline("index_stride = index.stride(0) if index.ndim > 0 else 0")

        code.writeline("BLOCK_SIZE = 128")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        code.writeline("with torch_device_fn.device(out.device):")
        with code.indent():
            code.writeline(f"{kernel_name}[grid](")
            with code.indent():
                code.writeline("index, src, out, N, out_shape_dim, index_stride, alpha,")
                if rank > 0:
                    s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")
                    o = ", ".join(f"out_strides[{i}]" for i in range(rank))
                    code.writeline(f"{o},")
                    sh = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{sh},")
                code.writeline("BLOCK_SIZE=BLOCK_SIZE,")
            code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any, ...],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, src, dim, out_shape_dim, N, alpha]
    shape = inputs[2].shape
    rank = len(shape)
    dim = int(inputs[3])

    code = generate_imports(code)
    code = generate_index_add_kernel(rank, dim, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexAddFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_add_wrapper",
                "_index_add_jit_function",
                code,
            )

            file_name = f"index_add_v2_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_v2_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_add_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        rank = args[2].ndim
        dim = int(args[3])
        return f"{rank}_dim_{dim}"


_index_add_func = IndexAddFunction()


def _validate_index_add_args(inp, dim, index, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    if index.numel() > 0:
        assert bool(
            ((0 <= index) & (index < inp.size(dim))).all().item()
        ), "0 <= index < self.size(dim)"
    return dim


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX_ADD")
    dim = _validate_index_add_args(inp, dim, index, src)
    index_1d = index.reshape(-1)

    orig_dtype = inp.dtype
    needs_cast = orig_dtype == torch.bfloat16
    if needs_cast:
        out = inp.clone().to(torch.float32)
        src_work = src.to(torch.float32)
    else:
        out = inp.clone()
        src_work = src

    N = src_work.numel()
    if N == 0:
        return out.to(orig_dtype) if needs_cast else out

    _index_add_func(
        out,
        index_1d,
        src_work,
        dim,
        out.size(dim),
        N,
        alpha,
    )
    if needs_cast:
        return out.to(orig_dtype)
    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX_ADD_")
    dim = _validate_index_add_args(inp, dim, index, src)
    index_1d = index.reshape(-1)

    N = src.numel()
    if N == 0:
        return inp

    orig_dtype = inp.dtype
    needs_cast = orig_dtype == torch.bfloat16
    if needs_cast:
        out = inp.to(torch.float32)
        src_work = src.to(torch.float32)
    else:
        out = inp
        src_work = src

    _index_add_func(
        out,
        index_1d,
        src_work,
        dim,
        out.size(dim),
        N,
        alpha,
    )
    if needs_cast:
        inp.copy_(out.to(orig_dtype))
    return inp
