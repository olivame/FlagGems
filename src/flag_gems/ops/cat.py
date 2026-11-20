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

import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def _is_float8_e8m0fnu(dtype: torch.dtype) -> bool:
    return str(dtype) == "torch.float8_e8m0fnu"


def _should_use_uint8_view_path(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> bool:
    if len(A) == 0:
        return False
    first_dtype = A[0].dtype
    if not _is_float8_e8m0fnu(first_dtype):
        return False
    if A[0].element_size() != 1:
        return False
    for tensor in A[1:]:
        if tensor.dtype != first_dtype or tensor.element_size() != 1:
            return False
    return True


def _cat_build_working_list_uint8_view(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int,
):
    original_dtype = A[0].dtype
    A_u8 = [tensor.view(torch.uint8) for tensor in A]
    mode, payload = _cat_build_working_list(A_u8, dim)
    return mode, payload, original_dtype


@triton.jit
def cat_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a,
    dim_size_in_b,
    dim_size_in_c,
    dim_size_in_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a: tl.int64,
    dim_offset_b: tl.int64,
    dim_offset_c: tl.int64,
    dim_offset_d: tl.int64,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_size_in = tl.cast(dim_size_in_a, tl.int64)
        dim_offset = tl.cast(dim_offset_a, tl.int64)
        total_elements = tl.cast(total_elements_a, tl.int64)
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_size_in = tl.cast(dim_size_in_b, tl.int64)
        dim_offset = tl.cast(dim_offset_b, tl.int64)
        total_elements = tl.cast(total_elements_b, tl.int64)
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_size_in = tl.cast(dim_size_in_c, tl.int64)
        dim_offset = tl.cast(dim_offset_c, tl.int64)
        total_elements = tl.cast(total_elements_c, tl.int64)
    else:
        in_ptr = in_ptr_d
        dim_size_in = tl.cast(dim_size_in_d, tl.int64)
        dim_offset = tl.cast(dim_offset_d, tl.int64)
        total_elements = tl.cast(total_elements_d, tl.int64)

    block_start = tl.cast(pid_x, tl.int64) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    mask = idx < total_elements

    dim_size_out_64 = tl.cast(dim_size_out, tl.int64)
    dim_prod_post_64 = tl.cast(dim_prod_post, tl.int64)
    safe_dim_size_in = tl.where(dim_size_in > 0, dim_size_in, 1)
    safe_dim_prod_post = tl.where(dim_prod_post_64 > 0, dim_prod_post_64, 1)

    pre_idx = idx // (safe_dim_size_in * safe_dim_prod_post)
    dim_idx = (idx // safe_dim_prod_post) % safe_dim_size_in
    post_idx = idx % safe_dim_prod_post

    out_idx = (
        pre_idx * dim_size_out_64 * dim_prod_post_64
        + (dim_idx + dim_offset) * dim_prod_post_64
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def _cat_run_kernel(
    A: List[torch.Tensor],
    dim: int,
    out_shape: List[int],
    out: torch.Tensor,
):
    if out.numel() == 0:
        return

    # Filter out zero-element tensors (e.g. shape[dim] == 0) while tracking dim_offset
    non_empty = []
    running_offset = 0
    for t in A:
        dim_sz = t.shape[dim]
        if t.numel() > 0:
            non_empty.append((t, dim_sz, running_offset))
        running_offset += dim_sz
    if not non_empty:
        return

    BLOCK = 1024
    dim_size_out = out_shape[dim]
    dim_prod_post = 1
    for d in range(dim + 1, A[0].ndim):
        dim_prod_post *= A[0].shape[d]

    with torch_device_fn.device(out.device):
        i = 0
        while i < len(non_empty):
            batch = non_empty[i : i + 4]
            num_tensors_in_batch = len(batch)

            args = []
            total_elements_list = []

            for j in range(4):
                if j < num_tensors_in_batch:
                    raw_t, dim_size_in, cur_offset = batch[j]
                    tensor = raw_t.contiguous()
                    total_elements = tensor.numel()

                    args.extend([tensor, dim_size_in, cur_offset, total_elements])
                    total_elements_list.append(total_elements)
                else:
                    args.extend([args[0], 1, 0, 0])
                    total_elements_list.append(0)

            grid_y = num_tensors_in_batch
            max_elements_in_batch = max(total_elements_list) if total_elements_list else 0
            if max_elements_in_batch == 0:
                i += num_tensors_in_batch
                continue
            grid = (triton.cdiv(max_elements_in_batch, BLOCK), grid_y)

            (
                tensor_a,
                dim_size_in_a,
                dim_offset_a,
                total_elements_a,
                tensor_b,
                dim_size_in_b,
                dim_offset_b,
                total_elements_b,
                tensor_c,
                dim_size_in_c,
                dim_offset_c,
                total_elements_c,
                tensor_d,
                dim_size_in_d,
                dim_offset_d,
                total_elements_d,
            ) = args

            cat_copy_func_kernel_4[grid](
                out,
                tensor_a,
                tensor_b,
                tensor_c,
                tensor_d,
                dim_size_in_a,
                dim_size_in_b,
                dim_size_in_c,
                dim_size_in_d,
                dim_size_out,
                dim_prod_post,
                dim_offset_a,
                dim_offset_b,
                dim_offset_c,
                dim_offset_d,
                total_elements_a,
                total_elements_b,
                total_elements_c,
                total_elements_d,
                BLOCK_X=BLOCK,
            )

            i += num_tensors_in_batch


def _cat_build_working_list(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int
):
    """Returns (mode, payload) where mode is 'single'|'empty'|'multi'."""
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return "single", A[0]

    device = A[0].device
    dtype = A[0].dtype
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return "empty", torch.tensor([], device=device, dtype=dtype)
    if len(A) == 1:
        return "single", A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    dim %= A[0].ndim

    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    dtypes = [t.dtype for t in A]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)
    A = [t.to(dtype) if t.dtype != dtype else t for t in A]

    shapes = [t.shape for t in A]
    cat_dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(cat_dim_sizes)
    return "multi", (A, dim, out_shape, dtype, A[0].device)


def cat_out(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int = 0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS CAT_OUT")
    A = list(A)
    if _should_use_uint8_view_path(A):
        mode, payload, original_dtype = _cat_build_working_list_uint8_view(A, dim)
        if mode == "single":
            t = payload.view(original_dtype)
            out.resize_(t.shape)
            if out.dtype != t.dtype:
                out.copy_(t.to(out.dtype))
            else:
                out.copy_(t)
            return out
        if mode == "empty":
            t = payload.view(original_dtype)
            out.resize_(t.shape)
            out.copy_(t)
            return out

        A_u8, dim, out_shape, _, _ = payload
        if out.dtype != original_dtype:
            raise RuntimeError(
                f"cat.out: expected out dtype {original_dtype}, got {out.dtype}"
            )
        if list(out.shape) != out_shape:
            out.resize_(out_shape)
        out_u8 = out.view(torch.uint8)
        _cat_run_kernel(A_u8, dim, out_shape, out_u8)
        return out

    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        t = payload
        out.resize_(t.shape)
        if out.dtype != t.dtype:
            out.copy_(t.to(out.dtype))
        else:
            out.copy_(t)
        return out
    if mode == "empty":
        t = payload
        out.resize_(t.shape)
        out.copy_(t)
        return out

    A, dim, out_shape, dtype, device = payload
    if out.dtype != dtype:
        raise RuntimeError(f"cat.out: expected out dtype {dtype}, got {out.dtype}")
    if list(out.shape) != out_shape:
        out.resize_(out_shape)
    if out.is_contiguous():
        _cat_run_kernel(A, dim, out_shape, out)
    else:
        buf = torch.empty(out_shape, dtype=dtype, device=device)
        _cat_run_kernel(A, dim, out_shape, buf)
        out.copy_(buf)
    return out


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS CAT")
    A = list(A)
    if _should_use_uint8_view_path(A):
        mode, payload, original_dtype = _cat_build_working_list_uint8_view(A, dim)
        if mode == "single":
            return payload.view(original_dtype).clone()
        if mode == "empty":
            return payload.view(original_dtype)

        A_u8, dim, out_shape, _, device = payload
        out_u8 = torch.empty(out_shape, dtype=torch.uint8, device=device)
        _cat_run_kernel(A_u8, dim, out_shape, out_u8)
        return out_u8.view(original_dtype)

    mode, payload = _cat_build_working_list(A, dim)
    if mode == "single":
        return payload.clone()
    if mode == "empty":
        return payload

    A, dim, out_shape, dtype, device = payload
    out = torch.empty(out_shape, dtype=dtype, device=device)
    _cat_run_kernel(A, dim, out_shape, out)
    return out
