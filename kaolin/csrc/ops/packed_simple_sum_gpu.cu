// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"
#include "../packed_base.cuh"

// currently based on sizeof(long), TODO(cfujitsang): make it variable to sizeof(idx_t)
#define NB_CHUNK_LIMIT 448
#define CHUNK_SIZE 32768
#define ILP 7
#define BLOCK_SIZE 1024

namespace kaolin {

/*
 * Make sum reduction within a block using shared memory and warp intrinsic.
 *
 * This is a genuine block-wide reduction (one output per chunk), so it must cover
 * ALL lanes of the final wavefront. The original code hardcoded a 32-lane warp
 * (shared-memory tree down to i>=64, fold x[tid]+x[tid+32] over tid<32, butterfly
 * start 16, 32-bit mask). On AMD CDNA the wavefront is 64 lanes, so those literals
 * silently HALVE the reduction (lanes 32..63 dropped). We parameterize every
 * warp-size literal on warpSize (=64 on wave64, 32 on wave32) so the final-warp
 * fold and butterfly span the whole wavefront on both platforms, and use a
 * wave-agnostic 64-bit mask (ROCm 7 rejects 32-bit masks at compile time).
 * KB: ft-blockreduce-lane-wid-mask-wave64-warpcount-bug,
 *     hardcoded-warpsize-literal-to-warpsize, hip-shfl-sync-64bit-mask-rocm7.
 */
template<typename DType>
__device__ __forceinline__ DType ReduceBlockIntoLanes(DType* x,
                                                      DType val) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  const int wsize = warpSize;

  if (block_size >= 2 * wsize) {
    x[tid] = val;
    __syncthreads();
  }

  for (int i = (block_size >> 1); i >= 2 * wsize; i >>= 1) {
    if (tid < i)
      x[tid] = x[tid] + x[tid+i];
    __syncthreads();
  }

  DType final;
  if (tid < wsize) {
    if (block_size >= 2 * wsize)
      final = x[tid] + x[tid+wsize];
    else
      final = val;

    for (int i = wsize >> 1; i >= 1; i >>= 1)
#if defined(__HIP_PLATFORM_AMD__)
      final = final + __shfl_down(final, i, wsize);
#else
      final = final + __shfl_down_sync(0xffffffff, final, i, wsize);
#endif
  }
  return final;
}

template<typename scalar_t, typename out_scalar_t, typename idx_t>
__global__ void packed_simple_sum_cuda_kernel(
    const scalar_t* __restrict__ packed_tensor,
    PackedSimpleKernelMetadata<unsigned char, NB_CHUNK_LIMIT, 256, 1, 0, false, false> param,
    out_scalar_t* output) {
  const idx_t chunk_start_idx = param.chunk_to_idx[blockIdx.x];
  const idx_t chunk_end_idx = param.chunk_to_idx[blockIdx.x + 1];
  const idx_t tensor_idx = param.chunk_to_tensor_idx[blockIdx.x];

  __shared__ out_scalar_t vals[BLOCK_SIZE];
  out_scalar_t val = 0;
  scalar_t incoming_vals[ILP];
  idx_t i_start = chunk_start_idx;
  idx_t i_end = chunk_start_idx + blockDim.x * ILP;
  // only do ILP if fully possible for the maximum number of iterations
  // TODO(cfujitsang): maybe not the most efficient with small arrays...
  for (;i_end < chunk_end_idx; i_end += blockDim.x * ILP) {
    idx_t i = i_start + threadIdx.x;
#pragma unroll
    for (int ii = 0; ii < ILP; ii++) {
      incoming_vals[ii] = packed_tensor[i + ii * blockDim.x];
    }

#pragma unroll
    for (int ii = 0; ii < ILP; ii++) {
      val += static_cast<out_scalar_t>(incoming_vals[ii]);
    }
    i_start = i_end;
  }

  for (idx_t i = i_start + threadIdx.x; i < chunk_end_idx; i += blockDim.x) {
    val += static_cast<out_scalar_t>(packed_tensor[i]);
  }

  const out_scalar_t final = ReduceBlockIntoLanes(vals, val);

  if (threadIdx.x == 0) {
    atomicAdd(output + tensor_idx + param.tensor_offset, final);
  }
}

#define DISPATCH_INOUT_DEDUCED_TYPES(TYPE, IN_TYPE_NAME, OUT_TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Bool, bool, int64_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Int, int32_t, int64_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Long, int64_t, int64_t, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Half, at::Half, float, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Float, float, float, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      PRIVATE_CASE_INOUT_DEDUCED_TYPES(at::ScalarType::Double, double, double, IN_TYPE_NAME, OUT_TYPE_NAME, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for output as '", toString(TYPE), "'"); \
    } \
  }()

/*
 * CUDA function for packed tensor sum over subtensor of last_dim = 1
 */
at::Tensor packed_simple_sum_cuda_kernel_launcher(
    at::Tensor packed_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output) {
  const unsigned int batch_size = shape_per_tensor.size(0);
  assert(shape_per_tensor.scalar_type() == at::ScalarType::Long);
  DISPATCH_INOUT_DEDUCED_TYPES(packed_tensor.scalar_type(), scalar_t, out_scalar_t, "packed_simple_sum", [&] {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output_ptr = output.data_ptr<out_scalar_t>();
    auto input_ptr = packed_tensor.data_ptr<scalar_t>();
    cudaMemsetAsync(output_ptr, 0, batch_size * sizeof(out_scalar_t), stream);
    packed_simple_cuda_launcher<BLOCK_SIZE, NB_CHUNK_LIMIT, 256, 1, false, false, unsigned char, 0>(
        input_ptr,
        shape_per_tensor.data_ptr<int64_t>(),
        batch_size,
        packed_simple_sum_cuda_kernel<scalar_t, out_scalar_t, unsigned int>,
        CHUNK_SIZE,
        std::array<int64_t*, 0>(),
        output_ptr);
  });
  return output;
}

#undef DISPATCH_INOUT_DEDUCE_TYPES

}  // namespace kaolin
