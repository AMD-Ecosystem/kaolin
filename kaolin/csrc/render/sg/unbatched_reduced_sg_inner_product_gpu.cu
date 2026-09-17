// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
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

// Spherical Gaussians fused Inner Product + sum on "others"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "../../utils.h"

namespace kaolin {

// Wave-size-agnostic width-pinned shuffle-down helper.
//
// These SG reductions use a fixed-width LOGICAL warp: the forward kernel launches
// dim3(32,32) and reduces across threadIdx.x (a 32-lane group), while the
// backward kernel launches dim3(16,16) and reduces across a 16-lane group
// (butterfly start offset 8). On NVIDIA a wavefront is exactly 32 lanes, so a
// plain __shfl_down_sync over the group is correct. On AMD CDNA a wavefront is
// 64 lanes, so a single wavefront spans TWO (forward) or FOUR (backward) of these
// logical groups; a full-wavefront shuffle would cross-contaminate independent
// SG rows. We therefore PIN the shuffle to the logical group width so each
// wavefront behaves as several independent width-`WIDTH` sub-groups. This keeps
// the existing launch geometry and the offset=16 (fwd) / offset=8 (bwd) butterfly
// starts correct on both wave32 and wave64.
//
// On AMD the maskless width-limited __shfl_down(val, offset, width) is used
// directly. On NVIDIA (ROCm 7 rejects 32-bit masks at compile time via
// HIP_ENABLE_WARP_SYNC_BUILTINS static_assert(sizeof(MaskT)==8); the original
// forward mask 0xfffffff was also a latent 28-bit truncated-participation bug even
// on CUDA) we use the native __shfl_down_sync with a full 32-bit mask and an
// explicit width so the participation domain is exactly the logical group.
// KB: shfl-down-reduction-mask-truncated-literal, hip-shfl-sync-64bit-mask-rocm7,
//     spmm-sddmm-literal32-warp-torch-ext-width-pin.
__device__ __forceinline__ float sg_shfl_down_width(float val, int offset, int width) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_down(val, offset, width);
#else
  return __shfl_down_sync(0xffffffff, val, offset, width);
#endif
}

// TODO(cfujitsang): There is a faster implementation if num_sg is very big or num_other is very small
//                   We should have both with kernel selection
__global__
void unbatched_reduced_sg_inner_product_forward_cuda_kernel(
    const float* __restrict__ intensity,
    const float* __restrict__ direction,
    const float* __restrict__ sharpness,
    const float* __restrict__ other_intensity,
    const float* __restrict__ other_direction,
    const float* __restrict__ other_sharpness,
    const int num_sg,
    const int num_other,
    float* output) {
  __shared__ float shm[32][3];
  for (int start_sg_idx = blockDim.x * blockIdx.x;
       start_sg_idx < num_sg;
       start_sg_idx += gridDim.x * blockDim.x) {
    int sg_idx = start_sg_idx + threadIdx.y;
    // Load the directions
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = direction[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    // Load the sharpness
    float sharp = sharpness[sg_idx];
    // Load the intensity
    __syncthreads();
    float val_x = shm[threadIdx.y][0] * sharp;
    float val_y = shm[threadIdx.y][1] * sharp;
    float val_z = shm[threadIdx.y][2] * sharp;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = intensity[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    float intensity_x = shm[threadIdx.y][0];
    float intensity_y = shm[threadIdx.y][1];
    float intensity_z = shm[threadIdx.y][2];
    float sum_x = 0.;
    float sum_y = 0.;
    float sum_z = 0.;
    if (start_sg_idx < num_sg) {
      for (int start_other_idx = 0; start_other_idx < num_other; start_other_idx += blockDim.x) {
	// load and share "others" sg parameters to do the broadcast operations
	// While the block is 32x32 we are only loading on the first 32 threads
	// The broadcast operation is then done with all the threads
        __syncthreads();
        int other_idx = start_other_idx + threadIdx.x;
        float other_sharp = other_sharpness[other_idx];
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_direction[start_other_idx * 3 + inner_idx];
            }
          }
        }
        __syncthreads();
        float other_val_x = shm[threadIdx.x][0] * other_sharp;
        float other_val_y = shm[threadIdx.x][1] * other_sharp;
        float other_val_z = shm[threadIdx.x][2] * other_sharp;
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_intensity[start_other_idx * 3 + inner_idx];
            }
          }
        }
        float tmp_x = val_x + other_val_x;
        float tmp_y = val_y + other_val_y;
        float tmp_z = val_z + other_val_z;
        float um = sqrtf(fmaf(tmp_x, tmp_x, fmaf(tmp_y, tmp_y, tmp_z * tmp_z)));
        float lm = sharp + other_sharp;
        __syncthreads();
        float intensity_prod_x = shm[threadIdx.x][0] * intensity_x;
        float intensity_prod_y = shm[threadIdx.x][1] * intensity_y;
        float intensity_prod_z = shm[threadIdx.x][2] * intensity_z;
        // Fast HW exp intrinsic + hoisted 2*pi + single reciprocal of um.
        const float two_pi = 2.f * (float)M_PI;
        float inv_um = 1.f / um;
        float mul_coeff = _EXP(um - lm) * two_pi * (1.f - _EXP(-2.f * um)) * inv_um;
        if (other_idx < num_other) {
          sum_x += intensity_prod_x * mul_coeff;
          sum_y += intensity_prod_y * mul_coeff;
          sum_z += intensity_prod_z * mul_coeff;
        }
      }
      // Sum reduction across threads of same threadIdx.y (32-lane logical warp;
      // width-pinned to 32 so a 64-lane wavefront acts as two independent groups).
      for (int offset = 16; offset > 0; offset /= 2) {
        sum_x += sg_shfl_down_width(sum_x, offset, 32);
        sum_y += sg_shfl_down_width(sum_y, offset, 32);
        sum_z += sg_shfl_down_width(sum_z, offset, 32);
      }
      // Use shared memory for coalescent store
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_x;
        shm[threadIdx.y][1] = sum_y;
        shm[threadIdx.y][2] = sum_z;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            output[start_sg_idx * 3 + inner_idx] = shm[inner_idx / 3][inner_idx % 3];
          }
        }
      }
    }
  }
}

__global__
void unbatched_reduced_sg_inner_product_backward_cuda_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ intensity,
    const float* __restrict__ direction,
    const float* __restrict__ sharpness,
    const float* __restrict__ other_intensity,
    const float* __restrict__ other_direction,
    const float* __restrict__ other_sharpness,
    const int num_sg,
    const int num_other,
    float* __restrict__ grad_intensity,
    float* __restrict__ grad_direction,
    float* __restrict__ grad_sharpness,
    float* __restrict__ grad_other_intensity,
    float* __restrict__ grad_other_direction,
    float* __restrict__ grad_other_sharpness) {
  // TODO(cfujitsang): need to add a lot of comments
  volatile __shared__ float shm[16][17];
  volatile float twopi = 2. * M_PI;
  volatile float zero = 0.;
 
  for (int start_sg_idx = blockDim.x * blockIdx.x;
       start_sg_idx < num_sg;
       start_sg_idx += gridDim.x * blockDim.x) {
    int sg_idx = start_sg_idx + threadIdx.y;
    bool is_active_sg = sg_idx < num_sg;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        const int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = direction[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    float sharp = sharpness[sg_idx];
    __syncthreads();
    float direction_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float direction_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float direction_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = intensity[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    // need to put zero because of some NaN bug on ampere
    float intensity_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float intensity_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float intensity_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = grad_out[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    float grad_re_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float grad_re_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float grad_re_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    float sum_grad_intensity_x = zero;
    float sum_grad_intensity_y = zero;
    float sum_grad_intensity_z = zero;
    float sum_grad_prod1_x = zero;
    float sum_grad_prod1_y = zero;
    float sum_grad_prod1_z = zero;
    float sum_grad_lm = zero;
    if (start_sg_idx < num_sg) {
      // Partition the num_other reduction across the blockIdx.y grid axis so the
      // launched CTA count rises toward/above the 304 CUs. Each y-split walks a
      // strided subset of the num_other tiles (tile width == blockDim.x); the
      // per-'other' grads atomicAdd into the final grad_other_* tensors (a given
      // other_idx is visited by exactly one y-split -> the only concurrency is
      // across blockIdx.x, exactly what the shared-buffer atomicAdd handles), and
      // the per-sg grads become partial sums that atomicAdd into the
      // zero-initialized grad_* outputs.
      for (int start_other_idx = blockIdx.y * blockDim.x; start_other_idx < num_other;
           start_other_idx += gridDim.y * blockDim.x) {
        __syncthreads();
        int other_idx = start_other_idx + threadIdx.x;
        float other_sharp = other_sharpness[other_idx];
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_direction[start_other_idx * 3 + inner_idx];
            }
          }
        }
        __syncthreads();
        float other_direction_x = shm[threadIdx.x][0];
        float other_direction_y = shm[threadIdx.x][1];
        float other_direction_z = shm[threadIdx.x][2];
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_intensity[start_other_idx * 3 + inner_idx];
            }
          }
        }

        float um_x = fmaf(direction_x, sharp, other_direction_x * other_sharp);
        float um_y = fmaf(direction_y, sharp, other_direction_y * other_sharp);
        float um_z = fmaf(direction_z, sharp, other_direction_z * other_sharp);
        float um_length = sqrtf(fmaf(um_x, um_x, fmaf(um_y, um_y, um_z * um_z)));
        float inv_um_length = 1.f / um_length;
        float lm = sharp + other_sharp;
        float exp_val = _EXP(um_length - lm);
        float other_exp = _EXP(-2.f * um_length);
        float other = 1. - other_exp;
        float exp_ratio = twopi * exp_val * other * inv_um_length;
        __syncthreads();
        float intensity_prod_x = is_active_sg ? shm[threadIdx.x][0] * intensity_x : zero;
        float intensity_prod_y = is_active_sg ? shm[threadIdx.x][1] * intensity_y : zero;
        float intensity_prod_z = is_active_sg ? shm[threadIdx.x][2] * intensity_z : zero;
        volatile float grad_intensity_prod_x = is_active_sg ? grad_re_x * exp_ratio : zero;
        volatile float grad_intensity_prod_y = is_active_sg ? grad_re_y * exp_ratio : zero;
        volatile float grad_intensity_prod_z = is_active_sg ? grad_re_z * exp_ratio : zero;
        float grad_exp_ratio = \
            grad_re_x * intensity_prod_x + \
            grad_re_y * intensity_prod_y + \
            grad_re_z * intensity_prod_z;
        float grad_pre_mul_exp_mul = twopi * grad_exp_ratio * inv_um_length;
        float grad_exp_val = grad_pre_mul_exp_mul * other;
        float grad_other = grad_pre_mul_exp_mul * exp_val;
        float grad_subtract = is_active_sg ? grad_exp_val * exp_val : zero;
        float grad_um_length = grad_subtract + grad_other * 2. * other_exp - \
                               grad_exp_ratio * exp_ratio * inv_um_length;
        float scaled_grad_um_length = grad_um_length * inv_um_length;
        float grad_um_x = is_active_sg ? scaled_grad_um_length * um_x : zero;
        float grad_um_y = is_active_sg ? scaled_grad_um_length * um_y : zero;
        float grad_um_z = is_active_sg ? scaled_grad_um_length * um_z : zero;

        if (other_idx < num_other) {
          sum_grad_intensity_x += grad_intensity_prod_x * shm[threadIdx.x][0];
          sum_grad_intensity_y += grad_intensity_prod_y * shm[threadIdx.x][1];
          sum_grad_intensity_z += grad_intensity_prod_z * shm[threadIdx.x][2];
          sum_grad_prod1_x += grad_um_x;
          sum_grad_prod1_y += grad_um_y;
          sum_grad_prod1_z += grad_um_z;
          sum_grad_lm += grad_subtract;
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_x * intensity_x;
        __syncthreads();
        grad_intensity_prod_x = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_y * intensity_y;
        __syncthreads();
        grad_intensity_prod_y = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_z * intensity_z;
        __syncthreads();
        grad_intensity_prod_z = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        // 16-lane logical group (backward launch is dim3(16,16), offset start 8);
        // width-pin to 16 so wave64 acts as four independent groups.
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_intensity_prod_x += sg_shfl_down_width(grad_intensity_prod_x, offset, 16);
          grad_intensity_prod_y += sg_shfl_down_width(grad_intensity_prod_y, offset, 16);
          grad_intensity_prod_z += sg_shfl_down_width(grad_intensity_prod_z, offset, 16);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = grad_intensity_prod_x;
          shm[threadIdx.y][1] = grad_intensity_prod_y;
          shm[threadIdx.y][2] = grad_intensity_prod_z;
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int ii = 0; ii < 3; ii++) {
            int inner_idx = blockDim.x * ii + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              atomicAdd(&grad_other_intensity[start_other_idx * 3 + inner_idx],
                        shm[inner_idx / 3][inner_idx % 3]);
            }
          }
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_x;
        __syncthreads();
        grad_um_x = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_y;
        __syncthreads();
        grad_um_y = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_z;
        __syncthreads();
        grad_um_z = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        // 16-lane logical group (width-pinned to 16 for wave64 correctness).
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_um_x += sg_shfl_down_width(grad_um_x, offset, 16);
          grad_um_y += sg_shfl_down_width(grad_um_y, offset, 16);
          grad_um_z += sg_shfl_down_width(grad_um_z, offset, 16);
        }
        __syncthreads();
        if (threadIdx.y == 0) {
          shm[threadIdx.x][3] = other_sharp;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = grad_um_x * shm[threadIdx.y][3];
          shm[threadIdx.y][1] = grad_um_y * shm[threadIdx.y][3];
          shm[threadIdx.y][2] = grad_um_z * shm[threadIdx.y][3];
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int ii = 0; ii < 3; ii++) {
            int inner_idx = blockDim.x * ii + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              atomicAdd(&grad_other_direction[start_other_idx * 3 + inner_idx],
                        shm[inner_idx / 3][inner_idx % 3]);
            }
          }
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_subtract;
        __syncthreads();
        grad_subtract = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        // 16-lane logical group (width-pinned to 16 for wave64 correctness).
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_subtract += sg_shfl_down_width(grad_subtract, offset, 16);
        }
        if (threadIdx.y == 0) {
          shm[threadIdx.x][1] = other_direction_x;
          shm[threadIdx.x][2] = other_direction_y;
          shm[threadIdx.x][3] = other_direction_z;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = \
              grad_um_x * shm[threadIdx.y][1] + \
              grad_um_y * shm[threadIdx.y][2] + \
              grad_um_z * shm[threadIdx.y][3] - \
              grad_subtract;
        }
        __syncthreads();
        if (threadIdx.y == 0) {
          if (other_idx < num_other) {
            atomicAdd(&grad_other_sharpness[start_other_idx + threadIdx.x],
                      shm[threadIdx.x][0]);
          }
        }
      }
      __syncthreads();
      // 16-lane logical group (width-pinned to 16 for wave64 correctness).
      for (int offset = 8; offset > 0; offset /= 2) {
        sum_grad_intensity_x += sg_shfl_down_width(sum_grad_intensity_x, offset, 16);
        sum_grad_intensity_y += sg_shfl_down_width(sum_grad_intensity_y, offset, 16);
        sum_grad_intensity_z += sg_shfl_down_width(sum_grad_intensity_z, offset, 16);
        sum_grad_prod1_x += sg_shfl_down_width(sum_grad_prod1_x, offset, 16);
        sum_grad_prod1_y += sg_shfl_down_width(sum_grad_prod1_y, offset, 16);
        sum_grad_prod1_z += sg_shfl_down_width(sum_grad_prod1_z, offset, 16);
        sum_grad_lm += sg_shfl_down_width(sum_grad_lm, offset, 16);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_grad_intensity_x;
        shm[threadIdx.y][1] = sum_grad_intensity_y;
        shm[threadIdx.y][2] = sum_grad_intensity_z;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            atomicAdd(&grad_intensity[start_sg_idx * 3 + inner_idx],
                      shm[inner_idx / 3][inner_idx % 3]);
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_grad_prod1_x * sharp;
        shm[threadIdx.y][1] = sum_grad_prod1_y * sharp;
        shm[threadIdx.y][2] = sum_grad_prod1_z * sharp;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            atomicAdd(&grad_direction[start_sg_idx * 3 + inner_idx],
                      shm[inner_idx / 3][inner_idx % 3]);
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = \
          sum_grad_prod1_x * direction_x + \
          sum_grad_prod1_y * direction_y + \
          sum_grad_prod1_z * direction_z - sum_grad_lm;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
        int inner_idx = start_sg_idx + threadIdx.x;
        if (inner_idx < num_sg) {
          atomicAdd(&grad_sharpness[inner_idx], shm[threadIdx.x][0]);
        }
      }
    }
  }
}

void unbatched_reduced_sg_inner_product_forward_cuda_impl(
    const at::Tensor intensity,
    const at::Tensor direction,
    const at::Tensor sharpness,
    const at::Tensor other_intensity,
    const at::Tensor other_direction,
    const at::Tensor other_sharpness,
    at::Tensor output) {
  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);

  const dim3 threads(32, 32, 1);
  const int blocks = (num_sg + 32 - 1) / 32;

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(intensity));
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO(cfujitsang): extend support to double / fp16
  unbatched_reduced_sg_inner_product_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      intensity.data_ptr<float>(),
      direction.data_ptr<float>(),
      sharpness.data_ptr<float>(),
      other_intensity.data_ptr<float>(),
      other_direction.data_ptr<float>(),
      other_sharpness.data_ptr<float>(),
      num_sg,
      num_other,
      output.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
}

void unbatched_reduced_sg_inner_product_backward_cuda_impl(
    at::Tensor grad_out,
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness,
    at::Tensor grad_intensity,
    at::Tensor grad_direction,
    at::Tensor grad_sharpness,
    at::Tensor grad_other_intensity,
    at::Tensor grad_other_direction,
    at::Tensor grad_other_sharpness) {
  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);
  int blocks = (num_sg + 16 - 1) / 16;
  if (blocks > 128)
    blocks = 128;

  // The kernel accumulates the per-other grads with atomicAdd directly into the
  // final [num_other,*] output tensors (pre-zeroed by the host wrapper), so the
  // old [blocks,num_other,*] scratch caches and the 3 at::sum_out block-axis
  // reductions are gone: 3 fewer FillFunctor allocations + 3 fewer reduce_kernel
  // dispatches per backward call, with identical results (summing over blocks is
  // exactly what the concurrent atomicAdds into one buffer do).
  //
  // The 1D grid (blocks == ceil(num_sg/16), <=16 for these shapes) leaves the
  // 304-CU gfx942 <6% occupied while the O(num_sg x num_other) reduction
  // serializes the whole num_other loop inside each block. Fill the device by
  // partitioning num_other across a second grid axis: fully split down to one
  // 16-wide tile per block (other_splits == other_tiles). Each (x,y) block
  // reduces one strided num_other tile; per-other grads atomicAdd into the final
  // tensors (each other_idx touched by exactly one y-split) and per-sg grads
  // atomicAdd their y-partial into the zero-initialized grad_* outputs.
  const int other_tiles = (num_other + 16 - 1) / 16;
  int other_splits = other_tiles;
  if (other_splits < 1) other_splits = 1;
  const dim3 threads(16, 16, 1);
  const dim3 grid(blocks, other_splits, 1);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_out));
  auto stream = at::cuda::getCurrentCUDAStream();

  unbatched_reduced_sg_inner_product_backward_cuda_kernel<<<grid, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      intensity.data_ptr<float>(),
      direction.data_ptr<float>(),
      sharpness.data_ptr<float>(),
      other_intensity.data_ptr<float>(),
      other_direction.data_ptr<float>(),
      other_sharpness.data_ptr<float>(),
      num_sg,
      num_other,
      grad_intensity.data_ptr<float>(),
      grad_direction.data_ptr<float>(),
      grad_sharpness.data_ptr<float>(),
      grad_other_intensity.data_ptr<float>(),
      grad_other_direction.data_ptr<float>(),
      grad_other_sharpness.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin

