// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// hipCUB (over rocPRIM) does not provide CUB's debug macro CubDebugExit, which
// kaolin uses to wrap cub::DeviceScan / cub::DeviceRadixSort dispatch and a few
// cudaMemcpy calls. CUB's CubDebugExit(e) evaluates e (which returns a
// cudaError_t) and aborts on a non-success status. Provide a faithful HIP shim so
// the SPC scan/sort call sites compile and keep their error checking on ROCm.
// Guarded on __HIP_PLATFORM_AMD__ / USE_ROCM so the CUDA build keeps CUB's own
// macro. KB: cub-hipcub, hipcub-namespace-alias-cub.

#ifndef KAOLIN_ROCM_CUB_COMPAT_H_
#define KAOLIN_ROCM_CUB_COMPAT_H_

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CubDebugExit
namespace kaolin {
static inline hipError_t _kaolin_cub_debug_exit(hipError_t status,
                                                const char* file, int line) {
  if (status != hipSuccess) {
    fprintf(stderr, "CUDA/HIP error %d [%s] at %s:%d\n",
            static_cast<int>(status), hipGetErrorString(status), file, line);
    fflush(stderr);
    abort();
  }
  return status;
}
}  // namespace kaolin
// Cast through hipError_t so a plain cudaError_t/int-returning expression
// (hipify maps cudaError_t -> hipError_t) type-checks.
#define CubDebugExit(e) \
  ::kaolin::_kaolin_cub_debug_exit(static_cast<hipError_t>(e), __FILE__, __LINE__)
#endif  // CubDebugExit

#endif  // __HIP_PLATFORM_AMD__ || USE_ROCM

#endif  // KAOLIN_ROCM_CUB_COMPAT_H_
