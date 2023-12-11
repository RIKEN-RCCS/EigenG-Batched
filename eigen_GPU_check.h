#ifndef __HEADER_EIGEN_GPU_CHECK_H__
#define __HEADER_EIGEN_GPU_CHECK_H__

#include <stdio.h>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"

#ifdef __cplusplus
extern "C" {
#endif

__host__ void
eigen_GPU_check_DP(const int L, const int nm, const int n, const int m, double *a_, double *w_, double *z_, const gpuStream_t stream);
__host__ void
eigen_GPU_check_FP(const int L, const int nm, const int n, const int m, float *a_, float *w_, float *z_, const gpuStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

