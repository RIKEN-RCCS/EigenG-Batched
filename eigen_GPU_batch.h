#ifndef __HEADER_EIGEN_GPU_BATCH_H__
#define __HEADER_EIGEN_GPU_BATCH_H__

#include <stdio.h>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_arch.h"

//
// if wk == NULL && n > 1 then ((size_t)w)[0] returns the length of
// work buffer required in the batch operation with an accordance to
// the matrix size and GPU architecture.
//

void
eigen_GPU_batch_DP(const int L, const int nm, const int n, const int m, double * a, double * w, double *wk, const gpuStream_t stream);

void
eigen_GPU_batch_FP(const int L, const int nm, const int n, const int m, float * a, float * w, float *wk, const gpuStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

