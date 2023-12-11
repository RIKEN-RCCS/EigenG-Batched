#ifndef __HEADER_EIGEN_GPU_CHECK_HPP__
#define __HEADER_EIGEN_GPU_CHECK_HPP__

#include <stdio.h>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"
#include "eigen_GPU_check.h"


template <class T>
__host__ void
eigen_GPU_check(const int L, const int nm, const int n, const int m, T *a_, T *w_, T *z_, const gpuStream_t stream = NULL)
{
//  if (std::is_same<T,half>::value) {
//    eigen_GPU_check_HP(L, nm, n, m, (half*)a_, (half*)w_, (half*)z_);
//  }
  if (std::is_same<T,float>::value) {
    eigen_GPU_check_FP(L, nm, n, m, (float*)a_, (float*)w_, (float*)z_, stream);
  }
  if (std::is_same<T,double>::value) {
    eigen_GPU_check_DP(L, nm, n, m, (double*)a_, (double*)w_, (double*)z_, stream);
  }
}

#endif

