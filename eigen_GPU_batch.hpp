#ifndef __HEADER_EIGEN_GPU_BATCH_HPP__
#define __HEADER_EIGEN_GPU_BATCH_HPP__

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPACC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"
#include "eigen_GPU_batch.h"



template <class T>
__host__ gpuError_t
eigen_GPU_batch_get_Launch_params(const int L, const int n, int &grpSZ_, int &numTB_, int &numGR_, int &numTH_, int &sizeSH_, int &sizeL2_)
{
  if (L<1||n<1) {
    grpSZ_ = numTB_ = numGR_ = numTH_ = sizeSH_ = sizeL2_ = 0;
    return gpuErrorInvalidValue;
  }
  gpuError_t err_code;
  int dev;
  err_code = gpuGetDevice( &dev );
  if ( err_code != gpuSuccess ) {
    grpSZ_ = numTB_ = numGR_ = numTH_ = sizeSH_ = sizeL2_ = 0;
    return err_code;
  }
  gpuDeviceProp deviceProp;
  err_code = gpuGetDeviceProperties (&deviceProp, dev);
  if ( err_code != gpuSuccess ) {
    grpSZ_ = numTB_ = numGR_ = numTH_ = sizeSH_ = sizeL2_ = 0;
    return err_code;
  }

  // The number of Thread per ThreadBlock
  const int numTH = WARP_GPU_SIZE*8;

  // groupSize = tile_size
  const int grpSZ = (n<=4) ?
         4  : ((n<=8)  ?
         8  : ((n<=16) ?
         16 : ((n<=32) ?
         32 : WARP_GPU_SIZE )));

  const int min_Batch_min = (n<=32) ?
	 16 : ((n<=64) ?
	 8 : ((n<=96) ?
	 2 : 1 ));
  const int LX = (L-1)/min_Batch_min+1;

  const int num_Goups_per_Warp = WARP_GPU_SIZE/grpSZ;
  // Warps to be activated = ...
  const int numWP = (LX-1)/num_Goups_per_Warp+1;
  // TBs required evenly invoked = numWP/(total_threads/Warpsize)
  const int num_Warps_per_TB = numTH/WARP_GPU_SIZE;
  const int reqTB = (numWP-1)/num_Warps_per_TB+1;

  // The number of ThreadBlocks to be invoked per SM
  const int maxTB = deviceProp.multiProcessorCount*8;
  const int numTB = min(reqTB, maxTB);

  // The number of groups per ThreadBlock
  const int numGR = numTH/grpSZ;

  // The required shared memory size per ThreadBlock
  const int sizeSH = sizeof(T)*numTH*2;

  // These parameter controlls assignment of threads and batches, statically.
  // Therefore, if numGR*numTB >> L, some CUDA cores are idling, and
  // dynamic schedulling will be effective in such a case.

  const int sizeL2 = deviceProp.l2CacheSize;

  grpSZ_ = grpSZ;
  numTB_ = numTB;
  numGR_ = numGR;
  numTH_ = numTH;
  sizeSH_ = sizeSH;
  sizeL2_ = sizeL2;

#if 0
  std::cout << "Group Size           = " << grpSZ_ << "\n";
  std::cout << "num of ThreadBlocks  = " << numTB_ << "\n";
  std::cout << "num of Group Teams   = " << numGR_ << "\n";
  std::cout << "num of Total Threads = " << numTH_ << "\n";
  std::cout << "Warp size            = " << WARP_GPU_SIZE << "\n";
  std::cout << "Shared memory size   = " << sizeSH_ << "\n";
  std::cout << "L2 Cache size        = " << sizeL2_ << "\n";
#endif

  return gpuSuccess;
}

template <class T>
__host__ gpuError_t
eigen_GPU_batch_BufferSize(const int L, const int nm, const int n, const int m, T * a, T * w, size_t *lwork)
{
  *lwork = 0;
  if (L<1||nm<n||n<1||m<1||m>n) return gpuErrorInvalidValue;
  int grpSZ=0, numTB=0, numGR=0, numTH=0, sizeSH=0, sizeL2=0;
  gpuError_t err_code = eigen_GPU_batch_get_Launch_params<T>(L, n, grpSZ, numTB, numGR, numTH, sizeSH, sizeL2);
  if ( err_code != gpuSuccess ) return err_code;

  size_t elem = sizeof(T)*(n+(n*nm));
  size_t len  = elem*min(numTB*numGR,L);
  *lwork = len;
  return gpuSuccess;
}

template <class T>
__host__ void
eigen_GPU_batch(const int L, const int nm, const int n, const int m, T * a, T * w, T * wk, const gpuStream_t stream=NULL)
{
#if 0
  int current_device;
  gpuGetDevice(&current_device);

  {
    gpuPointerAttributes attr_a, attr_w, attr_wk;

    gpuPointerGetAttributes(&attr_a, a);
    gpuPointerGetAttributes(&attr_w, w);
    gpuPointerGetAttributes(&attr_wk, wk);

    if ( attr_a.device != attr_w.device ||
         attr_w.device != attr_wk.device ||
         attr_wk.device != attr_a.device ) {
      fprintf(stderr,"device number confliction\n"); fflush(stderr);
      exit(1);
    }
    gpuSetDevice( attr_a.device );
  }
#endif

//  if (std::is_same<T,half>::value) {
//    ;
//  }
  if (std::is_same<T,float>::value) {
    eigen_GPU_batch_FP(L, nm, n, m, (float*)a, (float*)w, (float*)wk, stream);
  }
  if (std::is_same<T,double>::value) {
    eigen_GPU_batch_DP(L, nm, n, m, (double*)a, (double*)w, (double*)wk, stream);
  }
#if 0
  gpuSetDevice( current_device );
#endif
}

#endif

