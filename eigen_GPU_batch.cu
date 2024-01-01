#define	TIMER		0
#if defined(__NVCC__)
#define	USE_IMTQL	0
#define DO_PREFETCH	1
#endif
#if defined(__HIPCC__)
#define	USE_IMTQL	0
#define DO_PREFETCH	0
#endif
#include "gpu_arch.h"


//--
#include "misc_gpu.hpp"
#include "eigen_GPU_batch.hpp"
//--
#include "hhsy2tr.hpp"
#include "hhsy2tr_tiled.hpp"
#include "tred1_tiled.hpp"
//--
#include "imtql2.hpp"
#include "imtql2_tiled.hpp"
#include "tql2.hpp"
#include "tql2_tiled.hpp"
//--
#include "hhtr2sy.hpp"
#include "hhtr2sy_tiled.hpp"
//--
#include "jac_tiled.hpp"
//--



template <class T>
__global__ void
eigen_GPU_batch_n1_(const int L, const int nm, const int n, T *a_, T *w_)
{
  const int pos = threadIdx.x+blockIdx.x*blockDim.x;
  if (pos>=L) return;

  const size_t len = nm*n;
  T * a = a_ + pos*len;
  T * w = w_ + pos;

  const int step = blockDim.x*gridDim.x;
  #pragma unroll
  for (int id=pos; id<L; id+=step) {
    *w = *a;
    *a = static_cast<T>(1.0);
    a += len*step; w += n*step;
  }
}

template <class T, int tile_size>
__global__ void
eigen_GPU_batch_tiled_(const int L, const int nm, const int n, const int m, T *a_, T *w_, T *wk_, const int PRELOAD_SLOT)
{
  const int pos = (threadIdx.x+blockIdx.x*blockDim.x)/tile_size;
//  const int myid = threadIdx.x % tile_size;
//  if (pos>=L) return;

  const size_t len = (long)nm*n;
  const int step = (blockDim.x*gridDim.x)/tile_size;
  T * a, * w, * wk;

#if DO_PREFETCH
  a = a_ + pos*len;
  #pragma unroll 1
  for (int id_=0; id_<PRELOAD_SLOT; id_+=step) {
    int id = id_ + pos;
    if ( id < L )
    prefetch_mat_cg<T, tile_size> (nm*n, a);
    a += len*step;
  }
#endif

  a  = a_  + pos*len;
  w  = w_  + pos*m;
  wk = wk_ + pos*len;
  #pragma unroll 1
  for (int id_=0; id_<L; id_+=step) {
    const int id = id_ + pos;
    const bool run = (id < L);

    SYNC_IF_NEEDED();
#if DO_PREFETCH
    if ( PRELOAD_SLOT > 0 && id+PRELOAD_SLOT*step < L )
    prefetch_mat_cg<T, tile_size> (nm*n, a+PRELOAD_SLOT*len*step);
#endif

    SYNC_IF_NEEDED();
#if 1
#if 1
    if(run) tred1_tiled_<T, tile_size> (nm, n, a);
#else
    const int mb = tile_size<=16?2:3; // 2*(mb+1) <= n
    const int lenu = nm*mb+(4+(nm*mb)%2);
    if (2*lenu<=n*nm) {
    T * u_=wk+0*lenu; // mb*nm+max(BLK_J,BLK_K)
    T * v_=wk+1*lenu; // mb*nm+max(BLK_J,BLK_K)
    if(run) hhsy2tr_tiled_<T, tile_size> (nm, n, a, mb, u_, v_);
    } else {
    if(run) tred1_tiled_<T, tile_size> (nm, n, a);
    }
#endif

    int return_flag = false;
    SYNC_IF_NEEDED();
#if USE_IMTQL
    if(run) return_flag = imtql2_tiled_<T, tile_size> (nm, n, w, wk);
#else
    if(run) return_flag = tql2_tiled_<T, tile_size> (nm, n, w, wk);
#endif

    SYNC_IF_NEEDED();
    if(run) if ( !return_flag ) hhtr2sy_tiled_<T, tile_size> (nm, n, a, wk);
#else
    if(run) jac_tiled<T, tile_size> ( a, wk, nm, n, w );
#endif

    a += len*step; w += n*step;
  }
}


template <class T>
__global__ void
eigen_GPU_batch_(const int L, const int nm, const int n, const int m, T *a_, T *w_, T *e_, T * wk_, const int PRELOAD_SLOT)
{
  const int pos = (threadIdx.x+blockIdx.x*blockDim.x)/WARP_GPU_SIZE;
#if TIMER
  const int myid = (threadIdx.x % WARP_GPU_SIZE);
#endif
  if (pos>=L) return;

  const size_t len = nm*n;
  const int step = (blockDim.x*gridDim.x)/WARP_GPU_SIZE;
  T * a, * w, * wk, * e;

#if DO_PREFETCH
  a = a_ + pos*len;
  #pragma unroll 1
  for (int id_=0; id_<PRELOAD_SLOT; id_+=step) {
    int id = id_ + pos;
    if ( id < L )
    prefetch_mat_cg<T, WARP_GPU_SIZE> (nm*n, a);
    a += len*step;
  }
#endif

#if TIMER
  unsigned long t0, t1, t2, t3;
#endif

  a  = a_  + pos*len;
  w  = w_  + pos*n;
  wk = wk_ + pos*len;
  e  = e_  + pos*n;
  #pragma unroll 1
  for (int id_=0; id_<L; id_+=step) {
    const int id = id_ + pos;
    const bool run = (id < L);

#if DO_PREFETCH
    if ( PRELOAD_SLOT > 0 && id+PRELOAD_SLOT*step < L )
    prefetch_mat_cg<T, WARP_GPU_SIZE> (nm*n, a+PRELOAD_SLOT*len*step);
#endif

#if TIMER
    t0 = __global_timer__();
#endif
//    const int mb=3; // 3*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
    const int mb=4; // 3*(mb+1) <= n+1 && 2*mb <= WARP_GPU_SIZE
    T * u_=wk+0*nm*(mb+1); // mb*nm+max(BLK_J,BLK_K)
    T * v_=wk+1*nm*(mb+1); // mb*nm+max(BLK_J,BLK_K)
    if(run) hhsy2tr_ (nm, n, a, w, e, mb, u_, v_);
#if TIMER
    t1 = __global_timer__();
#endif
    int return_flag = false;
#if USE_IMTQL
    if(run) return_flag = imtql2_ (nm, n, w, e, wk);
#else
    if(run) return_flag = tql2_ (nm, n, w, e, wk);
#endif
    if ( return_flag ) break;
#if TIMER
    t2 = __global_timer__();
#endif
    if(run) hhtr2sy_ (nm, n, a, m, wk
#if DO_SORT
                             , (int*)e
#endif
		                       );
#if TIMER
    t3 = __global_timer__();
#endif

    a += len*step; w += n*step;
  }

#if TIMER
  if (pos==0 && myid == 0) {
    double t;
    t = (double)(t1-t0);
    printf("TRED1  : %le [ns]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 4.*n*n*n/3/t, (double)sizeof(T)*n*n/2/t);
    t = (double)(t2-t1);
    printf("TQL2   : %le [ns]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 0., (double)sizeof(T)*n*(n+2)/t);
    t = (double)(t3-t2);
    printf("TRBAK1 : %le [ns]: %le [GFLOPS]: %le [GB/s]\n",
		       t, 2.*n*n*n/t, (double)sizeof(T)*3*n*n/2/t);
  }
#endif
}


template <class T>
__host__ gpuError_t
eigen_GPU_batch_RUN(const int L, const int nm, const int n, const int m, T * a, T * w, T * wk_, const gpuStream_t stream)
{
  if (L<1||nm<n||n<1||m<1||m>n) return gpuErrorInvalidValue;

  gpuError_t err_code;
  int grpSZ=0, numTB=0, numGR=0, numTH=0, sizeSH=0, sizeL2=0;
  err_code = eigen_GPU_batch_get_Launch_params<T>(L, n, grpSZ, numTB, numGR, numTH, sizeSH, sizeL2);
  if (err_code != gpuSuccess) return err_code;

#if DO_PREFETCH
  const int len = nm*n;
  const int step = (numTH*numTB)/grpSZ;
  const int ELE = sizeof(T)*len*step;
//  const int lwk = sizeof(T)*(n+(n*nm))*min(numTB*numGR,L);
//  const int PRELOAD_SLOT = max(sizeL2-lwk,0)/ELE;
  const int PRELOAD_SLOT = min(L,(sizeL2-1)/ELE);
#else
  const int PRELOAD_SLOT = 0;
#endif

  if (n == 1) {
#if defined(__NVCC__)
    cudaFuncSetAttribute( eigen_GPU_batch_n1_ <T>,
		    cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
    eigen_GPU_batch_n1_ <T> <<<numTB, numTH, 0, stream>>> (L, nm, n, a, w);
  } if ( n <= WARP_GPU_SIZE ) {
    switch ( grpSZ ) {
    case 4:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,4>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 4> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 8:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,8>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 8> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 16:
#if defined(__NVCC__)
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,16>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 16> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
    case 32:
#if defined(__NVCC__)
    default:
      cudaFuncSetAttribute( eigen_GPU_batch_tiled_ <T,32>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
      eigen_GPU_batch_tiled_ <T, 32> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
#if WARP_GPU_SIZE==64
    case 64:
    default:
      eigen_GPU_batch_tiled_ <T, 64> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, wk_, PRELOAD_SLOT);
      break;
#endif
    }
  } else {
    T *e = wk_;
    T *wk = e + n*min(numTB*numGR,L);
#if defined(__NVCC__)
    cudaFuncSetAttribute( eigen_GPU_batch_ <T>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 16*1024 );
#endif
    eigen_GPU_batch_ <T> <<<numTB, numTH, sizeSH, stream>>> (L, nm, n, m, a, w, e, wk, PRELOAD_SLOT);
  }

  return gpuSuccess;
}

extern "C" {

__host__ void
eigen_GPU_batch_DP(const int L, const int nm, const int n, const int m, double * a, double * w, double *wk, const gpuStream_t stream)
{
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <double>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int64_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <double>(L, nm, n, m, a, w, wk, stream);
  }
}

__host__ void
eigen_GPU_batch_FP(const int L, const int nm, const int n, const int m, float * a, float * w, float *wk, const gpuStream_t stream)
{
  if ( wk == NULL ) {
    if ( n > 1 ) {
      size_t lwork;
      eigen_GPU_batch_BufferSize <float>(L, nm, n, m, NULL, NULL, &lwork);
      *((size_t *)w) = lwork;
    } else {
      *((int32_t *)w) = 0;
    }
  } else {
    eigen_GPU_batch_RUN <float>(L, nm, n, m, a, w, wk, stream);
  }
}

//__host__ void
//eigen_GPU_batch_HP(const int L, const int nm, const int n, const int m, half * a, half * w, half *wk)
//{
//  eigen_GPU_batch_RUN <half>(L, nm, n, m, a, w, wk);
//}

}
