#include <stdio.h>
#include <math.h>
#include <limits>
#include <omp.h>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "gpu_arch.h"
#include "misc.hpp"
#include "eigen_GPU_batch.hpp"
#include "eigen_GPU_check.hpp"

#if defined(__NVCC__)
#include "cusolver.hpp"
#include "cusolver_evd.hpp"
#endif
#if defined(__HIPCC__)
#include "hipsolver.hpp"
#endif


#if 0
extern "C" {
// for calling the original fortran EISPACK routines
  void tred1_(int *, int *, double *, double *, double *);
  void tql2_(int *, int *, double *, double *, double *, int *);
  void trbak1_(int *, int *, double *, int *, double *);
}
#endif

enum class Matrix_type {
  MATRIX_SYM_RAND,
  MATRIX_LETKF,
  MATRIX_FRANK
};
enum class Solver_type {
  EIGENG_BATCH,
  CUSOLVER_EVJ_BATCH,
  CUSOLVER_EVD,
  HIPSOLVER_EVJ_BATCH,
  HIPSOLVER_EVD
};


template < class T > __host__ T
rand_R( unsigned int * seed_ptr )
{
  return (T)( ((double)rand_r(seed_ptr))/RAND_MAX );
}

template < class T > __host__ void
set_mat(T *a, const int nm, const int n, const Matrix_type type, const int seed_)
{
  unsigned int seed = seed_;

  if ( type == Matrix_type::MATRIX_LETKF ) {

    T * w = (T *)malloc(sizeof(T)*n);
    for(int i=1;i<=n;i++){
      const T eps = std::numeric_limits<T>::epsilon();
      const T R = 16-1;
      const T delta = R*(2*rand_R<T>(&seed)-1.)*eps;
      const T DELTA = 1.0+delta;
      w[i-1] = (T)(n-1)*DELTA;
    }
    const int numV=2;
    for(int i=1;i<=numV;i++){
      const int j=(rand_r(&seed)%n);
      w[j] += rand_R<T>(&seed)*n;
    }

    T scale = 0;
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      scale = (T)fmax(scale, fabs(w[i-1]));
    }
    scale = (T)fmax(scale, 1.);
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      w[i-1] /= scale;
    }

    T * ht = (T *)malloc(sizeof(T)*n);
#pragma omp parallel for
    for(int i=1;i<=n;i++){
      ht[i-1] = (T)sqrt((double)i);
    }

#pragma omp parallel
    {

    T * hi = (T *)malloc(sizeof(T)*n);
    T * hj = (T *)malloc(sizeof(T)*n);

    int i0=0, j0=0;
#pragma omp for collapse(2)
    for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){

    if(i0!=i){
    if(i==1){
      T hi_ = ht[n-1];
#pragma omp simd
      for(int k=1;k<=n;k++) hi[k-1] = 1. / hi_;
    } else {
      T s = (T)(i-1);
      T hi_ = ht[i-2] * ht[i-1];
#pragma omp simd
      for(int k=1;k<=i-1;k++) hi[k-1] = 1. / hi_;
      hi[i-1] = -s / hi_;
#pragma omp simd
      for(int k=i+1;k<=n;k++) hi[k-1] = 0.;
    }
    i0=i;
    }

    if(j0!=j){
    if(j==1){
      T hj_ = ht[n-1];
#pragma omp simd
      for(int k=1;k<=n;k++) hj[k-1] = 1. / hj_;
    } else {
      T s = (T)(j-1);
      T hj_ = ht[j-2] * ht[j-1];
#pragma omp simd
      for(int k=1;k<=j-1;k++) hj[k-1] = 1. / hj_;
      hj[j-1] = -s / hj_;
#pragma omp simd
      for(int k=j+1;k<=n;k++) hj[k-1] = 0.;
    }
    j0=j;
    }

      T t = 0.;
#pragma omp simd
      for(int k=1;k<=n;k++) {
        const T hijk = hi[k-1]*hj[k-1];
        t += w[k-1]*hijk;
      }
      a[(j-1)+(i-1)*nm] = t;

    }
    }
      free(hi);
      free(hj);
    }

#pragma omp parallel for collapse(2)
    for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
      a[(j-1)+(i-1)*nm] *= scale;
    }}

    free(w); free(ht);

  } else {

#pragma omp parallel for
    for(int i=0;i<n;i++) {
    for(int j=0;j<=i;j++) {
      T x, t;
      switch ( type ) {
      case Matrix_type::MATRIX_SYM_RAND:
#pragma omp critical
	{
        t = static_cast<T>(static_cast<double>(rand_r(&seed))/RAND_MAX);
	}
        x = 2*t - static_cast<T>(1.0);
        break;
      case Matrix_type::MATRIX_FRANK:
	{
        int k = min(j+1,i+1);
        x = static_cast<T>(k);
	}
        break;
      default:
        x = static_cast<T>(0.);
        break;
      }
      a[i+nm*j] = x; a[j+nm*i] = x;
    }}
  }
}

template < class T, Solver_type Solver > __host__ void
GPU_batch_test(const int Itr, const int L, const int n, const Matrix_type type, const bool accuracy_test)
{
//  const int nm = n + (n&0x1);
  const int nm = n;
  const int m  = n;
  size_t len;

  len = sizeof(T)*L*nm*n;
  T *a_h = (T *)malloc(len);
  if ( a_h == NULL ) return;

  gpuSetDevice(0);
  T *a_d = NULL;  gpuMalloc(&a_d, len);
  if ( a_d == NULL ) { free(a_h); return; }
  T *b_d = NULL;  gpuMalloc(&b_d, len);
  if ( b_d == NULL ) { free(a_h); gpuFree(a_d); return; }
  len = sizeof(T)*L*m;
  T *w_d = NULL;  gpuMalloc(&w_d, len);
  if ( w_d == NULL ) { free(a_h); gpuFree(a_d); gpuFree(b_d); return; }

  {
#if PRINT_DIAGNOSTIC
    gpuDeviceSynchronize();
    double ts = get_wtime();
#endif
    #pragma omp parallel for
    for(int id=0; id<L; id++) {
      T *a = a_h + (size_t)id*(nm*n);
      set_mat(a, nm, n, type, id);
    }
#if PRINT_DIAGNOSTIC
    gpuDeviceSynchronize();
    double te = get_wtime();
    printf("  Data generation :: %le [s]\n", te-ts);
#endif
  }
  len = sizeof(T)*L*nm*n;
  {
#if PRINT_DIAGNOSTIC
    gpuDeviceSynchronize();
    double ts = get_wtime();
#endif
    gpuMemcpy(b_d, a_h, len, gpuMemcpyHostToDevice);
#if PRINT_DIAGNOSTIC
    gpuDeviceSynchronize();
    double te = get_wtime();
    printf("  Host -> Dev :: %le [s]\n", te-ts);
#endif
  }

  T *wk_d = NULL;
  if ( Solver == Solver_type::EIGENG_BATCH ) {
    eigen_GPU_batch_BufferSize(L, nm, n, m, a_d, w_d, &len);
    gpuMalloc(&wk_d, len);
    if ( wk_d == NULL ) { free(a_h);
      gpuFree(a_d); gpuFree(b_d); gpuFree(w_d); return; }
  }

  gpuStream_t stream;
  gpuStreamCreate( &stream );

  double total_time = 0;
  for(int ITR=0; ITR<Itr; ITR++) {

    len = sizeof(T)*L*nm*n;
    gpuMemcpy(a_d, b_d, len, gpuMemcpyDeviceToDevice);

    gpuDeviceSynchronize();
    gpuSetDevice(0);
    double ts = get_wtime();
    switch ( Solver )  {
    case Solver_type::EIGENG_BATCH:
      eigen_GPU_batch(L, nm, n, m, a_d, w_d, wk_d, stream);
      break;
#if defined(__NVCC__)
    case Solver_type::CUSOLVER_EVJ_BATCH:
      cusolver_test(n, a_d, nm, w_d, L);
      break;
    case Solver_type::CUSOLVER_EVD:
      cusolver_evd_test(n, a_d, nm, w_d, L);
      break;
#endif
#if defined(__HIPCC__)
    case Solver_type::HIPSOLVER_EVJ_BATCH:
      hipsolver_test(n, a_d, nm, w_d, L);
      break;
#endif
    default:
      break;
    }
    gpuSetDevice(0);

    gpuDeviceSynchronize();
    double te = get_wtime();
    total_time += (te - ts);
  }

  if (accuracy_test) {
#if PRINT_DIAGNOSTIC
    gpuDeviceSynchronize();
    double ts = get_wtime();
#endif
    eigen_GPU_check(L, nm, n, m, b_d, w_d, a_d, stream);
#if PRINT_DIAGNOSTIC
    double te = get_wtime();
    printf("  Accuracy Test :: %le [s]\n", te-ts);
#endif
  }

  gpuStreamDestroy( stream );
  double tm = total_time / Itr;
  double flop = L*(double)n*n*n*(4./3+2.);
  double ld_data = sizeof(T)*L*(double)n*(n/2.    +2.   +n/2.+n);
  double st_data = sizeof(T)*L*(double)n*(n/2.+2. +n+1. +n     );
  double data = ld_data + st_data;

  printf("N=%d time=%le %le[GF/s] %le[GB/s]\n",
         n, tm, 1e-9*flop/tm, 1e-9*data/tm);
  fflush(stdout);

  gpuFree(a_d);
  gpuFree(b_d);
  gpuFree(w_d);
  if ( Solver == Solver_type::EIGENG_BATCH ) { gpuFree(wk_d); }
  free(a_h);
}


int
main(int argc, char* argv[])
{
  print_header("GPU-Batch-eigensolver", argc, argv);

//  const int iter = 100;
  const int iter = 20;
  const int numBatch = 16384;
//  const Matrix_type type = Matrix_type::MATRIX_FRANK;
  const Matrix_type type = Matrix_type::MATRIX_LETKF;
//  const Matrix_type type = Matrix_type::MATRIX_SYM_RAND;

#if defined(__NVCC__)
  const int nums[] = { 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 23, 24, 25, 28, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192, 193, 223, 224, 225, 255, 256, 257, 319, 320, 321, 511, 512, 0 };
#endif
#if defined(__HIPCC__)
  const int nums[] = { 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 23, 24, 25, 28, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192, 193, 223, 224, 225, 255, 256, 0 };
  //const int nums[] = { 65, 96, 97, 128, 129, 0 };
#endif

//  printf(">> half accuracy test\n");
//  for(int n=1; n<=96; n*=2)
//    GPU_batch_test<half,Solver_type::EIGENG_BATCH>(1, 1, n, type, true);

  printf(">> float accuracy test\n");
  for(int i=0; nums[i] > 0; i++) { const int n = nums[i];
    GPU_batch_test<float,Solver_type::EIGENG_BATCH>(1, 1, n, type, true);
  }
  printf("\n");
  printf(">> double accuracy test\n");
  for(int i=0; nums[i] > 0; i++) { const int n = nums[i];
    GPU_batch_test<double,Solver_type::EIGENG_BATCH>(1, 1, n, type, true);
  }

  printf("\n");
  printf(">> float EigenG TQL average of %d iterations.\n",iter);
  for(int i=0; nums[i] > 0; i++) { const int n = nums[i];
    GPU_batch_test<float,Solver_type::EIGENG_BATCH>(iter, numBatch, n, type, true);
  }

  printf("\n");
  printf(">> double EigenG TQL avarage of %d iterations.\n",iter);
  for(int i=0; nums[i] > 0; i++) { const int n = nums[i];
    GPU_batch_test<double,Solver_type::EIGENG_BATCH>(iter, numBatch, n, type, true);
  }

#if defined(__NVCC__)
  printf("\n");
  printf(">> float cusolver Jacobi acc_check and avarage of %d iterations.\n",iter);
  for(int n=8; n<=128; n*=2) {
    GPU_batch_test<float,Solver_type::CUSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<float,Solver_type::CUSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }

  printf("\n");
  printf(">> double cusolver Jacobi acc_check and avarage of %d iterations.\n",iter);
  for(int n=8; n<=128; n*=2) {
    GPU_batch_test<double,Solver_type::CUSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<double,Solver_type::CUSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }
#endif

#if defined(__HIPCC__)
  printf("\n");
  printf(">> float hipsolver Jacobi acc_chk and avarage of the %d iterations.\n",iter);
  for(int n=8; n<=64; n*=2) {
    GPU_batch_test<float,Solver_type::HIPSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<float,Solver_type::HIPSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }

  printf("\n");
  printf(">> double hipsolver Jacobi acc_chk and avarage out of the %d iterations.\n",iter);
  for(int n=8; n<=64; n*=2) {
    GPU_batch_test<double,Solver_type::HIPSOLVER_EVJ_BATCH>(1, 1, n, type, true);
    GPU_batch_test<double,Solver_type::HIPSOLVER_EVJ_BATCH>(iter, numBatch, n, type, false);
  }
#endif

}


