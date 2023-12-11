#include "gpu_arch.h"
#include "misc_gpu.hpp"

template <class T>
__device__ void
check_(const int id, const int nm, const int n, T *a_, const int m, const T *d_, const T *z_)
{
  const int myid = threadIdx.x % WARP_GPU_SIZE + 1;
  sync_over_warp();
#define	a(row,col)	(*(a_+(row-1)+(col-1)*nm))
#define	z(row,col)	(*(z_+(row-1)+(col-1)*nm))
#define	d(index)	(*(d_+(index-1)))

  const double ZERO = static_cast<double>(0);
  const double ONE  = static_cast<double>(1);

#if 0
  double pi = 4*atan(ONE);

  for (int i=myid; i<=n; i++) {
    const double theta = (2*(n+1-i)-1) * pi / (2*n+1);
    const double r = ONE / (2 * (ONE-cos(theta)));
    const double err = Abs((double)d(i)-r);
    const double EPS  = (ONE / (1<<26)) / (1<<26);
    const double eps  = (ONE / (1<<23));
    const double TOL = 256*(double)(std::is_same<T,double>::value?EPS:eps);
    if ( err > Abs((double)d(n))*TOL ) {
      printf("d:=%le ext:=%le res=%le\n",
        (double)d(i), r, err/Abs((double)d(n)));
    }
  }
#endif

  double err1 = ZERO;
  for (int i=1; i<=m; i++) {
    for (int k=myid; k<=n; k+=WARP_GPU_SIZE) {
      double ek = (double)ZERO;
      for (int j=1; j<=n; j++) {
        ek += (double)a(k,j) * (double)z(j,i);
      }
      const double dxk = (double)d(i)*(double)z(k,i);
      const double r = ek - dxk;
      err1 = err1 + r * r;
    }
  } sum_over_warp(err1);
  err1 = Sqrt(err1);

  double err2 = ZERO;
  for (int i=1; i<=m; i++) {
    for (int j=i; j<=m; j++) {
      double r = ZERO;
      for (int k=myid; k<=n; k+=WARP_GPU_SIZE) {
        r += (double)z(k,i)*(double)z(k,j);
      } sum_over_warp(r);
      const double t = (i==j ? ONE : ZERO);
      const double c = (i==j ? ONE: 2*ONE);
      r -= t;
      r *= r;
      err2 += c * r;
    }
  } sum_over_warp(err2);
  err2 = Sqrt(err2);

  if ( myid == 1 ) {
    const double D_left = fabs((double)d(1));
    const double D_right = fabs((double)d(n));
    const double Dmax = fmax(D_left, D_right);
    const double EPS  = (ONE / (1<<26)) / (1<<26);
    const double eps  = (ONE / (1<<23));
//    const double TOL = (std::is_same<T,double>::value?EPS*512:eps*16)*sqrt((double)(n+24));
    const double TOL = (std::is_same<T,double>::value?EPS*(2*512-1):eps*(2*16-1))*sqrt((double)(n+16));

    double e1 = (( err1 > Dmax*TOL ) ? (err1/Dmax) : ZERO);
    double e2 = (( err2 > n*TOL ) ? err2 : ZERO);

    float * ans = (float *)(a_);
    ans[0] = (float)(e1);
    ans[1] = (float)(e2);
    ans[2] = (float)(TOL);
    ans[3] = (float)(std::is_same<T,double>::value?EPS:eps);
  }

#undef	a
#undef	z
#undef	d
#undef	e
  sync_over_warp();
}

template <class T>
__global__ void
parallel_check_(const int L, const int nm, const int n, T *a_, const int m, const T *d_, const T *z_)
{
  const int pos = (threadIdx.x+blockIdx.x*blockDim.x)/WARP_GPU_SIZE;
  const int step = (blockDim.x*gridDim.x)/WARP_GPU_SIZE;
  for(int id=pos; id<L; id+=step){
    T *a = (T *)a_ + (size_t)id*nm*n;
    T *d = (T *)d_ + (size_t)id*n;
    T *z = (T *)z_ + (size_t)id*nm*n;
    check_(L, nm, n, a, m, d, z);
  }
}

#if defined(__NVCC__)
template <class T>
__global__ void
print_logs_(const int L, const int nm, const int n, const T *a_, const int m, const T *d_, const T *z_)
{
  const double ZERO = static_cast<double>(0);
//  const double ONE  = static_cast<double>(1);

#define	a(row,col)	(*(a_+(row-1)+(col-1)*nm))
#define	z(row,col)	(*(z_+(row-1)+(col-1)*nm))
#define	d(index)	(*(d_+(index-1)))

  for(int id=0; id<L; id++){
    T *a = (T *)a_ + (size_t)id*nm*n;
    T *z = (T *)z_ + (size_t)id*nm*n;
    T *d = (T *)d_ + (size_t)id*n;

    float * ans = (float *)(a);
    double err1 = (double)(ans[0]);
    double err2 = (double)(ans[1]);
    double TOL  = (double)(ans[2]);
    double eps  = (double)(ans[3]);

    if ( err1 != ZERO ) {
      printf("[%06d] Accuracy error in the relative resisdual (||Ax-dx||_F=%le/%le/%le)\n", id, err1,TOL,eps);
    }
    if ( err2 != ZERO ) {
      printf("[%06d] Orthonormality error (||ZZ-I||_F=%le/%le/%le)\n", id, err2,TOL,eps);
    }

  }

//  printf("Eigen %le / %le\n", (double)d_[0],(double)d_[n-1]);

#undef	a
#undef	z
#undef	d
}
#endif

template <class T>
__host__ gpuError_t
eigen_GPU_check_RUN(const int L, const int nm, const int n, const int m, T *a_, T *w_, T *z_, const gpuStream_t stream)
{
  parallel_check_ <T> <<< L, WARP_GPU_SIZE, 0, stream >>> (L, nm, n, a_, m, w_, z_);
#if defined(__HIPCC__)
{
  const double ZERO = static_cast<double>(0);
  const double ONE  = static_cast<double>(1);

  size_t len = sizeof(T)*L*nm*n;
  T *a_h = (T *)malloc(len);
  if ( a_h == NULL ) { return gpuErrorInvalidValue; }
//  gpuMemcpy(a_h, w_, sizeof(T)*n, gpuMemcpyDeviceToHost);
//  printf("Eigen %le / %le\n", (double)a_h[0],(double)a_h[n-1]);
  gpuMemcpy(a_h, a_, len, gpuMemcpyDeviceToHost);
  for(int id=0;id<L;id++) {
    float *ans = (float *)(a_h + (size_t)nm*n*id);
    double err1 = (double)(ans[0]);
    double err2 = (double)(ans[1]);
    double TOL  = (double)(ans[2]);
    double eps  = (double)(ans[3]);
    if ( err1 != ZERO ) {
      printf("[%06d] Accuracy error in the relative resisdual (||Ax-dx||_F=%le/%le/%le)\n", id, err1,TOL,eps);
    }
    if ( err2 != ZERO ) {
      printf("[%06d] Orthonormality error (||ZZ-I||_F=%le/%le/%le)\n", id, err2,TOL,eps);
    }
  }
  free(a_h);
}
#endif
#if defined(__NVCC__)
  print_logs_ <T> <<< 1, 1, 0, stream >>> (L, nm, n, a_, m, w_, z_);
#endif

  return gpuSuccess;
}

extern "C" {

__host__ void
eigen_GPU_check_DP(const int L, const int nm, const int n, const int m, double *a_, double *w_, double *z_, const gpuStream_t stream)
{
  eigen_GPU_check_RUN <double>(L, nm, n, m, a_, w_, z_, stream);
}

__host__ void
eigen_GPU_check_FP(const int L, const int nm, const int n, const int m, float *a_, float *w_, float *z_, const gpuStream_t stream)
{
  eigen_GPU_check_RUN <float>(L, nm, n, m, a_, w_, z_, stream);
}

}

