#include <type_traits>
#include "gpu_arch.h"
#include "misc_gpu.hpp"

template <class T, int tile_size>
__global__ __noinline__ void
symdmt_batch( int const nm, int const n, T const *a_, T const *d_ , T *z_ )
{
  sync_over_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))
#define d(index)        (*(d_+((index)-1)))

  T * shmem = __SHMEM__();

  T constexpr ZERO = static_cast<T>(0.0e0);
  T constexpr ONE  = static_cast<T>(1.0e0);

  bool const eee = (myid <= n);
  int const mxid = (eee ? myid : n);

  int const _J_ = n%4;
  int const _K_ = n%4;

  for(int j=1; j<=_J_; j++){
    T z0 = ZERO;
    for(int k=1; k<=n; k++){
      T a0 = a(mxid,k+0);
      sync_over_cg<T,tile_size>();
      shmem[myid] = d(k+0) * a0;
      sync_over_cg<T,tile_size>();
      z0 += a0 * shmem[j+0];
    }
    _if_(eee) { z(myid,j+0) = z0; }
  }

  for(int j=_J_+1; j<=n; j+=4){

    int const L = myid-j;
    bool const fff = (0<=L && L<4);

    T Z[4]; for(int J_=0;J_<4;J_++){ Z[J_] = ZERO; }

    for(int k=1; k<=_K_; k++){
      T a0 = a(mxid,k+0);
      sync_over_cg<T,tile_size>();
      _if_(fff) { shmem[L+ 0] = d(k+0) * a0; }
      sync_over_cg<T,tile_size>();
      for(int J_=0;J_<4;J_++){
        Z[J_] += a0 * shmem[J_];
      }
    }

    for(int k=_K_+1; k<=n; k+=4) {

      T A[4], D[4];
      for(int K_=0;K_<4;K_++){ D[K_] = d(k+K_) * (A[K_] = a(mxid,k+K_)); }
      sync_over_cg<T,tile_size>();
      _if_(fff) {
        for(int K_=0;K_<4;K_++){ shmem[K_+4*L] = D[K_]; } }
      sync_over_cg<T,tile_size>();

      for(int J_=0;J_<4;J_++){
      for(int K_=0;K_<4;K_++){
        Z[J_] += A[K_] * shmem[K_+4*J_];
      }
      }

    }

    _if_(eee) { for(int J_=0;J_<4;J_++) { z(myid,j+J_) = Z[J_]; } }

  }

#undef	a
#undef	z
#undef	d
  sync_over_cg<T,tile_size>();
}


void
sub( int nm, int n, float * a_, float * d_, float * z_ )
{
  symdmt_batch < float, 32 > <<< 32, 1, sizeof(float)*32, NULL >>> ( nm, n, a_, d_ , z_ );
}

