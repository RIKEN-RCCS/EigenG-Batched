#ifndef __HEADER_TRBAK1_HPP__
#define __HEADER_TRBAK1_HPP__

extern __shared__ char __shmem[];

template <class T>
__device__ __noinline__ void
trbak1_( const int nm, const int n, T *a_, const int m, T *z_)
{
  const int myid = threadIdx.x % 32 + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);


  if (m == 0) return;
  if (n == 1) {
    if (myid == 1) {
      z(1, 1) = ONE;
    } __syncwarp();
    return;
  }
  if (n == 2) {
    if (myid <= m) {
      const T ei = a(2, 2);
      if (ei != ZERO) {
        const T t = ONE + Div(a(1, 2), ei);
        z(1, myid) *= t;
      }
    } __syncwarp();
    return;
  }

  const int BLK = 4;

  __syncwarp();
  #pragma unroll 1
  for (int i=2; i<=n; i++) {
    const int l = i - 1;
    const T ei = a(i,i);
    if (ei == ZERO) continue;

    const T reciprocal_ali_ei = Div(Div(ONE,a(l,i)),ei);

    #pragma unroll 1
    for (int j=1; j<=m%BLK; j++) {
      T s = ZERO;
      for (int k=myid; k<=l; k+=32) {
        s += a(k, i) * z(k, j);
      } sum_over_warp(s);
      const T t = s * reciprocal_ali_ei;
      for (int k=myid; k<=l; k+=32) {
        z(k, j) += t * a(k, i);
      } __syncwarp();
    }

    #pragma unroll 1
    for (int j=1+m%BLK; j<=m; j+=BLK) {
      T s[BLK]; for (int J=0;J<BLK;J++) s[J] = ZERO;
      T *aki_ptr; aki_ptr = &a(myid,i);
      T *zkj_ptr; zkj_ptr = &z(myid,j);
      for (int k=myid; k<=l; k+=32, aki_ptr+=32, zkj_ptr+=32) {
        const T aki = *aki_ptr;
        #pragma unroll
        for (int J=0;J<BLK;J++) s[J] += aki * zkj_ptr[J*nm];
      } for (int J=0;J<BLK;J++) sum_over_warp(s[J]);
      #pragma unroll
      for (int J=0;J<BLK;J++) s[J] *= reciprocal_ali_ei;
      aki_ptr = &a(myid,i);
      zkj_ptr = &z(myid,j);
      for (int k=myid; k<=l; k+=32, aki_ptr+=32, zkj_ptr+=32) {
        const T aki = *aki_ptr;
        #pragma unroll
        for (int J=0;J<BLK;J++) zkj_ptr[J*nm] += s[J] * aki;
      } __syncwarp();
    }
  }

  #pragma unroll 1
  for(int i=1; i<=n; i++) {
    T * aa_ = &a(myid,i);
#if DO_SORT
    T * zz_ = &z(myid,pos(i));
#else
    T * zz_ = &z(myid,i);
#endif
    for (int j=myid; j<=n; j+=32) {
      *aa_ = *zz_;
      aa_ +=32;
      zz_ +=32;
    }
  }

#undef a
#undef z
  __syncwarp();
}

#endif
