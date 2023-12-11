#ifndef __HEADER_TRBAK1_SMALL_HPP__
#define __HEADER_TRBAK1_SMALL_HPP__

template <class T>
__device__ T calc_reciprocal_ali_ei(const T ei, const T ali)
{
  const T ZERO = static_cast<T>(0.0e0);
  const T ONE = static_cast<T>(1.0e0);

  T ret = ONE;
  if (ei != ZERO) {
    ret = Div(ret, ali);
    ret = Div(ret, ei);
  }
  return ret;
}


template <class T>
__device__ __noinline__ void
trbak1_small_(const long nm, const int n, T *a_, T *z_, T *w_)
{
  __syncwarp();
  const int myid = threadIdx.x % 32 + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))
#define w(index)	(*(w_+((index)-1)))
#define pos(index)	(*(pos_+((index)-1)))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE = static_cast<T>(1.0e0);

  int * pos_ = (int *)(shmem + 32);
  // pos is passed from tql

  if (n <= 1) { return; }
  if (n == 2) {
    if (myid <= n) {
      a(myid, 1) = z(myid, pos(1));
      a(myid, 2) = z(myid, pos(2));
    } __syncwarp();
    return;
  }


  const int nlz_ = max(3,32-__clz(n-1));
  const int nlz  = 1<<nlz_;
  const int xid  = ((myid-1)&(nlz-1))+1;
  const int yid  = ((myid-1)>>nlz_)+1;
  const int step = 32>>nlz_;

  const int BLK  = 4; // blocking on 'j' is not effective rather than 'i'
  const int step_= BLK*step;

  #pragma unroll 1
  for (int i=2; i<=n; i++) {
    const int l = i - 1;

    const T a_xid = (xid<=l) ? a(xid,i) : ZERO;
    T reciprocal_ali_ei;
    if (myid==l) {
      reciprocal_ali_ei = calc_reciprocal_ali_ei(a(i,i), a_xid);
    }
    bcast_over_warp(reciprocal_ali_ei,l);
    const T b_xid = a_xid * reciprocal_ali_ei;

    #pragma unroll 1
    for (int j=1; j<=n%BLK; j++) {
      const bool eee = (myid <= l);
      T zj = eee ? z(myid,j) : ZERO;
      T s  = a_xid * zj;
      sum_over_warp(s);
      zj += s * b_xid;
      if ( eee ) { z(myid,j) = zj; }
    }

    #pragma unroll 1
    for (int j_=1+n%BLK; j_<=n; j_+=step_) {
      int j=j_+(yid-1)*BLK;
      const bool eee = (xid <= l) && (j <= n);
      T zj[BLK];
      T s[BLK];
      T *zkj_ptr = &z(xid,j);
      if ( eee ) {
        #pragma unroll
        for (int J=0;J<BLK;J++) { zj[J] = zkj_ptr[J*nm]; }
      } else {
        #pragma unroll
        for (int J=0;J<BLK;J++) { zj[J] = ZERO; }
      }
      #pragma unroll
      for (int J=0;J<BLK;J++) { s[J] = a_xid * zj[J]; }
      #pragma unroll 1
      for(int lane=1; lane<nlz; lane<<=1) {
        #pragma unroll
        for (int J=0;J<BLK;J++) {
          s[J] += __shfl_xor_sync( 0xffffffff, s[J], lane, nlz );
        }
      }
      if( eee ) {
        #pragma unroll
        for (int J=0;J<BLK;J++) { zj[J] += s[J] * b_xid; }
        #pragma unroll
        for (int J=0;J<BLK;J++) { zkj_ptr[J*nm] = zj[J]; }
      }
    } __syncwarp();
  }

  if (xid<=n && yid<=n) {
    T * aa_ = &a(yid,xid);
#if DO_SORT
    T * zz_ = &z(yid,pos(xid));
#else
    T * zz_ = &z(yid,xid);
#endif
    #pragma unroll 1
    for (int i=yid; i<=n; i+=step) {
      *aa_ = *zz_;
      aa_ +=step;
      zz_ +=step;
    }
  }

#undef a
#undef z
#undef w
#undef d
#undef pos
  __syncwarp();
}

#endif
