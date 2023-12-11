#ifndef __HEADER_TRBAK1_TILED_HPP__
#define __HEADER_TRBAK1_TILED_HPP__

template <class T, int tile_size>
__device__ __noinline__ void
trbak1_tiled_(const long nm, const int n, T *a_, T *z_, T *w_)
{
  setup_cg(tile_size);
  sync_over_cg();
  const int myid = threadIdx.x % tile_size + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))
#define w(index)	(*(w_+((index)-1)))
#define pos(index)	(*(pos_+((index)-1)))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-tile_size))<<1]);

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE = static_cast<T>(1.0e0);

  int * pos_ = (int *)(shmem + tile_size);
  // pos is passed from tql

  if (n <= 1) { return; }
  if (n == 2) {
    if (myid <= n) {
      a(myid, 1) = z(myid, pos(1));
      a(myid, 2) = z(myid, pos(2));
    } sync_over_cg();
    return;
  }


  const int BLK  = 4; // blocking on 'j' is not effective rather than 'i'
  const int jj = n % BLK;

  #pragma unroll 1
  for (int i=2; i<=n; i++) {
    const int l = i - 1;
    const bool eee = (myid <= l);

    const T a_xid = eee ? a(myid,i) : ZERO;
    T reciprocal_ali_ei;
    if (myid==l) {
      reciprocal_ali_ei = calc_reciprocal_ali_ei(a(i,i), a_xid);
    }
    bcast_over_cg(reciprocal_ali_ei,l);
    const T b_xid = a_xid * reciprocal_ali_ei;

    #pragma unroll 1
    for (int j=1; j<=jj; j++) {
      T zj = eee ? z(myid,j) : ZERO;
      T s  = a_xid * zj;
      sum_over_cg(s);
      zj += s * b_xid;
      if ( eee ) { z(myid,j) = zj; }
    }

    #pragma unroll 1
    for (int j=1+jj; j<=n; j+=BLK) {
      T zj[BLK];
      T s[BLK] = { ZERO, };
      T *zkj_ptr = &z(myid,j);
      if ( eee ) {
        #pragma unroll
        for (int J=0;J<BLK;J++) {
          zj[J] = zkj_ptr[J*nm];
          s[J] = a_xid * zj[J];
        }
      }
      #pragma unroll
      for (int J=0;J<BLK;J++) { sum_over_cg( s[J] ); }
      if( eee ) {
        #pragma unroll
        for (int J=0;J<BLK;J++) {
          zj[J] += s[J] * b_xid;
          zkj_ptr[J*nm] = zj[J];
        }
      }
    } sync_over_cg();
  }

  if (myid<=n) {
    T * aa_ = &a(1,myid);
#if DO_SORT
    T * zz_ = &z(1,pos(myid));
#else
    T * zz_ = &z(1,myid);
#endif
    #pragma unroll 1
    for (int i=1; i<=n; i++) {
      *aa_ = *zz_;
      aa_ ++;
      zz_ ++;
    }
  }

#undef a
#undef z
#undef w
#undef d
#undef pos
  sync_over_cg();
}

#endif
