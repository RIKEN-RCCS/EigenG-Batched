#ifndef __HEADER_HHTR2SY_SMALL_HPP__
#define __HEADER_HHTR2SY_SMALL_HPP__

extern __shared__ char __shmem[];

template <class T>
__device__ __noinline__ void
hhtr2sy_small_( const long nm, const int n, T *a_, T *z_, T *w_)
{
  const int myid = threadIdx.x % 32 + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MONE = static_cast<T>(-1.0e0);
  const T MTWO = static_cast<T>(-2.0e0);


  if (n == 1) {
    if (myid == 1) {
      z(1, 1) = ONE;
    } __syncwarp();
    return;
  }
  if (n == 2) {
    if (myid <= n) {
      const T ei = a(2, 2);
      if (ei != ZERO) {
        const T t = ONE + Div(a(1, 2), ei);
        z(1, myid) *= t;
      }
    } __syncwarp();
    return;
  }


  const int BLK_I = 4;
  const int BLK_J = 4;


  __syncwarp();
  #pragma unroll 1
  for (int i=2; i<=1+(n-1)%BLK_I; i++) {
    const int l = i - 1;
    const bool eee = (myid <= l);

    const T a_xid = eee ? a(myid,i) : ZERO;
    const T ei = a(i,i);
    if (ei == ZERO) continue;
    T reciprocal_ali_ei;
    if (myid==1) {
      reciprocal_ali_ei = Div(Div(ONE,a(l,i)),ei);
    } bcast_over_warp(reciprocal_ali_ei,1);
    const T b_xid = a_xid * reciprocal_ali_ei;

    #pragma unroll 1
    for (int j=1; j<=n; j++) {
      T z_xid = eee ? z(myid,j) : ZERO;
      T s = a_xid * z_xid;
      sum_over_warp(s);
      z_xid += s * b_xid;
      if ( eee ) { z(myid,j) = z_xid; }
    }
  }


  #pragma unroll 1
  for (int i=2+(n-1)%BLK_I; i<=n; i+=BLK_I) {
    const int ii = i+(BLK_I-1)-1;
    const bool eee = (myid<=ii);

    if(myid==1){
    for(int I=0;I<BLK_I-1;I++) {
    for(int K=I;K<BLK_I-1;K++) {
      a(i+K,i+I) = ZERO;
    }}}
    T G[BLK_I][BLK_I];
    for(int I=0;I<BLK_I;I++) {
    for(int K=0;K<=I;K++) {
      G[K][I]= ZERO;
    }} __syncwarp();

    T ai_myid[BLK_I];
    for(int I=0;I<BLK_I;I++) {
      ai_myid[I] = eee ? a(myid, i+I): ZERO;
      for(int K=0;K<=I;K++) {
        G[K][I] += ai_myid[K] * ai_myid[I];
        sum_over_warp(G[K][I]);
      }
    if (myid==1) {
      if(G[I][I]==ZERO)G[I][I]=ONE;
      G[I][I]=Div(MTWO,G[I][I]);
    } bcast_over_warp(G[I][I],1);
    }

    const int jj=n%BLK_J;

    #pragma unroll 1
    for (int j=1; j<=jj; j++) {
      T s[BLK_I] = { ZERO, };
      T z_myid;
      if( eee ) {
        z_myid = z(myid,j+0);
        for(int I=0;I<BLK_I;I++) {
          s[I] = ai_myid[I] * z_myid;
        }
      } for(int I=0;I<BLK_I;I++) {
        sum_over_warp(s[I]);
      }
      if( eee ) {
        for(int I=0;I<BLK_I;I++) {
        for(int K=0;K<I;K++) {
          s[I] += s[K]*G[K][I];
        }
          s[I] *= G[I][I];
          z_myid += s[I] * ai_myid[I];
        }
        z(myid,j+0) = z_myid;
      }
    }

    #pragma unroll 1
    for (int j=1+jj; j<=n; j+=BLK_J) {
      T s[BLK_I][BLK_J] = { ZERO, };
      T z_myid[BLK_J];
      if ( eee ) {
        for(int J=0;J<BLK_J;J++) {
          z_myid[J] = z(myid,j+J);
        for(int I=0;I<BLK_I;I++) {
          s[I][J] = ai_myid[I] * z_myid[J];
        }}
      } for(int J=0;J<BLK_J;J++) {
        for(int I=0;I<BLK_I;I++) {
          sum_over_warp(s[I][J]);
      }}
      if ( eee ) {
        for(int J=0;J<BLK_J;J++) {
        for(int I=0;I<BLK_I;I++) {
        for(int K=0;K<I;K++) {
          s[I][J] += s[K][J]*G[K][I];
        }
          s[I][J] *= G[I][I];
          z_myid[J] += s[I][J] * ai_myid[I];
        }
          z(myid,j+J) = z_myid[J];
        }
      }
    }
  }

  if (myid <= n) {
  for(int i=1; i<=n; i++) {
    T * aa_ = &a(myid,i);
#if DO_SORT
    T * zz_ = &z(myid,pos(i));
#else
    T * zz_ = &z(myid,i);
#endif
    *aa_ = *zz_;
  }}

#undef a
#undef z
  __syncwarp();
}

#endif
