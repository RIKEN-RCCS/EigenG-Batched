#pragma once

template <class T>
__device__ __noinline__ void
//__device__ __forceinline__ void
hhtr2sy_( const int nm, const int n, T * __restrict__ a_, const int m, T * __restrict__ z_
#if DO_SORT
, int * __restrict__ pos_
#endif
)
{
  const int myid = threadIdx.x % WARP_GPU_SIZE + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))
#define pos(index)      (*(pos_+(index-1)))

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MTWO = static_cast<T>(-2.0e0);


  _if_ (m == 0) return;
  _if_ (n == 1) {
    _if_ (myid == 1) {
      z(1, 1) = ONE;
    } sync_over_warp();
    return;
  }
  _if_ (n == 2) {
    _if_ (myid <= m) {
      const T ei = a(2, 2);
      _if_ (ei != ZERO) {
        const T t = ONE + Div(a(1, 2), ei);
        z(1, myid) *= t;
      }
    } sync_over_warp();
    return;
  }


#if defined(__HIPCC__)
  const int BLK_I = (std::is_same<T,float>::value) ? 5 : 4;
  const int BLK_J = (std::is_same<T,float>::value) ? 4 : 3;
#else
  const int BLK_I = (std::is_same<T,float>::value) ? 4 : 3;
  const int BLK_J = (std::is_same<T,float>::value) ? 4 : 3;
#endif
  const int ii = (n-1)%BLK_I + 1;

  sync_over_warp();
  #pragma unroll 1
  for (int i=2; i<=ii; i++) {
    const int l = i - 1;
    const T reciprocal_ali_ei = Reciprocal(flip0to1(a(i,i)*a(l,i)));

    #pragma unroll 1
    for (int j=1; j<=m; j++) {
      T s = ZERO;
      for (int k=myid; k<=l; k+=WARP_GPU_SIZE) {
        s += a(k, i) * z(k, j);
      } sum_over_warp(s);
      const T t = s * reciprocal_ali_ei;
      for (int k=myid; k<=l; k+=WARP_GPU_SIZE) {
        z(k, j) += t * a(k, i);
      } sync_over_warp();
    }
  }


  #pragma unroll 1
  for (int i=ii+1; i<=n; i+=BLK_I) {
    _if_(myid==1){
    for(int I=0;I<BLK_I-1;I++) {
    for(int K=I;K<BLK_I-1;K++) {
      a(i+K,i+I) = ZERO;
    }}}
    sync_over_warp();

    T G[BLK_I][BLK_I];
    for(int I=0;I<BLK_I;I++) {
    for(int K=0;K<=I;K++) {
      G[K][I]= ZERO;
    }}
    for(int k=myid; k<=i+(BLK_I-1)-1; k+=WARP_GPU_SIZE) {
      T aa[BLK_I];
      for(int I=0;I<BLK_I;I++) {
        aa[I] = a(k,i+I);
      }
      for(int I=0;I<BLK_I;I++) {
      for(int K=0;K<=I;K++) {
        G[K][I] += aa[K] * aa[I];
      }}
    }
#if 1
    { int I=0; int K=0; int IIKK=BLK_I*(BLK_I+1)/2;
    for(int IK=0;IK<IIKK%4;IK++) {
      sum_over_warp(G[K][I]);
      K++; _if_(K>I) { I++; K=0; }
    }
    for(int IK=IIKK%4;IK<IIKK;IK+=4) {
      int I0=I; int K0=K; K++; _if_(K>I) { I++; K=0; }
      int I1=I; int K1=K; K++; _if_(K>I) { I++; K=0; }
      int I2=I; int K2=K; K++; _if_(K>I) { I++; K=0; }
      int I3=I; int K3=K; K++; _if_(K>I) { I++; K=0; }
      sum4_over_warp(G[K0][I0],G[K1][I1],G[K2][I2],G[K3][I3]);
    }}
#else
    for(int I=0;I<BLK_I;I++) {
      for(int K=0;K<=I;K++) {
        sum_over_warp(G[K][I]);
    }}
#endif
    for(int I=0;I<BLK_I;I++) {
      G[I][I]=Div(MTWO, flip0to1(G[I][I]));
    }

    #pragma unroll 1
    for (int j=1; j<=m%BLK_J; j++) {
      T s[BLK_I]; for(int I=0; I<BLK_I; I++) { s[I] = ZERO; }
      for (int k=myid; k<=i+(BLK_I-1)-1; k+=WARP_GPU_SIZE) {
        T f0 = z(k,j+0);
        for(int I=0;I<BLK_I;I++) {
          s[I] += a(k, i+I) * f0;
        }
      }{ int II=BLK_I%4;
        _if_(II&0x2) sum2_over_warp(s[0],s[1]);
        _if_(II&0x1) sum_over_warp(s[II-1]);
      for(int I=II;I<BLK_I;I+=4) {
        sum4_over_warp(s[I],s[I+1],s[I+2],s[I+3]);
      }}
      for(int I=0;I<BLK_I;I++) {
      for(int K=0;K<I;K++) {
        s[I] += s[K] * G[K][I];
      }
        s[I] *= G[I][I];
      }
      for (int k=myid; k<=i+(BLK_I-1)-1; k+=WARP_GPU_SIZE) {
        T f0 = z(k,j+0);
        for(int I=0;I<BLK_I;I++) {
          f0 += s[I] * a(k, i+I);
        }
        z(k,j+0) = f0;
      } sync_over_warp();
    }
    #pragma unroll 1
    for (int j=1+m%BLK_J; j<=m; j+=BLK_J) {
      T s[BLK_I][BLK_J]; for(int I=0; I<BLK_I; I++) {
      for(int J=0; J<BLK_I; J++) { s[I][J] = ZERO; } }
      T * aki_ptr = &a(myid,i);
      T * zkj_ptr = &z(myid,j);
      for (int k=myid; k<=i+(BLK_I-1)-1; k+=WARP_GPU_SIZE) {
        T aa[BLK_I];
        for(int I=0;I<BLK_I;I++) {
          aa[I] = aki_ptr[I*nm];
        } aki_ptr+=WARP_GPU_SIZE;
        for(int J=0;J<BLK_J;J++) {
          const T ff = zkj_ptr[J*nm];
        for(int I=0;I<BLK_I;I++) {
          s[I][J] += aa[I] * ff;
        }} zkj_ptr+=WARP_GPU_SIZE;
      }
#if 1
      { int II=(BLK_I*BLK_J)%4;
        _if_(II&0x2) sum2_over_warp(s[0][0],s[1%BLK_I][1/BLK_I]);
        _if_(II&0x1) sum_over_warp(s[(II-1)%BLK_I][(II-1)/BLK_I]);
      for(int IJ=II;IJ<BLK_I*BLK_J;IJ+=4) {
        int I0=(IJ+0)%BLK_I; int J0=(IJ+0)/BLK_I;
        int I1=(IJ+1)%BLK_I; int J1=(IJ+1)/BLK_I;
        int I2=(IJ+2)%BLK_I; int J2=(IJ+2)/BLK_I;
        int I3=(IJ+3)%BLK_I; int J3=(IJ+3)/BLK_I;
        sum4_over_warp(s[I0][J0],s[I1][J1],s[I2][J2],s[I3][J3]);
      }}
#else
      for(int J=0;J<BLK_J;J++) {
      for(int I=0;I<BLK_I%4;I++) {
        sum_over_warp(s[I][J]);
      } for(int I=BLK_I%4;I<BLK_I;I+=4) {
        sum4_over_warp(s[I][J],s[I+1][J],s[I+2][J],s[I+3][J]);
      }}
#endif
      for(int J=0;J<BLK_J;J++) {
      for(int I=0;I<BLK_I;I++) {
      for(int K=0;K<I;K++) {
        s[I][J] += s[K][J] * G[K][I];
      }
        s[I][J] *= G[I][I];
      }}
      aki_ptr = &a(myid,i);
      zkj_ptr = &z(myid,j);
      for (int k=myid; k<=i+(BLK_I-1)-1; k+=WARP_GPU_SIZE) {
        T f[BLK_J];
        for(int J=0;J<BLK_J;J++) {
          f[J] = zkj_ptr[J*nm];
        }
        for(int I=0;I<BLK_I;I++) {
          const T aa = aki_ptr[I*nm];
        for(int J=0;J<BLK_J;J++) {
          f[J] += s[I][J] * aa;
        }} aki_ptr+=WARP_GPU_SIZE;
        for(int J=0;J<BLK_J;J++) {
          zkj_ptr[J*nm] = f[J];
        } zkj_ptr+=WARP_GPU_SIZE;
      } sync_over_warp();
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
    for (int j=myid; j<=n; j+=WARP_GPU_SIZE) {
      *aa_ = *zz_;
      aa_ +=WARP_GPU_SIZE; zz_ +=WARP_GPU_SIZE;
    }
  }

#undef a
#undef z
  sync_over_warp();
}

