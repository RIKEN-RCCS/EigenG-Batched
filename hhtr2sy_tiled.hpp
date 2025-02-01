#pragma once

template <class T, int tile_size>
//__device__  __forceinline__ void
__device__  __noinline__ void
hhtr2sy_tiled_( const long nm, const int n, T * __restrict__ a_, T * __restrict__ z_, const bool do_sort = (DO_SORT == 1) )
{
  sync_on_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define z(row,col)      (*(z_+((row)-1)+((col)-1)*nm))

  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MTWO = static_cast<T>(-2.0e0);


  _if_ (n == 1) {
    _if_ (myid == 1) {
      z(1, 1) = ONE;
    } sync_on_cg<T,tile_size>();
    return;
  }
  _if_ (n == 2) {
    _if_ (myid <= n) {
      const T ei = a(2, 2);
      _if_ (ei != ZERO) {
        const T t = ONE + Div(a(1, 2), ei);
        z(1, myid) *= t;
      }
    } sync_on_cg<T,tile_size>();
    return;
  }


#if defined(__HIPCC__)
  const int BLK_I = (tile_size>=16)?3:((tile_size>=8)?3:2);
  const int BLK_J = (tile_size>=16)?4:((tile_size>=8)?3:2);
#else
  const int BLK_I = (tile_size>=16)?3:((tile_size>=8)?3:2);
  const int BLK_J = (tile_size>=16)?4:((tile_size>=8)?3:2);
#endif
  const int ii = (n-1) % BLK_I + 1;

  sync_on_cg<T,tile_size>();
  #pragma unroll 1
  for (int i=2; i<=ii; i++) {
    const int l = i - 1;
    const bool eee = (myid <= l);

    const int myk = min(myid,n);
    const T a_xid = __MASK__( a(myk,i), eee );
    const T reciprocal_ali_ei = Reciprocal(flip0to1(a(i,i)*a(l,i)));
    const T b_xid = a_xid * reciprocal_ali_ei;

    T *zki_ptr = &z(myk,1);
    #pragma unroll 1
    for (int j=1; j<=n; j++) {
      T z_xid = __MASK__( *zki_ptr, eee );
      T s = a_xid * z_xid;
      sum_on_cg<T,tile_size>(s);
      z_xid += s * b_xid;
      _if_ ( eee ) { *zki_ptr = z_xid; }
      zki_ptr += nm;
    }
  }


  //
  // Based on Joffraint's HH reflector aggregation, ACM TOMS 32(2), 2006
  //
  // 1. compute G = A[0:BLK_I]^T A[0:BLK_I] only upper triangle
  // 2. update the diagonal by half of reciprocal with singularity flips
  // 3. compute S = A^T Z
  // 4. update  S = G^{-1} S
  // 5. update  Z = Z - A S
  

  #pragma unroll 1
  for (int i=ii+1; i<=n; i+=BLK_I) {
    const int ll = i+(BLK_I-1)-1;
    const bool eee = (myid<=ll);

    _if_(myid==1){
    for(int I=0;I<BLK_I-1;I++) {
    for(int K=I;K<BLK_I-1;K++) {
      a(i+K,i+I) = ZERO;
    }}}
    sync_on_cg<T,tile_size>();

    T G[BLK_I][BLK_I];
    T ai_myid[BLK_I];
    for(int I=0;I<BLK_I;I++) {
      ai_myid[I] = eee ? a(myid, i+I): ZERO;
      for(int K=0;K<=I;K++) {
        G[K][I] = ai_myid[K] * ai_myid[I];
    }}
#if 1
    { int I=0; int K=0; int IIKK=BLK_I*(BLK_I+1)/2;
    for(int IK=0;IK<IIKK%4;IK++) {
      sum_on_cg<T,tile_size>(G[K][I]);
      K++; _if_(K>I) { I++; K=0; }
    }
    for(int IK=IIKK%4;IK<IIKK;IK+=4) {
      int I0=I; int K0=K; K++; _if_(K>I) { I++; K=0; }
      int I1=I; int K1=K; K++; _if_(K>I) { I++; K=0; }
      int I2=I; int K2=K; K++; _if_(K>I) { I++; K=0; }
      int I3=I; int K3=K; K++; _if_(K>I) { I++; K=0; }
      sum4_on_cg<T,tile_size>(G[K0][I0],G[K1][I1],G[K2][I2],G[K3][I3]);
    }}
#else
    for(int I=0;I<BLK_I;I++) {
    for(int K=0;K<=I;K++) {
        sum_on_cg<T,tile_size>(G[K][I]);
    }}
#endif
    for(int I=0;I<BLK_I;I++) {
      G[I][I] = Div(MTWO, flip0to1(G[I][I]));
    }

    const int jj=n % BLK_J;

    T * zkj_ptr = &z(min(myid,n),1);
    #pragma unroll 1
    for (int j=1; j<=jj; j++) {
      T s[BLK_I]; for(int I=0; I<BLK_I; I++) { s[I] = ZERO; }
      T z_myid;
      {
        z_myid = zkj_ptr[0];
        for(int I=0;I<BLK_I;I++) {
          s[I] = ai_myid[I] * z_myid;
        }
      }
#if 1
      { int II=BLK_I%4;
        _if_(II&0x2) sum2_on_cg<T,tile_size>(s[0],s[1]);
        _if_(II&0x1) sum_on_cg<T,tile_size>(s[II-1]);
      for(int I=II;I<BLK_I;I+=4) {
        sum4_on_cg<T,tile_size>(s[I],s[I+1],s[I+2],s[I+3]);
      }}
#else
      _if_(BLK_I%2==1) {
        sum_on_cg<T,tile_size>(s[0]);
      } for(int I=BLK_I%2;I<BLK_I;I+=2) {
        sum2_on_cg<T,tile_size>(s[I],s[I+1]);
      }
#endif
      {
        for(int I=0;I<BLK_I;I++) {
        for(int K=0;K<I;K++) {
          s[I] += s[K]*G[K][I];
        }
          s[I] *= G[I][I];
          z_myid += s[I] * ai_myid[I];
        }
	_if_(eee) { zkj_ptr[0] = z_myid; }
      }
      zkj_ptr += nm;
    }

    #pragma unroll 1
    for (int j=1+jj; j<=n; j+=BLK_J) {
      T s[BLK_I][BLK_J]; for(int I=0; I<BLK_I; I++) {
      for(int J=0; J<BLK_J; J++) { s[I][J] = ZERO; } }
      T z_myid[BLK_J];
      {
        for(int J=0;J<BLK_J;J++) {
          z_myid[J] = zkj_ptr[J*nm];
	}
        for(int J=0;J<BLK_J;J++) {
        for(int I=0;I<BLK_I;I++) {
          s[I][J] = ai_myid[I] * z_myid[J];
        }}
      }
#if 1
      { int II=(BLK_I*BLK_J)%4;
        _if_(II&0x2) sum2_on_cg<T,tile_size>(s[0][0],s[1%BLK_I][1/BLK_I]);
        _if_(II&0x1) sum_on_cg<T,tile_size>(s[(II-1)%BLK_I][(II-1)/BLK_I]);
      for(int IJ=II;IJ<BLK_I*BLK_J;IJ+=4) {
        int I0=(IJ+0)%BLK_I; int J0=(IJ+0)/BLK_I;
        int I1=(IJ+1)%BLK_I; int J1=(IJ+1)/BLK_I;
        int I2=(IJ+2)%BLK_I; int J2=(IJ+2)/BLK_I;
        int I3=(IJ+3)%BLK_I; int J3=(IJ+3)/BLK_I;
        sum4_on_cg<T,tile_size>(s[I0][J0],s[I1][J1],s[I2][J2],s[I3][J3]);
      }}
#else
      for(int J=0;J<BLK_J;J++) {
      _if_(BLK_I%2==1) {
        sum_on_cg<T,tile_size>(s[0][J]);
      } for(int I=BLK_I%2;I<BLK_I;I+=2) {
        sum2_on_cg<T,tile_size>(s[I][J],s[I+1][J]);
      }}
#endif
      {
        for(int J=0;J<BLK_J;J++) {
        for(int I=0;I<BLK_I;I++) {
        for(int K=0;K<I;K++) {
          s[I][J] += s[K][J]*G[K][I];
        }
          s[I][J] *= G[I][I];
          z_myid[J] += s[I][J] * ai_myid[I];
        }
	_if_(eee) { zkj_ptr[J*nm] = z_myid[J]; }
        }
      }

      zkj_ptr += BLK_J*nm;
    }
  }


  _if_ (myid <= n) {
  int * pos_ = (int *)(shmem + tile_size);
  for(int i=1; i<=n; i++) {
    T * aa_ = &a(myid,i);
    const int col = do_sort ? pos(i) : i;
    T * zz_ = &z(myid,col);
    *aa_ = *zz_;
  }}

#undef a
#undef z
  sync_on_cg<T,tile_size>();
}

