#ifndef __HEADER_HHSY2TR_TILED_HPP__
#define __HEADER_HHSY2TR_TILED_HPP__


template <class T, unsigned tile_size>
__device__ __noinline__ void
//__device__ __forceinline__ void
//hhsy2tr_tiled_(const int nm, const int n, T * __restrict__ a_, const int mb, T * __restrict__ u_, T * __restrict__ v_ )
hhsy2tr_tiled_(const int nm, const int n, T * __restrict__ a_, const int mb_, T * __restrict__ u_, T * __restrict__ v_ )
{
  const int mb = 1;
  sync_over_cg<T,tile_size>();
  const unsigned myid = (unsigned)threadIdx.x % tile_size + 1;
#define a(row,col)      (*(a_+((row)-1)+((col)-1)*nm))
#define u(row,col)      (*(u_+((row)-1)+((col)-1)*nm))
#define v(row,col)      (*(v_+((row)-1)+((col)-1)*nm))
#define d(index)        (*(d_+((index)-1)))
#define e(index)        (*(e_+((index)-1)))
#define um(index)       (*(um_+((index)-1)))
#define vm(index)       (*(vm_+((index)-1)))
#define uk(index)       (*(uk_+((index)-1)))
#define vk(index)       (*(vk_+((index)-1)))

  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>( 0.0e0);
  const T ONE  = static_cast<T>( 1.0e0);

  T * d_ = shmem;
  T * e_ = d_ + tile_size;

  _if_ (n==1) { return; }
  _if_ (n==2) {
    _if_ (myid==1) {
      const T t = Abs(a(1,2));
      d(1) = a(1,1);
      d(2) = a(2,2);
      a(1,1) = ZERO;
      a(2,2) = -t;
      a(1,2) = 2*t;
      e(1) = -t;
      e(2) = ZERO;
    } sync_over_cg<T,tile_size>();
    return;
  }

  const int ib0 = Div(n-1,mb)+1;
  const int ib1 = max(1,2-mb);

  T el = ZERO;
#pragma unroll 1
  for(int ib=ib0; ib>=ib1; ib--) {

    const int i0 = (ib-1)*mb;
    const int i1 = min(i0+mb,n);
    const int m0 = i1 - i0;
    const int m1 = max(1,2*(2-ib));

#pragma unroll 1
    for(int m=m0; m>=m1; m--) {
      const int i = i0 + m;
      const int l = i - 1;

      T * um_ = (&u(1,m));
      T * vm_ = (&v(1,m));

      const bool eee = (myid<=i);
      const bool fff = (myid<=l);
      T u_myid, beta; {
        _if_ (myid<=i1) { um(myid) = vm(myid) = ZERO; }
        const int j = min(myid,i);
        u_myid = eee ? a(myid,i) : ZERO;
	T * U_ = &u(0,m0);
	T * V_ = &v(0,m0);
#pragma unroll 1
        for(int k=m0; k>=m+1; k--) {
          u_myid = u_myid + U_[j] * V_[i] + V_[j] * u_[i];
	  U_ -= nm; V_ -= nm;
        }

        _if_ (myid==i) { d(myid) = u_myid; }
        u_myid = __MASK__(u_myid,fff);
        T anorm = u_myid * u_myid;
        sum_over_cg<T,tile_size>(anorm);

	T ul = u_myid - (anorm = - Sign(Sqrt(anorm), u_myid));
        //beta = __MASK__(ONE, anorm != ZERO) / flip0to1(ul * anorm);
        beta = Reciprocal(flip0to1(ul * anorm));
        const bool ggg = (myid == l);
        __UPDATE__(el, anorm, ggg);
        __UPDATE__(u_myid, ul, ggg);
        _if_ (eee) { um(myid) = u_myid; }
      } bcast_over_cg<T,tile_size>(beta,l);


      // v := (UV + VU) * u
      T v_myid = ZERO;
      _if_ (m0>=m+1) {
        const int j = min(myid,l);
        T *uk_ = &u(j,m0);
        T *vk_ = &v(j,m0);
#pragma unroll 1
        for(int k=m0; k>=m+1; k--) {
          const T vk = __MASK__(*vk_,fff);
          T f = vk * u_myid;
          const T uk = __MASK__(*uk_,fff);
          T g = uk * u_myid;
          sum2_over_cg<T,tile_size>(f,g);
          v_myid = v_myid + f*uk + g*vk;
          uk_ -= nm;
          vk_ -= nm;
        }
      }


      // v := A * u + v
      {
        const int lx1 = (l & 0x1);
	_if_ (lx1) {
          T vj = (*a_) * u_myid;
          _if_ (myid==lx1) { v_myid += vj; }
        }
        // min(myid,nm) is added due to a guard for out-of-bound
        T *ajk_ptr = &a(min(myid,nm),1+lx1);
        const int k0 = 1+lx1;
        const int kx = l-myid;
#pragma unroll 1
        for (int k=k0, km=k-myid; km<=kx; k+=2, km+=2, ajk_ptr+=(2*nm)) {
          const T ajk0 = *(ajk_ptr+0*nm);
          const bool eee = (km>=-1);
          const T ajk1 = __MASK__(*(ajk_ptr+1*nm), eee);

          const T vj = v_myid + ajk0 * um(k+0) + ajk1 * um(k+1);
          __UPDATE__(v_myid, vj, km>=0);
          const bool fff = (km>=+1);
          const T vk0 = ajk0 * __MASK__(u_myid, fff);
          const T vk1 = ajk1 * u_myid;

          const T vkk = red2_over_cg<T,tile_size>(vk0, vk1, -km);
          v_myid += vkk;
        }
      }


      // v := v + alpha*u
      {
        v_myid *= beta;
        T alpha = v_myid * u_myid;
        sum_over_cg<T,tile_size>(alpha);
        alpha *= (beta * static_cast<T>(0.5));
        v_myid += alpha * u_myid;
        _if_ ( fff ) { vm(myid) = v_myid; }
      }

    } sync_over_cg<T,tile_size>();

    
    const unsigned XDIM = (tile_size<=8)?2:4;
    const unsigned YDIM = tile_size/XDIM;
    const unsigned xid = ((myid-1) % XDIM)+1;
    const unsigned yid = ((myid-1) / XDIM)+1;


    // A := A + UV + VU
    {
    const int i2 = i0 + m1 - 1;
    const int mm = m0 - m1 + 1;
#if defined(__HIPCC__)
    const int BLK_J = (tile_size<=16)?3:4;
    const int BLK_K = (tile_size<=16)?3:4;
    const int BLK_M = (tile_size<=16)?2:3;
#else
    const int BLK_J = (tile_size<=16)?2:3;
    const int BLK_K = (tile_size<=16)?2:3;
    const int BLK_M = (tile_size<=16)?2:3;
#endif
#pragma unroll 1
    for(int k0=1; k0<=i2; k0+=YDIM*BLK_K) { int k=k0+BLK_K*(yid-1);
#pragma unroll 1
    for(int j0=1; j0<=i2; j0+=XDIM*BLK_J) { int j=j0+BLK_J*(xid-1);
    _if_ (j0+1<=k0+YDIM*BLK_K) {

      T ajk[BLK_J][BLK_K];
      for(int K=0;K<BLK_K;K++){
      for(int J=0;J<BLK_J;J++){
        ajk[J][K] = ZERO;
      }}
      for(int K=0;K<BLK_K;K++){
      _if_(k+K<=i2) {
      for(int J=0;J<BLK_J;J++){
      _if_(j+J<=k+K) {
        ajk[J][K]= a(j+J,k+K);
      }}}} 


      T uj[BLK_J][BLK_M], uk[BLK_K][BLK_M], vj[BLK_J][BLK_M], vk[BLK_K][BLK_M];

      asm volatile ("//--open 0");
      const long duv = (long)v_ - (long)u_;
      const long kj = k-j;
      T * ukm = &u(k,m1);
      const int m2=m1+(mm%BLK_M);
#pragma unroll 1
      for(int m=m1; m<=m2-1; m++) { const int M=0;
	  T * ujm = ukm - kj;
          for(int K=0;K<BLK_K;K++){
            const char * UK_ = (char *)&ukm[K];
            uk[K][M] = *(T*)(UK_+0);
            vk[K][M] = *(T*)(UK_+duv);
          for(int J=0;J<BLK_J;J++){
          _if_(K==0){
            const char * UJ_ = (char *)&ujm[J];
            uj[J][M] = *(T*)(UJ_+0);
            vj[J][M] = *(T*)(UJ_+duv);
	  }
            ajk[J][K] = ajk[J][K]
                      + uj[J][M] * vk[K][M]
	              + vj[J][M] * uk[K][M];
	  }} ukm += nm;
      }
      asm volatile ("//--open 1");
#pragma unroll 1
      for(int m=m2; m<=m0; m+=BLK_M) {
          for(int M=0;M<BLK_M;M++){
	  T * ujm = ukm - kj;
          for(int K=0;K<BLK_K;K++){
            const char * UK_ = (char *)&ukm[K];
            uk[K][M] = *(T*)(UK_+0);
            vk[K][M] = *(T*)(UK_+duv);
          for(int J=0;J<BLK_J;J++){
	  _if_(K==0){
            const char * UJ_ = (char *)&ujm[J];
            uj[J][M] = *(T*)(UJ_+0);
            vj[J][M] = *(T*)(UJ_+duv);
          }
            ajk[J][K] = ajk[J][K]
                      + uj[J][M] * vk[K][M]
	              + vj[J][M] * uk[K][M];
	  }} ukm += nm; ujm = ukm - k + j; }
      }
      asm volatile ("//--close");

      for(int K=0;K<BLK_K;K++){
      _if_(k+K<=i2) {
      for(int J=0;J<BLK_J;J++){
      _if_(j+J<=k+K) {
        a(j+J,k+K) = ajk[J][K];
      }}}}
    }}}}
    sync_over_cg<T,tile_size>();

    {
      T *a_ptr = &a(myid,i0+m0);
      T *u_ptr = &u(myid,m0);
#pragma unroll 1
      for(int m=m0; m>=m1; m--) {
        _if_ ( myid<=i0+m ) {
          *a_ptr = *u_ptr;
        }
        a_ptr -= nm; u_ptr -= nm;
      }
    }

  }
  _if_ (myid<=n) {
    e(myid) = el;
  }
  sync_over_cg<T,tile_size>();

  _if_ (myid==1) {
    d(1) = a(1,1);
    a(1,1) = e(n) = ZERO;
  }
  for(int j=myid+1; j<=n; j+=32) {
    a(j,j) = e(j-1);
  }


#undef a
#undef u
#undef v
#undef d
#undef e
#undef um
#undef vm
#undef uk
#undef vk
  sync_over_cg<T,tile_size>();
}

#endif
