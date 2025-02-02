#pragma once

template <class T, int tile_size>
__device__  __noinline__ void
//__device__  __forceinline__ void
tred1_tiled_(const int nm, const int n, T * __restrict__ a_)
{
  sync_on_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define        a(row,col)        (*(a_+((row)-1)+((col)-1)*nm))
#define        u(index)        (*(u_+((index)-1)))
#define        v(index)        (*(v_+((index)-1)))
#define        d(index)        (*(d_+((index)-1)))
#define        e(index)        (*(e_+((index)-1)))

  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MONE = static_cast<T>(-1.0e0);

  T * const d_ = shmem;
  T * const e_ = shmem + tile_size;

  _if_ (n<=1) { return; }
  _if_ (n==2) {
    _if_ (myid==1) {
      d(1) = a(1,1);
      d(2) = a(2,2);
      e(1) = a(1,2);
      e(2) = ZERO;
    } sync_on_cg<T,tile_size>();
    return;
  }

#define U_ON_SHMEM	1


  T el = ZERO;

  #pragma unroll 1
  for (int i=n; i>=2; i--) {
    const int l = i - 1;


#if U_ON_SHMEM
    T * const u_ = shmem;
    T u_myid = (myid<=l) ? a(min(myid,l),i) : ZERO;
#else
    T * const u_ = &a(1,i);
    T u_myid = (myid<=l) ? u(min(myid,l)) : ZERO;
#endif


    T beta; {
      T anorm = u_myid * u_myid;
      sum_on_cg<T,tile_size>(anorm);
      const T ul = u_myid - (anorm = - Sign(Sqrt(anorm), u_myid));
      beta = Reciprocal(flip0to1(ul * anorm));
      const bool eee = (myid == l);
      __UPDATE__(el, anorm, eee);
      __UPDATE__(u_myid, ul, eee);
#if U_ON_SHMEM
      u(myid) = u_myid;
#else
      _if_ (myid<=l) { u(myid) = u_myid; }
#endif
    } bcast_on_cg<T,tile_size>(beta,l);


    T v_myid = ZERO; {
      const int lx1 = (l & 0x1);
      /*
       if(l%2==1){
         v(1) = a(1,1) * u(1)
       }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<k+0;j++){
           ajk0 = a(j,k+0)
           v(j) += ajk0 * u(k+0)
           v(k+0) += ajk0 * u(j)
	 }
         v(k+0) += a(k+0,k+0) * u(k+0)
         for(j=1;j<k+1;j++){
           ajk1 = a(j,k+1)
           v(j) += ajk1 * u(k+1)
           v(k+1) += ajk1 * u(j)
         }
         v(k+1) += a(+1k,k+1) * u(k+1)
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<k+1;j++){
           ajk0 = a(j,k+0)
           ajk1 = a(j,k+1)
           if(j<k) v(j) += ajk0 * u(k+0)
           v(j) += ajk1 * u(k+1)
           v(k+0) += ajk0 * u(j)
           v(k+1) += ajk1 * u(j)
         }
         v(k+1) += a(k+1,k+1) * u(k+1)
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<=k+1;j++){
           ajk0 = (j<=k) ? a(j,k+0) : 0
           ajk1 = (j<=k+1) ? a(j,k+1) : 0
           if(j<=k+0) v(j) += ajk0 * u(k+0)
           if(j<=k+0) v(j) += ajk1 * u(k+1)
           v(k+0) += ajk0 * ((j<=k-1) ? u(j) : 0)
           v(k+1) += ajk1 * u(j)
         }
       }
       */
      /*
       if(l%2==1){ v(1) = a(1,1) * u(1) }
       for(k=1+l%2;k<=l;k+=2){
         for(j=1;j<=k+1;j++){
           ajk0 = (j<=k) ? a(j,k+0) : 0
           ajk1 = (j<=k+1) ? a(j,k+1) : 0
           vj = v(j) + ajk0 * u(k+0) + ajk1 * u(k+1)
           if(j<=k+0) v(j) = vj
           vk0(j) = ajk0 * ((j<=k-1) ? u(j) : 0)
           vk1(j) = ajk1 * u(j)
         }
	 red2_on {vk0(:),vk1(:)} => v0 = {0,...,sum(vk0),sum(vk1),0,...,0}
	 v(:) += v0(:)
       }
       */
      _if_ (lx1 != 0) {
        const T vj = (*a_) * u_myid;
        _if_ (myid==lx1) { v_myid = vj; }
      }
      // min(myid,nm) is added due to a guard for out-of-bound
      T *ajk_ptr = &a(min(myid,nm),1+lx1);
      const int k0 = 1+lx1;
      const int kx = l-myid;
      #pragma unroll 1
      for (int k=k0, km=k-myid; km<=kx; k+=2, km+=2, ajk_ptr+=(2*nm)) {
        const bool eee = (km>=0); // k >= myid
        const T ajk0 = __MASK__(*(ajk_ptr+0*nm), eee);
        const T ajk1 = __MASK__(*(ajk_ptr+1*nm), km+1>=0);

        const T vj = (v_myid + ajk0 * u(k+0)) + ajk1 * u(k+1);
        __UPDATE__(v_myid, vj, eee); // k >= myid
        const bool fff = (km>=+1); // k >= myid+1
        const T vk0 = ajk0 * __MASK__(u_myid, fff);
        const T vk1 = ajk1 * u_myid;

	// -km = myid - k -> { myid-(1+lx1), myid-(3+lx1), ... }
	//  myid = {1+lx1, 2+lx1}, {3+lx1,4+lx1}, ...
	//       = {k0+0,k0+1}, {k0+2,k0+3}, ...

        const T vkk = red2_on_cg<T,tile_size>(vk0, vk1, -km);
        v_myid += vkk;
      }
#if U_ON_SHMEM
      _if_ (myid<=l) { *ajk_ptr = u_myid; }
#endif
    }


#if U_ON_SHMEM
    T * const v_ = shmem + tile_size;
#else
    T * const v_ = shmem;
#endif
    {
      v_myid *= beta;
      T alpha = v_myid * u_myid;
      sum_on_cg<T,tile_size>(alpha);
      alpha *= (beta * static_cast<T>(0.5));
      v_myid += alpha * u_myid;
      v(myid) = v_myid;
    } sync_on_cg<T,tile_size>();


    {
      T * ajk_ptr = &a(min(myid,nm),1);
      const int lx1 = (l & 0x1);
      _if_ (lx1) {
        T ajk = *ajk_ptr;
        const T uj = u_myid + u_myid;
        ajk += uj * v_myid;
        _if_ (myid==lx1) { *ajk_ptr = ajk; }
        ajk_ptr += nm;
      }
      #pragma unroll 1
      for (int k=1+lx1; k<=l; k+=2) {
        T ajk0 = *(ajk_ptr);
        ajk0 += u(k+0) * v_myid;
        T ajk1 = *(ajk_ptr+nm);
        ajk1 += u(k+1) * v_myid;
        ajk0 += v(k+0) * u_myid;
        _if_ (myid<=k+0) { *ajk_ptr = ajk0; } ajk_ptr += nm;
        ajk1 += v(k+1) * u_myid;
        _if_ (myid<=k+1) { *ajk_ptr = ajk1; } ajk_ptr += nm;
      }
    }

  } sync_on_cg<T,tile_size>();

  _if_ (myid<=n) {
    const int j = (myid<n) ? myid+1 : 1;
    T * ajj_ptr = a_+(j-1)*(nm+1);
    d(j) = *ajj_ptr;
    e(myid) = *ajj_ptr = el;
  }

  // d and e are on shmem, they are passed to tql as well.

#undef        a
#undef        u
#undef        v
#undef        d
#undef        e
  sync_on_cg<T,tile_size>();
}

