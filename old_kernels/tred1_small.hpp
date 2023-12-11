#ifndef __HEADER_TRED1_SMALL_HPP__
#define __HEADER_TRED1_SMALL_HPP__

extern __shared__ char __shmem[];

template <class T>
__device__ __noinline__ void
tred1_small_(const long nm, const int n, T *a_)
{
  __syncwarp();
  const int myid = threadIdx.x % 32 + 1;
#define	a(row,col)	(*(a_+((row)-1)+((col)-1)*nm))
#define	u(index)	(*(u_+((index)-1)))
#define	v(index)	(*(v_+((index)-1)))
#define	d(index)	(*(d_+((index)-1)))
#define	e(index)	(*(e_+((index)-1)))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);
  const T MONE = static_cast<T>(-1.0e0);

  T * d_ = shmem;
  T * e_ = shmem + 32;

  if (n<=1) { return; }
  if (n==2) {
    if (myid==1) {
      d(1) = a(1,1);
      d(2) = a(2,2);
      e(1) = a(1,2);
      e(2) = ZERO;
    } __syncwarp();
    return;
  }


  T * u_ = d_;
  T * v_ = e_;
  T el = ZERO;

  #pragma unroll 1
  for (int i=n; i>=2; i--) {
    const int l = i - 1;

    T u_myid = (myid<=l) ? a(myid,i) : ZERO;

    T scale = Abs(u_myid);
    sum_over_warp(scale);
    scale = (scale == ZERO) ? ONE : scale;

    u_myid = Div(u_myid,scale);
    T h = u_myid * u_myid;
    sum_over_warp(h);

    if (myid==l) {
      const T g = Sign(Sqrt(h), u_myid);
      el = - scale * g;
      h = Div(MONE, u_myid * g + h);
      u_myid += g;
    } __syncwarp();
    u(myid) = u_myid;
    bcast_over_warp(h,l);

    T v_myid = ZERO;
    T *ajk_ptr = &a(myid,1);
    if(l%2){
      if (myid==1) {
        const T ajk = *ajk_ptr;
        v_myid = ajk * u_myid;
      } ajk_ptr+=(nm);
    }
    #pragma unroll 1
    for (int k=1+l%2; k<=l; k+=2) {
      T vk0 = ZERO;
      T vk1 = ZERO;
      const T ajk0 = (myid<=k+0) ? *(ajk_ptr) : ZERO;
      const T ajk1 = (myid<=k+1) ? *(ajk_ptr+nm) : ZERO;
      if (myid<=k-1) {
        vk0 = ajk0 * u_myid;
        vk1 = ajk1 * u_myid;
        v_myid = v_myid
            + ajk0 * u(k+0)
            + ajk1 * u(k+1);
      } sum_over_warp(vk0);
      if (myid==k+0) {
        vk1 = ajk1 * u_myid;
        v_myid = vk0
            + ajk0 * u_myid
            + ajk1 * u(k+1);
      } sum_over_warp(vk1);
      if (myid==k+1) {
        v_myid = vk1
            + ajk1 * u_myid;
      } ajk_ptr+=(2*nm);
    }

    {
      v_myid *= h;
      T f = v_myid * u_myid;
      sum_over_warp(f);
      h = f * h * static_cast<T>(0.5);
      v_myid += h * u_myid;
      v(myid) = v_myid;
    } __syncwarp();

    {
      const int nlz_ = 32-__clz(n-1);
      const int nlz  = 1<<nlz_;
      const int xid  = ((myid-1)&(nlz-1))+1;
      const int yid  = ((myid-1)>>nlz_)+1;
      const int step = 32>>nlz_;

      u_myid = u(xid);
      v_myid = v(xid);

      ajk_ptr = &a(xid,yid);
      #pragma unroll 1
      for (int k=yid; k<=l; k+=step) {
        if (xid<=k) {
          T ajk = *ajk_ptr;
	  const T uk = u(k);
	  const T vk = v(k);
          ajk += uk * v_myid;
          ajk += vk * u_myid;
          *ajk_ptr = ajk;
        } ajk_ptr+=(step*nm);
      }
    } __syncwarp();
 
    if (myid<=l) {
      a(myid,i) = u_myid * scale;
    }

  }
  if (myid <= n) {
    d(myid) = a(myid,myid);
    const int j = (myid<n) ? myid+1 : 1;
    e(myid) = a(j,j) = el;
  }

  // d and e are on shmem, they are passed to tql as well.

#undef	a
#undef	u
#undef	v
#undef	d
#undef	e
  __syncwarp();
}

#endif

