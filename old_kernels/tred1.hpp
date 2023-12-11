#ifndef __HEADER_TRED1_HPP__
#define __HEADER_TRED1_HPP__

extern __shared__ char __shmem[];

template <class T>
__device__ __noinline__ void
tred1_(const int nm, const int n, T *a_, T *d_, T *e_)
{
  __syncwarp();
  const int myid = threadIdx.x % 32 + 1;
#define	a(row,col)	(*(a_+((row)-1)+((col)-1)*nm))
#define	u(index)	(*(u_+((index)-1)))
#define	v(index)	(*(v_+((index)-1)))
#define	d(index)	(*(d_+((index)-1)))
#define	e(index)	(*(e_+((index)-1)))

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  const T ZERO = static_cast<T>( 0.0e0);
  const T ONE  = static_cast<T>( 1.0e0);
  const T MONE = static_cast<T>(-1.0e0);


  if (n==1) { return; }
  if (n==2) {
    if (myid==1) {
      const T t = Abs(a(1,2));
      d(1) = a(1,1);
      d(2) = a(2,2);
      a(1,1) = ZERO;
      a(2,2) = -t;
      a(1,2) = 2*t;
      e(1) = -t;
      e(2) = ZERO;
    } __syncwarp();
    return;
  }


  #pragma unroll 1
  for (int i=n; i>=2; i--) {
    const int l = i - 1;

    T *u_ = (&a(1, i));
    T *v_ = (&e(1));

    T scale = ZERO;
    for (int k=myid; k<=l; k+=32) {
      scale += Abs(u(k));
    } sum_over_warp(scale);

    if (scale == ZERO) {
      if (myid == 1) {
        e(l) = ZERO;
        d(i) = a(i,i);
      } __syncwarp();
      continue;
    }

    T h = ZERO;
    for (int k=myid; k<=l; k+=32) {
      const T uk = u(k) = Div(u(k), scale);
      h += uk * uk;
    } sum_over_warp(h);

    T el;
    if (myid==1) {
      const T f = u(l);
      T g = Sign(Sqrt(h), f);
      el  = -scale * g;
      h = Div(MONE, f * g + h);
      u(l) = f + g;
      d(i) = a(i,i);
    } bcast_over_warp(h,1);

    #pragma unroll 1
    for (int k=1; k<=l; k++) {
      const T uk = u(k);
      T vk = ZERO;
      T * ajk_ptr = &a(myid,k);
      for (int j=myid; j<=k-1; j+=32, ajk_ptr+=32) {
        const T ajk = *ajk_ptr;
        vk += ajk * u(j);
        v(j) += ajk * uk;
      } sum_over_warp(vk);
      if (myid == 1) {
        vk += a(k, k) * uk;
        v(k) = vk;
      } __syncwarp();
    }

    {
      T f = ZERO;
      for (int j=myid; j<=l; j+=32) {
        v(j) = v(j) * h;
        f += v(j) * u(j);
      } sum_over_warp(f);
      h = f * h * static_cast<T>(0.5);
    }

    for (int j=myid; j<=l; j+=32) {
      v(j) += h * u(j);
    } __syncwarp();

    #pragma unroll 1
    for (int k=1; k<=l; k++) {
      const T uk = u(k);
      const T vk = v(k);
      T * ajk_ptr = &a(myid,k);
      for (int j=myid; j<=k; j+=32, ajk_ptr+=32) {
	T ajk = *ajk_ptr;
        ajk += uk * v(j);
       	ajk += vk * u(j);
	*ajk_ptr = ajk;
      }
    } __syncwarp();
    if (myid==1) {
      e(l) = el;
    }
    for (int k=myid; k<=l; k+=32) {
      u(k) *= scale;
    } __syncwarp();

  } __syncwarp();

  for (int k=myid+1; k<=n; k+=32) {
    a(k,k) = e(k-1);
  }
  if (myid==1) {
    d(1) = a(1,1);
    e(n) = ZERO;
    a(1,1) = ZERO;
  }

#undef	a
#undef	u
#undef	v
#undef	d
#undef	e
  __syncwarp();
}

#endif

