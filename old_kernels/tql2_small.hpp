#ifndef __HEADER_TQL2_SMALL_HPP__
#define __HEADER_TQL2_SMALL_HPP__

extern __shared__ char __shmem[];

template <class T>
__device__ __noinline__ int
tql2_small_( const long nm, const int n, T *w_, T *a_)
{
  __syncwarp();
  const int myid = threadIdx.x % 32 + 1;
#define	a(row,col)	(*(a_+((row)-1)+((col)-1)*nm))
#define	w(index)	(*(w_+((index)-1)))
#define	d(index)	(*(d_+((index)-1)))
#define	e(index)	(*(e_+((index)-1)))
#define	pos(index)	(*(pos_+((index)-1)))

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);

  T * shmem = &(((T *)__shmem)[(threadIdx.x&(unsigned)(-32))<<1]);

  int ierror = 0;
  if (n <= 1) { return ierror; }

  const bool eee = ( myid <= n );

  // d and e are passed from tred1
  T * d_ = shmem;
  T * e_ = shmem + 32;

  #pragma unroll 1
  for(int i=1; i<=n; i++) {
    if ( eee ) {
      a(myid, i) = ZERO;
    }
  } __syncwarp();
  if ( eee ) {
    a(myid, myid) = ONE;
  } __syncwarp();

  T shift = ZERO;
  T tst1 = ZERO;

  #pragma unroll 1
  for(int l=1; l<=n; l++) { // most outer loop

    int m;
    if (myid == 1) {
       tst1 = Max(tst1, Abs(d(l)) + Abs(e(l)));
      #pragma unroll 1
      for(m=l; m<=n; m++) {
        const T tst2 = tst1 + Abs(e(m));
        if (tst2 == tst1) break;
      }
    } bcast_over_warp(m,1);

    if (m != l) { // non-isolated diagonal

      int itr;
      #pragma unroll 1
      for(itr=0; itr<30; itr++) {

	T dl1;
        T delta_d;
        if (myid == 1) {
          const T dl_old = d(l);
          const T el = e(l);
          const T p = Div(d(l+1) - dl_old, el + el);
          const T r = pythag(p, ONE);
          const T psr = p + Sign(r, p);
          const T dl  = d(l)   = Div(el, psr);
                  dl1 = d(l+1) = el * psr;
                  delta_d = dl_old - dl;
        } bcast_over_warp(delta_d,1);

        if ( l+2 <= myid && myid <= n ) {
          d(myid) -= delta_d;
        } __syncwarp();
        shift += delta_d;

        T c = ONE;
        T c2 = c;
        T c3 = c;
        T s = ZERO;
        T s2 = s;

        T p = d(m);
        const T el1 = e(l+1);

        T * aki1_ptr = &a(myid,m);
        T h1;
        if ( eee ) { h1 = *aki1_ptr; }
        #pragma unroll 1
        for(int i=m-1; i>=l; i--) {

          if (myid == 1) {
            c3 = c2;
            c2 = c;
            s2 = s;

	    const T ei = e(i);
	    const T di = d(i);
            const T g = c * ei;
            const T h = c * p;
            const T r = pythag(p, ei);

            e(i + 1) = s * r;
            s = Div(ei, r);
            c = Div(p, r);
            p = c * di - s * g;
            d(i + 1) = h + s * (c * g + s * di);
          } bcast_over_warp(c,1); bcast_over_warp(s,1);

	  T * aki0_ptr = aki1_ptr - nm;
	  if ( eee ) {
            T h0 = *aki0_ptr;
            *aki1_ptr = s * h0 + c * h1;
            h1 = c * h0 - s * h1;
          }
	  aki1_ptr = aki0_ptr;

        }
        if ( eee ) { *aki1_ptr = h1; }

        int conv;
        if (myid == 1) { // s2, c3 are from myid=1 in the previous loop
          const T r = Div(-s * s2 * c3 * el1 * e(l), dl1);
          e(l) = s * r;
          d(l) = c * r;
          const T tst2 = tst1 + Abs(e(l));
          conv = (tst2 <= tst1);
        } bcast_over_warp(conv,1);
        if (conv) break;

      } __syncwarp();
      if (itr>=30) { ierror = l; break; }

    }

    if (myid==1) {
      d(l) += shift;
    } __syncwarp();

  }


  int * pos_ = (int *)(shmem + 32);
  pos(myid) = myid;
#if DO_SORT
  if (ierror==0) {
    __syncwarp();
    for (int i=2; i<=n; i++) {
      const int l = i - 1;
      // find minimum element in [l:n] and set it on pos(l)
      if (myid==1) {
        T dl = d(l);
        int il = l;
        for (int j=i; j<=n; j++) {
          T dj = d(j);
          const bool flag = dl > dj;
          dl = flag ? dj : dl;
          il = flag ? j : il;
        }
        if (il!=l) {
          int p=pos(l); pos(l)=pos(il); pos(il)=p;
          d(il)=d(l); d(l)=dl;
        }
      }
    }
    __syncwarp();
  }
#endif
  if (myid<=n) {
    // store back the sorted eigenvalues d() onto w()
    w(myid) = d(myid);
  }

  // pos is on shmem, and passd to trbak1

#undef	a
#undef	w
#undef	d
#undef	e
#undef	pos
  __syncwarp();
  return ierror;
}

#endif
