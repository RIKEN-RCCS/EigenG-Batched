#ifndef __HEADER_TQL2_HPP__
#define __HEADER_TQL2_HPP__


template <class T>
__device__ __noinline__ int
tql2_( const int nm, const int n,
        T * __restrict__ d_, T * __restrict__ e_, T * __restrict__ z_,
	// optional
	int const MAX_SWEEP = 100,
        T const tol = std::numeric_limits<T>::epsilon()*(std::is_same<T,double>::value?512:16),
        bool const DO_SORT = false
     )
{
  const int myid = threadIdx.x % WARP_GPU_SIZE + 1;
#define	z(row,col)	(*(z_+((row)-1)+((col)-1)*nm))
#define	d(index)	(*(d_+((index)-1)))
#define	e(index)	(*(e_+((index)-1)))
#define	pos(index)	(*(pos_+((index)-1)))

  const int tile_size = WARP_GPU_SIZE;
  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);


  int ierror = 0;
  _if_ (n == 1) { return ierror; }

  #pragma unroll 1
  for(int i=1; i<=n; i++) {
    #pragma unroll 1
    for(int j=myid; j<=n; j+=WARP_GPU_SIZE) {
      z(j, i) = ZERO;
    }
  }
  #pragma unroll 1
  for(int i=myid; i<=n; i+=WARP_GPU_SIZE) {
    z(i, i) = ONE;
  } sync_over_warp();

  T shift = ZERO;
  T tst1 = ZERO;

  #pragma unroll 1
  for(int l=1; l<=n; l++) { // most outer loop

    tst1 = Max(tst1, (Abs(d(l)) + Abs(e(l)))*tol);

    sync_over_warp();

    int m = 1;
    _if_ (myid == 1) {
      #pragma unroll 1
      for(m=l; m<=n; m++) {
        const T tst2 = Abs(e(m));
        _if_ (tst2 <= tst1) break;
      }
    } bcast_over_warp(m,1);

    _if_ (m != l) { // non-isolated diagonal

      int itr;
      #pragma unroll 1
      for(itr=0; itr<MAX_SWEEP; itr++) {

	T dl1;
        T delta_d;
        sync_over_warp(); {
          const T dl_old = d(l);
          const T el = e(l);
          const T p = Div(d(l+1) - dl_old, el + el);
          const T r = pythag1(p);
          const T psr = p + Sign(r, p);
          const T dl  = Div(el, psr);
                  dl1 = el * psr;
                  _if_ (myid==1) { d(l) = dl; d(l+1) = dl1; }
                  delta_d = dl_old - dl;
        } sync_over_warp();

        for(int i=l+1+myid; i<=n; i+=WARP_GPU_SIZE) {
          d(i) -= delta_d;
        } sync_over_warp();
        shift += delta_d;

        T c = ONE;
        T c2 = c;
        T c3 = c;
        T s = ZERO;
        T s2 = s;

        T p = d(m);
        const T el1 = e(l+1);

        #pragma unroll 1
        for(int i=m-1; i>=l; i--) {

          c3 = c2;
          c2 = c;
          s2 = s;

	  const T ei = e(i);
	  const T di = d(i);
          const T g = c * ei;
          const T h = c * p;
          const T r = pythag(p, ei);

          const T ei1 = s * r;
          s = Div(ei, r);
          c = Div(p, r);
          p = c * di - s * g;
          const T di1 = h + s * (c * g + s * di);
          sync_over_warp();
	  _if_ (myid==1) { e(i + 1) = ei1; d(i + 1) = di1; }

	  {
	    T * zki0_ptr = &z(myid,i+0);
            T * zki1_ptr = &z(myid,i+1);
            #pragma unroll 1
            for(int k=myid; k<=n; k+=WARP_GPU_SIZE ) {
              const T h0 = *zki0_ptr;
              const T h1 = *zki1_ptr;
              *zki1_ptr = s * h0 + c * h1;
              *zki0_ptr = c * h0 - s * h1;
              zki0_ptr+=WARP_GPU_SIZE; zki1_ptr+=WARP_GPU_SIZE;
            }
          }

        } sync_over_warp();

        _if_ (myid == 1) {
          const T r = Div(-s * s2 * c3 * el1 * e(l), dl1);
          e(l) = s * r;
          d(l) = c * r;
        } sync_over_warp();

	{
          const T tst2 = Abs(e(l));
          _if_(tst2 <= tst1) break;
        }

      } sync_over_warp();
      _if_ (itr>=MAX_SWEEP) { ierror = l; break; }

    }

    _if_ (myid==1) {
      d(l) += shift;
    } sync_over_warp();

  }

  _if_ (DO_SORT) {
    _if_ (ierror == 0) {
      int * const pos_ = (int *)e_;
      for(int i=myid; i<=n; i+=WARP_GPU_SIZE) {
        pos(i) = i;
      } sync_over_warp();
      #pragma unroll 1
      for(int i=2; i<=n; i++) {
        const int l = i - 1;
        _if_ (myid==1) {
          T dl = d(l);
          int il = l;
          #pragma unroll 1
          for (int j=i; j<=n; j++) {
            const T dj = d(j);
            const bool flag = dl > dj;
            __UPDATE__(dl, dj, flag);
            __UPDATE__(il, j, flag);
          }
          _if_ (il!=l) {
            int p=pos(l); pos(l)=pos(il); pos(il)=p;
            d(il)=d(l); d(l)=dl;
          }
        }
      } sync_over_warp();
    }
  }

#undef	z
#undef	d
#undef	e
  sync_over_warp();
  return ierror;
}

#endif
