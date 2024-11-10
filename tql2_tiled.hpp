#pragma once

template <class T,int tile_size>
__device__ __noinline__ int
tql2_tiled_( const long nm, const int n,
                T * __restrict__ w_, T * __restrict__ z_,
                int const max_sweep = 100,
                T const tol = machine_epsilon<T>()*(std::is_same<T,double>::value?512:16),
                bool const do_sort = (DO_SORT==1)
           )
{
  sync_over_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define	z(row,col)	(*(z_+((row)-1)+((col)-1)*nm))
#define	w(index)	(*(w_+((index)-1)))
#define	d(index)	(*(d_+((index)-1)))
#define	e(index)	(*(e_+((index)-1)))
#define	pos(index)	(*(pos_+((index)-1)))

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);

  T * shmem = __SHMEM__();

  int ierror = 0;
  _if_ (n <= 1) { return ierror; }

  const bool eee = ( myid <= n );

  // d and e are passed from tred1

  _if_ ( eee ) {
    #pragma unroll 1
    for(int i=1; i<=n; i++) {
      z(myid, i) = ZERO;
    }
    z(myid, myid) = ONE;
  } sync_over_cg<T,tile_size>();

  T shift = ZERO;
  T tst1 = ZERO;

  #pragma unroll 1
  for(int l=1; l<=n; l++) { // most outer loop

    T * const d_ = shmem;
    T * const e_ = shmem + tile_size;

    int m;
    {
      tst1 = Max(tst1, (Abs(d(l)) + Abs(e(l)))*tol);
      m = min(l+myid-1,n);
      const T tst2 = Abs(e(m));
      m = (tst2 <= tst1 ? m : n+1);
      min_over_cg<int,tile_size>(m);
    }
    _if_ (m != l) { // non-isolated diagonal

      int itr;
      #pragma unroll 1
      for(itr=0; itr<max_sweep; itr++) {

	T dl1;
        T delta_d;
	{
          const T dl_old = d(l);
          const T el = e(l);
          const T p = Div(d(l+1) - dl_old, el + el);
          const T r = pythag1(p);
          const T psr = p + Sign(r, p);
          const T dl  = Div(el, psr);
		  dl1 = el * psr;
                  _if_ (myid==1) { d(l) = dl; d(l+1) = dl1; }
                  delta_d = dl_old - dl;
        } sync_over_cg<T,tile_size>();

        _if_ ( l+2 <= myid && eee ) { //myid <= n ) {
          d(myid) -= delta_d;
        } sync_over_cg<T,tile_size>();
        shift += delta_d;

        T c = ONE;
        T c2 = c;
        T c3 = c;
        T s = ZERO;
        T s2 = s;

        T p = d(m);
        const T el1 = e(l+1);

        T * aki1_ptr = &z(myid,m);
        T h1, ei;
        _if_ ( eee ) { h1 = *aki1_ptr; }
        #pragma unroll 1
        for(int i=m-1; i>=l; i--) {

          c3 = c2;
          c2 = c;
          s2 = s;

	          ei = e(i);
	  const T di = d(i);
          const T g = c * ei;
          const T h = c * p;
          const T r = pythag(p, ei);

          const T ei1 = s * r;
          s = Div(ei, r);
          c = Div(p, r);
          p = c * di - s * g;
          const T di1 = h + s * (c * g + s * di);
	  sync_over_cg<int,tile_size>();
          _if_ (myid==1) { e(i+1) = ei1; d(i+1) = di1; }

	  T * aki0_ptr = aki1_ptr - nm;
	  _if_ ( eee ) {
            T h0 = *aki0_ptr;
            *aki1_ptr = s * h0 + c * h1;
            h1 = c * h0 - s * h1;
          }
          aki1_ptr = aki0_ptr;

        }
        _if_ ( eee ) { *aki1_ptr = h1; }

        _if_ (myid == 1) {
          const T r = Div(-s * s2 * c3 * el1 * ei, dl1);
          e(l) = s * r;
          d(l) = c * r;
	} sync_over_cg<int,tile_size>();

        {
          const T tst2 = Abs(e(l));
          _if_ (tst2 <= tst1) break;
        }

      } sync_over_cg<T,tile_size>();
      _if_ (itr>=max_sweep) { ierror = l; break; }

    }

    _if_ (myid==1) {
      d(l) += shift;
    } sync_over_cg<T,tile_size>();

  }


  _if_ (do_sort) {
    _if_ (ierror==0) {
      T * const d_ = shmem;
      int * const pos_ = (int *)(shmem + tile_size);
      if (myid<=n) {
        pos(myid) = myid;
      } sync_over_cg<T,tile_size>();
      for (int i=2; i<=n; i++) {
        const int l = i - 1;
        // find minimum element in [l:n] and set it on pos(l)
        _if_ (myid==1) {
          T dl = d(l);
          int il = l;
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
      }
      sync_over_cg<T,tile_size>();
    }
  }
  _if_ (myid<=n) {
    T * const d_ = shmem;
    // store back the sorted eigenvalues d() onto w()
    w(myid) = d(myid);
  }

  // pos is on shmem, and passd to trbak1

#undef	a
#undef	w
#undef	d
#undef	e
#undef	pos
  sync_over_cg<T,tile_size>();
  return ierror;
}

