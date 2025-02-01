#pragma once

template <class T, int tile_size>
//__device__  __forceinline__ int
__device__  __noinline__ int
imtql2_tiled_( const int nm, const int n,
                T * __restrict__ w_, T * __restrict__ z_,
                // optional arguments
                int const max_sweep = 100,
		T const tol = machine_epsilon<T>()*(std::is_same<T,double>::value?512:16),
                bool const do_sort = (DO_SORT==1)
             )
{
  sync_on_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define        z(row,col)        (*(z_+((row)-1)+((col)-1)*nm))
#define        w(index)          (*(w_+((index)-1)))
#define        d(index)          (*(d_+((index)-1)))
#define        e(index)          (*(e_+((index)-1)))
#define        pos(index)        (*(pos_+((index)-1)))

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);

  T * shmem = __SHMEM__();

  int ierror = 0;
  _if_ (n == 1) { return ierror; }

  const bool eee = ( myid <= n );

  // d and e are passed from tred1

  _if_ ( eee ) {
    #pragma unroll 1
    for(int i=1; i<=n; i++) {
      z(myid, i) = ZERO;
    }
    z(myid, myid) = ONE;
  } sync_on_cg<T,tile_size>();

  #pragma unroll 1
  for(int l=1; l<=n; l++) { // most outer loop

    SYNC_IF_NEEDED();

    int itr;
    #pragma unroll 1
    for (itr=0; itr<=max_sweep; itr++) {

      T * const d_ = shmem;
      T * const e_ = shmem + tile_size;

      sync_on_cg<T,tile_size>();
      int m; {
        m = min(l+myid,n)-1;
        const T tst1 = (Abs(d(m)) + Abs(d(m+1)))*tol;
        const T tst2 = Abs(e(m));
        const bool eee = (tst2 > tst1) || (l == n);
        m = (eee ? n : m);
        min_on_cg<int,tile_size>(m);
      } _if_ ( m==l ) break; // converged


      // Wilkinson initial guess for the 2x2 corner matrix
      // [dl el; el dl1]
      T di1, r; {
        const T dl = d(l);
        const T dl1 = d(l+1);
        const T el = e(l);
        const T f = Div(dl1 - dl, el+el);
        const T g = pythag1(f);
        di1 = d(m);
        r = (di1 - dl) + Div(el, f + Sign(g, f));
      }


      T s = ONE;
      T c = ONE;
      T delta_d = ZERO;

      T * zki1_ptr = &z(min(myid,n),m);
      T h1 = *zki1_ptr;

      int i;
      #pragma unroll 1
      for(i=m-1; i>=l; i--) {

        const T ei = e(i);
        const T f = s * ei;
        const T g = r;                  
        r = pythag(f, g);
        _if_ (r == ZERO) break;

        const T b = c * ei;
        s = Div(f, r);
        c = Div(g, r);

        const T dix = di1 - delta_d;
        di1 = d(i);
        const T q = (di1 - dix) * s + 2 * c * b;
        const T dx1 = fma(s, q, dix);
        delta_d = dx1 - dix;
        sync_on_cg<T,tile_size>();
        _if_ (myid==1) { e(i+1) = r; d(i+1) = dx1; }
        r = c * q - b;

        T * const zki0_ptr = zki1_ptr - nm;
        const T h0 = *zki0_ptr;
        const T hh = s * h0 + c * h1;
        _if_ ( eee ) { *zki1_ptr = hh; }
        zki1_ptr = zki0_ptr;
        h1 = c * h0 - s * h1;

      }
      _if_ ( eee ) { *zki1_ptr = h1; }

      _if_ (myid == 1) {
        const int j = max(i+1,l);
        d(j) -= delta_d;
        e(j) = r;
        e(m) = ZERO;
      }

    }
    _if_ (itr>max_sweep) { ierror = l; break; }

  } sync_on_cg<T,tile_size>();


  _if_ ( do_sort ) {
    _if_ ( ierror == 0 ) {
      T * const d_ = shmem;
      int * const pos_ = (int *)(shmem + tile_size);
      _if_(myid<=n) {
        pos(myid) = myid;
      } sync_on_cg<T,tile_size>();
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
      } sync_on_cg<T,tile_size>();
    }
  }
  _if_ (myid<=n) {
    T * const d_ = shmem;
    // store back the sorted eigenvalues d() onto w()
    // eigenvectors z(,) will be sorted back in hhtr2sy or trbak1
    w(myid) = d(myid);
  }


  // pos is on shmem, and passd to trbak1

#undef        z
#undef        w
#undef        d
#undef        e
#undef        pos
  sync_on_cg<T,tile_size>();
  return ierror;
}

