

template <class T, int tile_size>
__device__ __noinline__ void
jac_tiled( T * a_, T * z_, const int nm, const int n, T * w_ )
{
  sync_over_cg<T,tile_size>();
  const int myid = threadIdx.x % tile_size + 1;
#define	a(row,col)	(*(a_+((row)-1)+((col)-1)*nm))
#define	z(row,col)	(*(z_+((row)-1)+((col)-1)*nm))
#define	w(indx)		(*(w_+(indx-1)))

  T * shmem = __SHMEM__();

  const T ZERO = static_cast<T>(0.0e0);
  const T ONE  = static_cast<T>(1.0e0);

  const bool eee = ( myid <= n );


  if ( eee ) {
    for(int i=1;i<=n;i++) {
      z(myid,i) = ZERO;
    }
  } sync_over_cg<T,tile_size>();
  if ( eee ) {
    z(myid,myid) = ONE;
    w(myid) = a(myid,myid);
    a(myid,myid) = ZERO;
  } sync_over_cg<T,tile_size>();


  #pragma unroll 1
  while ( 1 ) {

    int stable = 1;
    #pragma unroll 1
    for (int i0=1; i0<=n; i0++) {

      int i; int j; {
        int ii = myid; T x = eee ? Abs(a(myid,i0)) : -ONE;
        #pragma unroll
        for(int lane=1; lane<tile_size; lane<<=1) {
          const auto __cooperative_group__ =
            cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
          const T z    = __cooperative_group__.shfl_xor(x, lane);
          const int kk = __cooperative_group__.shfl_xor(ii, lane);
          const bool flagi = (z>x) || (z==x && ii<kk);
          ii = flagi ? kk : ii;
          x = Max(z, x);
        }
        int jj = myid; T y = eee ? Abs(a(myid,ii)) : -ONE;
        #pragma unroll
        for(int lane=1; lane<tile_size; lane<<=1) {
          const auto __cooperative_group__ =
            cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
          const T z    = __cooperative_group__.shfl_xor(y, lane);
          const int kk = __cooperative_group__.shfl_xor(jj, lane);
          const bool flagi = (z>y) || (z==y && jj<kk);
          jj = flagi ? kk : jj;
          y = Max(z, y);
        }
        i = ii; j = jj; if ( i > j ) { int k=i; i=j; j=k; }
      }
      if ( i==j ) continue;


      T aij = a(j,i);
      const T aii = w(i);
      const T ajj = w(j);
      const T tol = (std::is_same<T,float>::value) ? 1e-10 : 1e-20;
      if ( Abs(aij) < (Abs(aii)+Abs(ajj))*tol ) continue;


      const bool fff = (myid!=i && myid!=j);
      const bool ggg = eee && ( !fff );
      T  *am_ptr = &a(myid,0);

      const T alpha = 2*aij;
      const T beta = ajj - aii;
      const T t = Div(alpha, beta + Sign(pythag(alpha, beta), beta));
      const T c = Reciprocal(pythag1(t));
      const T s = c*t;
      const T cc = c*c;
      const T ss = s*s;
      const T cs = Neg(c*s);

      //[c -s][a b][c +s] = 
      //[+s c][b d][-s c]
      // [ca-sb cb-sd][c  s] = [cca-csb-csb+ssd
      // [sa+cb sb+cd][-s c]   [csa+ccb-ssb-scd

      aij = ((c-s)*(c+s)*aij) + cs*beta;
      const T wi = ((cc*aii) + ss*ajj) + cs*alpha;
      const T wj = (aii + ajj) - wi;

      if ( ggg ) {
        const int k=i+j-myid;
        am_ptr[k*nm] = ZERO;
	w(myid) = (myid==j) ? wj : wi;
      } sync_over_cg<T,tile_size>();

      T  *zm_ptr = &z(myid,0);

      if ( eee ) {

        //
        // multiply U(theta) to a from the left-right side
        //
        const T aki = am_ptr[i*nm];
        const T akj = am_ptr[j*nm];
        const T t = Neg(s);
        const T bki = c*aki + t*akj;
        const T bkj = s*aki + c*akj;

        if ( fff ) {
          a(i,myid) = bki;
          a(j,myid) = bkj;
          am_ptr[i*nm] = bki;
          am_ptr[j*nm] = bkj;
        } else {
          int k=i+j-myid;
          am_ptr[k*nm] = aij;
        }

        //
        // multiply U(theta) to z from the right side
        //
        const T zki = zm_ptr[i*nm];
        const T zkj = zm_ptr[j*nm];
        const T cki = c*zki + t*zkj;
        const T ckj = s*zki + c*zkj;
        zm_ptr[i*nm] = cki;
        zm_ptr[j*nm] = ckj;

      } sync_over_cg<T,tile_size>();
      stable = 0;
    }

    if ( stable ) break;
  }

  if ( eee ) {
    for(int k=1; k<=n; k++) {
      a(myid,k) = z(myid,k);
    }
  }

#undef	a
#undef	z
#undef	w
}

