#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <omp.h>


int min( int x, int y ) { return (x <= y) ? x : y; }
int max( int x, int y ) { return (x >= y) ? x : y; }
float Sqrt( float x ) { return sqrtf(x); }
double Sqrt( double x ) { return sqrt(x); }
float Abs( float x ) { return fabsf(x); }
double Abs( double x ) { return fabs(x); }


template <class T>
T Sign( T x, T y )
{
  if ( y >= 0 ) {
    return x;
  } else {
    return -x;
  }
}

template <class T>
void set_mat(T *a, const int nm, const int n, const int seed_)
{
  for(int i=0;i<n;i++) {
    for(int j=0;j<=i;j++) {
      T x, t;
      int k = min(j+1,i+1);
      x = static_cast<T>(k);
      a[i+nm*j] = a[j+nm*i] = x;
    }
  }
}

template <class T>
void find_maxpos(const int n, int &i, int &j, int *maxpos, T *absele)
{
  const T ZERO = static_cast<T>(0);
  int j0=1; T bik = ZERO;
    printf("aele: %le:%d %le:%d %le:%d %le:%d\n",
                  absele[1], maxpos[1],
                  absele[2], maxpos[2],
                  absele[3], maxpos[3],
                  absele[4], maxpos[4]);
  for(int k=1; k<=n; k++) {
    T aik = absele[k];
    if ( bik < aik ) {
      bik = aik; j0 = k;
    }
  }
  i = maxpos[j0]; j = j0;
  if ( i < j ) { int k = i; i = j; j = k; }
}

template <class T>
void jac( T * a_, T * z_, const int nm, const int n, T * w_, T tol, void * work )
{
#define	a(row,col)	(*(a_+((row)-1)+((col)-1)*nm))
#define	z(row,col)	(*(z_+((row)-1)+((col)-1)*nm))
#define	w(indx)		(*(w_+(indx-1)))

  const T ZERO = static_cast<T>(0);
  const T ONE  = static_cast<T>(1);

  T * absele = (T *)work;
  int * maxpos = (int *)(absele+n);
  absele--;
  maxpos--;

  for(int l=1;l<=n;l++) {
    for(int k=1;k<=n;k++) {
      z(k,l) = ZERO;
    }
    z(l,l) = ONE;
    w(l) = a(l,l);
  }

  for(int l=1;l<=n;l++) {
    int pos = 0;  T amax = ZERO;
    for(int k=1;k<=n;k++) {
      if ( k!=l ) {
        T akl = Abs(a(k,l));
        if ( akl > amax ) { amax = akl; pos = k; }
      }
    }
    maxpos[l] = pos;
    absele[l] = amax;
  }

  int i=0,j=0;
  for ( int itr=0; ; itr++ ) {

    int i0=i; int j0=j;
    find_maxpos(n, i, j, maxpos, absele);
    if ( i0==i && j0==j ) break; // impossible to converge any more

    T aij = a(i,j);
    T bij = absele[i];
    T aii = w(i);
    T ajj = w(j);
#if 1
    printf("[%d]|aij|=%le %d %d\n", itr,bij,i,j);
#endif
    if ( bij < (Abs(aii)+Abs(ajj))*tol ) break; // converge

      printf("A=\n");
      printf("%le %le %le %le\n", a(1,1),a(1,2),a(1,3),a(1,4));
      printf("%le %le %le %le\n", a(2,1),a(2,2),a(2,3),a(2,4));
      printf("%le %le %le %le\n", a(3,1),a(3,2),a(3,3),a(3,4));
      printf("%le %le %le %le\n", a(4,1),a(4,2),a(4,3),a(4,4));

    T alpha = 2*aij;
    T beta = ajj - aii;
    T t;
    if ( Abs(alpha) >= Abs(beta) ) {
      T tau = beta / alpha;
      T r = Sqrt(ONE + tau*tau); 
      t = ONE / (tau + Sign(r, tau));
    } else {
      T tau = alpha / beta;
      T r = Sqrt(ONE + tau*tau);
      t = tau / (ONE + r);
    }
    T c = ONE / Sqrt(ONE + t*t);
    T s = c * t;

    //[c -s][a b][c +s] = 
    //[+s c][b d][-s c]
    // [ca-sb cb-sd][c  s] = [cca-csb-csb+ssd
    // [sa+cb sb+cd][-s c]   [csa+ccb-ssb-scd

    w(i) = (c*c)*aii - 2*(c*s)*aij + (s*s)*ajj;
    w(j) = (s*s)*aii + 2*(c*s)*aij + (c*c)*ajj;
    aij = (c*s)*(aii-ajj) + (c-s)*(c+s)*aij;
    a(i,j) = a(j,i) = aij;

    absele[i] = absele[j] = Abs(aij);


    for(int k=1;k<=n;k++) {
      {
        // multiply U(theta) to z from the right side
	T zki = z(k,i);
        T zkj = z(k,j);

        T bki = c*zki - s*zkj;
        T bkj = s*zki + c*zkj;
	z(k,i) = bki;
	z(k,j) = bkj;
      }
      if ( k!=i && k!=j ) {
        // multiply U(theta) to a from the left-right side
        T aki = a(k,i);
        T akj = a(k,j);

        T bik = c*aki - s*akj;
        T bjk = s*aki + c*akj;
        a(i,k) = a(k,i) = bik;
        a(j,k) = a(k,j) = bjk;

	if ( maxpos[k] == i ) {
          absele[k] = Abs(bik);
	}
	if ( maxpos[k] == j ) {
          absele[k] = Abs(bjk);
	}
      }
    }

      printf("Z=\n");
      printf("%le %le %le %le\n", z(1,1),z(1,2),z(1,3),z(1,4));
      printf("%le %le %le %le\n", z(2,1),z(2,2),z(2,3),z(2,4));
      printf("%le %le %le %le\n", z(3,1),z(3,2),z(3,3),z(3,4));
      printf("%le %le %le %le\n", z(4,1),z(4,2),z(4,3),z(4,4));

    for(int k=1;k<=n;k++) {
      if ( k!=i && k!=j ) {

        T bik = Abs(a(i,k));
        T bjk = Abs(a(j,k));

	// update maxpos[] and absele[]
        if ( absele[i] < bik ) {
          maxpos[i] = k;
          absele[i] = bik;
        }
        if ( absele[j] < bjk ) {
          maxpos[j] = k;
          absele[j] = bjk;
        }

        if ( absele[k] < bik ) {
          maxpos[k] = i;
          absele[k] = bik;
	}
        if ( absele[k] < bjk ) {
          maxpos[k] = j;
          absele[k] = bjk;
	}

      }
    }

#if 0
    printf("<<i,j=%d %d\n", i,j);
    for(int k=1;k<=n;k++) {
      printf("maxpos[%d]=%d absele[%d]=%le\n",k,maxpos[k], k,absele[k]);
    }
#endif

  }

}

int
main(int argc, char *argv[])
{
  int n=4;
  int nm=4;

  float *a_ = (float*)malloc(sizeof(float)*nm*n);
  float *z_ = (float*)malloc(sizeof(float)*nm*n);
  float *w_ = (float*)malloc(sizeof(float)*n);
  void *work = (void*)malloc(sizeof(float)*n+sizeof(int)*n);
  float eps = std::numeric_limits<float>::epsilon();
  float tol = eps*eps;

  set_mat(a_, n, n, 0);
  jac( a_, z_, n, n, w_, tol, work );

  free(a_);
  free(z_);
  free(w_);
  free(work);
}

