#ifndef __HEADER_MISC_GPU_HPP__
#define __HEADER_MISC_GPU_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#endif

#undef COOPERATIVE_GROUP_STANDARD
#if defined(__HIPCC__)
#include <hip/hip_version.h>
#if HIP_VERSION<50700000
#define COOPERATIVE_GROUP_STANDARD	1
#else
//
// 5.7.0 has a strange behavior on compilation, where cooperative group
//       would be wrong in class inheritance or friendship.
// 5.6.x prior is OK
//
#if HIP_VERSION<60000000
#define COOPERATIVE_GROUP_STANDARD	0
#else
#define COOPERATIVE_GROUP_STANDARD	1
#endif
//
//
#endif
#endif
#if !defined(COOPERATIVE_GROUP_STANDARD)
#define COOPERATIVE_GROUP_STANDARD	1
#endif


extern __shared__ __align__(sizeof(float)*4) char __shmem[];
#define	__SHMEM__(...)	(&(((T *)__shmem)[(threadIdx.x&(unsigned)(-tile_size))<<1]))

#if defined(__NVCC__)
#define	_if_	asm volatile ("// if branch is here."); if
#endif

#if defined(__HIPCC__)
#define	_if_  if
#endif

// NOTE::
//	__NVCC__	is on when nvcc compiler is used,
//	__CUDA_ARCH__	is on when nvcc targets to compile
//			the device functions, and
//	__HIPCC__	is on when hipcc compiler is used.

//
// Machine epsilon
// to avoid the conflition to refere to std::numeric_limits<T>::epsilon()
// in a CUDA device function
//
#if defined(__NVCC__)
template < typename T > __host__ __device__
T constexpr machine_epsilon() noexcept { return nextafter(T(1),T(2))-T(1); }
template <> __host__ __device__
double constexpr machine_epsilon <double> () noexcept { return DBL_EPSILON; }
template <> __host__ __device__
float constexpr machine_epsilon <float> () noexcept { return FLT_EPSILON; }
#endif
#if defined(__HIPCC__)
template < typename T > __host__ __device__
T constexpr machine_epsilon() noexcept { return std::numeric_limits<T>::epsilon(); }
#endif

__host__ __device__ __forceinline__
double __Volatile__(const double x) {
  double x_ = x;
#if defined(__NVCC__)
  asm volatile ("": "+d"(x_));
#endif
  return x_;
}
__host__ __device__ __forceinline__
float __Volatile__(const float x) {
  float x_ = x;
#if defined(__NVCC__)
  asm volatile ("": "+f"(x_));
#endif  
  return x_;
}
template < class T >
__host__ __device__ __forceinline__
T __MASK__(const T x, const bool cond) {
  const T ZERO = static_cast<T>(0);
#if defined(__NVCC__)
  asm volatile ("// MASK");
#endif
  return cond ? x : ZERO;
}
template < class T >
__host__ __device__ __forceinline__
void __UPDATE__(T &x, const T y, const bool cond) {
#if defined(__NVCC__)
  asm volatile ("// UPDATE");
#endif
  x = cond ? y : x;
}
template < class T >
__host__ __device__ __forceinline__
void __Cond_swap__(T &x, T &y, const bool cond) {
  T z = x; x = cond ? y : z; y = cond ? z : y;
}


template < class T >
__host__ __device__ __forceinline__
T Add(const T x, const T y) {
  return __Volatile__(x) + __Volatile__(y);
}

template < class T >
__host__ __device__ __forceinline__
T Sub(const T x, const T y) {
  return __Volatile__(x) - __Volatile__(y);
}

template < >
__host__ __device__ __forceinline__
float Sub(const float x, const float y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return __fsub_rn( x, y );
#else
  return __Volatile__(x) - __Volatile__(y);
#endif
}

template < >
__host__ __device__ __forceinline__
double Sub(const double x, const double y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return __dsub_rn( x, y );
#else
  return __Volatile__(x) - __Volatile__(y);
#endif
}

template < class T >
__host__ __device__ __forceinline__
T Mul(const T x, const T y) {
  return __Volatile__(x) * __Volatile__(y);
}

template < class T >
__host__ __device__ __forceinline__
T Div(const T x, const T y) {
  return __Volatile__(x) / __Volatile__(y);
}
__host__ __device__ __forceinline__
float Div(const float x, const float y) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ( "div.rn.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y) );
  return ( z );
#else
  return __Volatile__(x) / __Volatile__(y);
#endif
}
__host__ __device__ __forceinline__
int Div(const int x, const int y) {
  return (int)((unsigned)x /(unsigned)y);
}

__host__ __device__ __forceinline__
double Min(const double x, const double y) {
  return fmin(x,y);
}
__host__ __device__ __forceinline__
float Min(const float x, const float y) {
  return fminf(x,y);
}
__host__ __device__ __forceinline__
int Min(const int x, const int y) {
  return (x<y)?x:y;
}

__host__ __device__ __forceinline__
double Max(const double x, const double y) {
  return fmax(x,y);
}
__host__ __device__ __forceinline__
float Max(const float x, const float y) {
  return fmaxf(x,y);
}
__host__ __device__ __forceinline__
int Max(const int x, const int y) {
  return (x>y)?x:y;
}

__host__ __device__ __forceinline__
double Abs(const double x) {
  return fabs(x);
}
__host__ __device__ __forceinline__
float Abs(const float x) {
  return fabsf(x);
}

__host__ __device__ __forceinline__
double Sqrt(const double x) {
  return sqrt(x);
}
__host__ __device__ __forceinline__
float Sqrt(const float x) {
  return sqrtf(x);
}

__host__ __device__ __forceinline__
double Rsqrt(const double x) {
#if defined(__CUDA_ARCH__)
  double z;
  asm volatile ("rsqrt.f64 %0, %1;" : "=d"(z) : "d"(x) );
  return (z);
#else
#if defined(__HIPCC__)
  return rsqrt(x);
#else
  return 1.0e0/sqrt(x);
#endif
#endif
}
__host__ __device__ __forceinline__
float Rsqrt(const float x) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ("rsqrt.f32 %0, %1;" : "=f"(z) : "f"(x) );
  return (z);
#else
#if defined(__HIPCC__)
  return rsqrtf(x);
#else
  return 1.0f/sqrtf(x);
#endif
#endif
}

__host__ __device__ __forceinline__
float FastSqrt(const float x) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ("sqrt.approx.ftz.f32 %0, %1;" : "=f"(z) : "f"(x) );
  return (z);
#else
  return Sqrt(x);
#endif
}
__host__ __device__ __forceinline__
double FastSqrt(const double x) {
  return Sqrt(x);
}

__host__ __device__ __forceinline__
float Neg(const float x) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ( "neg.f32 %0, %1;" : "=f"(z) : "f"(x) );
  return ( z );
#else
  return ( -x );
#endif
}
__host__ __device__ __forceinline__
double Neg(const double x) {
#if defined(__CUDA_ARCH__)
  double z;
  asm volatile ( "neg.f64 %0, %1;" : "=d"(z) : "d"(x) );
  return ( z );
#else
  return ( -x );
#endif
}

__device__ __forceinline__
float Reciprocal(const float x) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ( "rcp.rn.f32 %0, %1;" : "=f"(z) : "f"(x) );
  return ( z );
#else
  return static_cast<float>(1) / x;
#endif
}
__device__ __forceinline__
double Reciprocal(const double x) {
#if defined(__CUDA_ARCH__)
  double z;
  asm volatile ( "rcp.rn.f64 %0, %1;" : "=d"(z) : "d"(x) );
  return ( z );
#else
  return static_cast<double>(1) / x;
#endif
}

__host__ __device__ __forceinline__
float Sign(const float x, const float y) {
#if defined(__CUDA_ARCH__)
  float z;
  asm volatile ( "copysign.f32\t%0, %1, %2;" : "=f"(z) : "f"(y), "f"(x) );
  return z;
#else
  return copysignf(x, y);
#endif
}
__host__ __device__ __forceinline__
double Sign(const double x, const double y) {
#if defined(__CUDA_ARCH__)
  double z;
  asm volatile ( "copysign.f64\t%0, %1, %2;" : "=d"(z) : "d"(y), "d"(x) );
  return z;
#else
  return copysign(x, y);
#endif
}

__host__ __device__ __forceinline__
float flip0to1(const float x)
{
#if defined(__CUDA_ARCH__)
  float y = x;
  asm volatile (
  "// flip0to1\n\t"
  "{.reg.pred\t%pp;\n\t"
  "setp.equ.f32\t%pp, %0, 0f00000000;\n\t"
  "@%pp mov.b32\t%0, 0f3F800000;\n\t"
  "}" : "+f"(y) );
#else
  float y = (x == (float)0.) ? (float)1. : x;
#endif
  return y;
}
__host__ __device__ __forceinline__
double flip0to1(const double x)
{
#if defined(__CUDA_ARCH__)
  double y = x;
  asm volatile (
  "// flip0to1\n\t"
  "{.reg.pred\t%pp;\n\t"
  "setp.equ.f64\t%pp, %0, 0d0000000000000000;\n\t"
  "@%pp mov.b64\t%0, 0d3FF0000000000000;\n\t"
  "}" : "+d"(y) );
#else
  double y = (x == (double)0.) ? (double)1. : x;
#endif
  return y;
}

template < class T >
__host__ __device__ __forceinline__
T pythag1(const T x)
{
  // Borges 2020, ACM TOMS 1014
  const T ONE = static_cast<T>(1);
  const T x_sq = x * x;
  const T sigma = x_sq + ONE;
  const T h = FastSqrt(sigma);
  const T sigma_e = Sub(sigma,x_sq) - ONE;
  const T tau = fma(x,x,-x_sq) - sigma_e + fma(-h,h,sigma);
  const T HF = ONE / 2;
  const T r = fma(Div(tau,flip0to1(h)),HF,h);
  return r;
}

template < class T >
__host__ __device__ __forceinline__
T pythag(const T x, const T y)
{
  // Borges 2020, ACM TOMS 1014
  const T ONE = static_cast<T>(1);
  const T x_sq = x * x;
  const T y_sq = y * y;
  const T sigma = x_sq + y_sq;
  const T h = FastSqrt(sigma);
  const T sigma_e = Sub(sigma,x_sq) - y_sq;
  const T tau = fma(y,y,-y_sq) + fma(x,x,-x_sq) - sigma_e + fma(-h,h,sigma);
  const T HF = ONE / 2;
  const T r = fma(Div(tau,flip0to1(h)),HF,h);
  return r;
}

#if defined(__NVCC__)
__device__ __forceinline__
unsigned long __global_timer__(void)
{
  unsigned long r;
  asm volatile ( "mov.u64 %0, %globaltimer;" : "=l"(r) );
  return r;
}
#endif

#if defined(__NVCC__)
template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __noinline__
void prefetch_mat_cg(const int len, const T *a) {
  asm volatile ("// prefetch in");
  const int myid  = threadIdx.x % tile_size + 1;
  const int L = 32;
        size_t a_    = (size_t)a + L*(myid-1);
  const size_t a_end = (size_t)a + sizeof(T)*(len);
  #pragma unroll 1
  while ( true ) {
    _if_ ( a_ >= a_end ) break;
    asm volatile (
      "{ .reg.b32 %t;\n\t"
      "ld.global.cg.b32\t%t,[%0];\n\t"
      "}" :: "l"(a_) );
    a_ += (L*tile_size);
  }
  asm volatile ("// prefetch out");
}
#endif


template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void sync_over_cg(void) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  __cooperative_group__.sync();
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void bcast_over_cg(T &x, const int root) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  x = __cooperative_group__.shfl(x, (root)-1);
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void sum_over_cg(T &x_) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  const int myid = __cooperative_group__.thread_rank()+1;
  T x = x_;
  for(int lane=1; lane<tile_size; lane<<=1) {
    x += __cooperative_group__.shfl_xor(x, lane);
  }
  x_ = x;
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void max_over_cg(T &x_) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  const int myid = __cooperative_group__.thread_rank()+1;
  T x = x_;
  for(int lane=1; lane<tile_size; lane<<=1) {
    x = Max(x, __cooperative_group__.shfl_xor(x, lane));
  }
  x_ = x;
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void min_over_cg(T &x_) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  const int myid = __cooperative_group__.thread_rank()+1;
  T x = x_;
  for(int lane=1; lane<tile_size; lane<<=1) {
    x = Min(x, __cooperative_group__.shfl_xor(x, lane));
  }
  x_ = x;
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
T red2_over_cg(const T x_, const T y_, const int km) {
#if COOPERATIVE_GROUP_STANDARD
  auto const __cooperative_group__ = 
    cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
  cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
  const int myid = __cooperative_group__.thread_rank()+1;
  T x = x_; T y = y_;
  const bool eee = ((myid & 0x1) != 0);
  __Cond_swap__(x, y, eee);
  x = y + __cooperative_group__.shfl_xor(x, 1);
  for(int lane=2; lane<tile_size; lane<<=1) {
    x += __cooperative_group__.shfl_xor(x, lane);
  }
  x = __MASK__( x, myid<=2 );
  x = __cooperative_group__.shfl(x, km);
  return x;
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void sum2_over_cg(T &x_, T &y_) {
  if (tile_size<=1) {
    sum_over_cg<T,tile_size>(x_);
    sum_over_cg<T,tile_size>(y_);
  } else {
#if COOPERATIVE_GROUP_STANDARD
    auto const __cooperative_group__ = 
      cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
    cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
    const int myid = __cooperative_group__.thread_rank()+1;
    T x = x_; T y = y_;
    const bool eee = ((myid & 0x1) != 0);
    __Cond_swap__(x, y, eee);
    x = y + __cooperative_group__.shfl_xor(x, 1);
    for(int lane=2; lane<tile_size; lane<<=1) {
      x += __cooperative_group__.shfl_xor(x, lane);
    }
    y = __cooperative_group__.shfl_xor(x, 1);
    __Cond_swap__(x, y, !eee);
    x_ = x; y_ = y;
  }
}

template < class T = int, int tile_size = WARP_GPU_SIZE >
__device__ __forceinline__
void sum4_over_cg(T &x_, T &y_, T &z_, T &w_) {
  if (tile_size<=2) {
    sum2_over_cg<T,tile_size>(x_,y_);
    sum2_over_cg<T,tile_size>(z_,w_);
  } else {
#if COOPERATIVE_GROUP_STANDARD
    auto const __cooperative_group__ = 
      cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block());
#else
    cooperative_groups::thread_block_tile_base<tile_size> __cooperative_group__;
#endif
    const int myid = __cooperative_group__.thread_rank()+1;
    T x = x_; T y = y_; T z = z_; T w = w_;
    const bool eee = ((myid & 0x1) != 0);
    __Cond_swap__(x, y, eee);
    x = y + __cooperative_group__.shfl_xor(x, 1);
    __Cond_swap__(z, w, eee);
    z = w + __cooperative_group__.shfl_xor(z, 1);
    const bool fff = ((myid & 0x2) != 0);
    __Cond_swap__(x, z, fff);
    x = z + __cooperative_group__.shfl_xor(x, 2);
    for(int lane=4; lane<tile_size; lane<<=1) {
      x += __cooperative_group__.shfl_xor(x, lane);
    }
    z = __cooperative_group__.shfl_xor(x, 2);
    __Cond_swap__(x, z, !fff);
    y = __cooperative_group__.shfl_xor(x, 1);
    __Cond_swap__(x, y, !eee);
    w = __cooperative_group__.shfl_xor(z, 1);
    __Cond_swap__(w, z, !eee);
    x_ = x; y_ = y; z_ = z; w_ = w;
  }
}


__device__ __forceinline__ void
sync_over_warp(void)
{
#if defined(__NVCC__)
  __syncwarp();
#endif
#if defined(__HIPCC__)
  int constexpr tile_size = WARP_GPU_SIZE;
#if COOPERATIVE_GROUP_STANDARD
  cooperative_groups::tiled_partition<tile_size>(cooperative_groups::this_thread_block()).sync();
#else
  cooperative_groups::thread_block_tile_base<tile_size>::sync();
#endif
#endif
  
}
#if defined(__NVCC__)
#define SYNC_IF_NEEDED(...) _if_(tile_size<WARP_GPU_SIZE)sync_over_warp()
#else
#define SYNC_IF_NEEDED(...) /* */
#endif

template <class T>
__device__ __forceinline__ void
bcast_over_warp(T &x_, const int root)
{
#if defined(__NVCC__)
  T x = x_;
  x = __shfl_sync( 0xffffffff, x, root-1, WARP_GPU_SIZE );
  x_ = x;
#endif
#if defined(__HIPCC__)
  T x = x_;
  x = __shfl( x, root-1, WARP_GPU_SIZE );
  x_ = x;
#endif
}

#if defined(__NVCC__)
template <>
__device__ __forceinline__ void
bcast_over_warp(float &x_, const int root)
{
  float x = x_;
  asm volatile ( 
  "{.reg.pred\t%pp;\n\t"
  "shfl.sync.idx.b32\t%0|%pp, %0, %1, 31, -1;\n\t"
  "}"
  : "+f"(x) : "r"(root-1) );
  x_ = x;
}
#endif

template <class T>
__device__ __forceinline__ void
bcast2_over_warp(T &x_, T &y_, const int root)
{
  bcast_over_warp(x_, root);
  bcast_over_warp(y_, root);
}

#if defined(__NVCC__)
template <>
__device__ __forceinline__ void
bcast2_over_warp(float &x_, float &y_, const int root)
{
  float x = x_;
  float y = y_;
  asm volatile ( 
  "{.reg.pred\t%pp;\n\t"
  "shfl.sync.idx.b32\t%0|%pp, %0, %2, 31, -1;\n\t"
  "shfl.sync.idx.b32\t%1|%pp, %1, %2, 31, -1;\n\t"
  "}"
  : "+f"(x), "+f"(y) : "r"(root-1) );
  x_ = x;
  y_ = y;
}
#endif

template <class T>
__device__ __inline__ void
sum_over_warp(T &x_)
{
#if defined(__CUDA_ARCH__)
  T x = x_;
  x += __shfl_xor_sync( 0xffffffff, x, 1, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 2, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 4, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 8, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 16, 32 );
  x_ = x;
#endif
#if defined(__HIPCC__)
  T x = x_;
  x += __shfl_xor( x, 1, WARP_GPU_SIZE );
  x += __shfl_xor( x, 2, WARP_GPU_SIZE );
  x += __shfl_xor( x, 4, WARP_GPU_SIZE );
  x += __shfl_xor( x, 8, WARP_GPU_SIZE );
  x += __shfl_xor( x, 16, WARP_GPU_SIZE );
#if WARP_GPU_SIZE == 64
  x += __shfl_xor( x, 32, 64 );
#endif
  x_ = x;
#endif

}

#if defined(__NVCC__)
template <>
__device__ __inline__ void
sum_over_warp(float &x_)
{
  float x = x_;
  asm volatile (
  "{.reg.f32\t%ftemp;\n\t"
  ".reg.pred\t%pp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  1, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  2, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  4, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  8, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0, 16, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "}"
  : "+f"(x) );
  x_ = x;
}
#endif

template <class T>
__device__ __inline__ void
sum2_over_warp(T &x_, T &y_)
{
  T x = x_;
  T y = y_;

  const bool eee = (threadIdx.x & 0x1);
  {
    const T t = (eee ? x : y);
    y = (eee ? y : x);
    x = t;
  }
#if defined(__CUDA_ARCH__)
  x = y + __shfl_xor_sync( 0xffffffff, x, 1, 32 );

  x += __shfl_xor_sync( 0xffffffff, x, 2, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 4, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 8, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 16, 32 );

  y = __shfl_xor_sync( 0xffffffff, x, 1, 32 );
#else
  x = y + __shfl_xor( x, 1 , WARP_GPU_SIZE );
  x +=    __shfl_xor( x, 2 , WARP_GPU_SIZE );
  x +=    __shfl_xor( x, 4 , WARP_GPU_SIZE );
  x +=    __shfl_xor( x, 8 , WARP_GPU_SIZE );
  x +=    __shfl_xor( x, 16, WARP_GPU_SIZE );
#if WARP_GPU_SIZE == 64
  x +=    __shfl_xor( x, 32, 64 );
#endif
  y = __shfl_xor( x, 1, WARP_GPU_SIZE );
#endif
  
  {
    const T t = (eee ? y : x);
    y = (eee ? x : y);
    x = t;
  }

  x_ = x;
  y_ = y;
}

#if defined(__NVCC__)
template <>
__device__ __inline__ void
sum2_over_warp(float &x_, float &y_)
{
  float x = x_;
  float y = y_;

  const bool eee = (threadIdx.x & 0x1);
  __Cond_swap__(x, y, eee);
  asm volatile (
  "{.reg.f32\t%ftemp;\n\t"
  ".reg.pred\t%pp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  1, 31, -1;\n\t"
  "add.f32\t%0, %1, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  2, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  4, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  8, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%ftemp|%pp, %0,  16, 31, -1;\n\t"
  "add.f32\t%0, %0, %ftemp;\n\t"
  "shfl.sync.bfly.b32\t%1|%pp, %0,  1, 31, -1;\n\t"
  "}"
  : "+f"(x), "+f"(y) );
#if 0
  z = w + __shfl_xor_sync( 0xffffffff, z, 1, 32 );
  z += __shfl_xor_sync( 0xffffffff, z, 2, 32 );
  z += __shfl_xor_sync( 0xffffffff, z, 4, 32 );
  z += __shfl_xor_sync( 0xffffffff, z, 8, 32 );
  z += __shfl_xor_sync( 0xffffffff, z, 16, 32 );
  w = __shfl_xor_sync( 0xffffffff, z, 1, 32 );
#endif
  __Cond_swap__(x, y, !eee);

  x_ = x;
  y_ = y;
}
#endif

template <class T>
__device__ __inline__ void
sum4_over_warp(T &x_, T &y_, T &z_, T &w_)
{
  T x = x_;
  T y = y_;
  T z = z_;
  T w = w_;

#if defined(__CUDA_ARCH__)
  const bool eee = (threadIdx.x & 0x1);
  __Cond_swap__(x, y, eee);
  x = y + __shfl_xor_sync( 0xffffffff, x, 1, 32 );
  __Cond_swap__(z, w, eee);
  z = w + __shfl_xor_sync( 0xffffffff, z, 1, 32 );

  const bool fff = (threadIdx.x & 0x2);
  __Cond_swap__(x, z, fff);
  x = z + __shfl_xor_sync( 0xffffffff, x, 2, 32 );

  x += __shfl_xor_sync( 0xffffffff, x, 4, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 8, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 16, 32 );

  z = __shfl_xor_sync( 0xffffffff, x, 2, 32 );
  __Cond_swap__(x, z, !fff);

  y = __shfl_xor_sync( 0xffffffff, x, 1, 32 );
  __Cond_swap__(x, y, !eee);
  w = __shfl_xor_sync( 0xffffffff, z, 1, 32 );
  __Cond_swap__(z, w, !eee);
#endif
#if defined(__HIPCC__)
  const bool eee = (threadIdx.x & 0x1);
  __Cond_swap__(x, y, eee);
  x = y + __shfl_xor( x, 1, WARP_GPU_SIZE );
  __Cond_swap__(z, w, eee);
  z = w + __shfl_xor( z, 1, WARP_GPU_SIZE );

  const bool fff = (threadIdx.x & 0x2);
  __Cond_swap__(x, z, fff);
  x = z + __shfl_xor( x, 2, WARP_GPU_SIZE );

  x += __shfl_xor( x, 4, WARP_GPU_SIZE );
  x += __shfl_xor( x, 8, WARP_GPU_SIZE );
  x += __shfl_xor( x, 16, WARP_GPU_SIZE );
#if WARP_GPU_SIZE == 64
  x += __shfl_xor( x, 32, 64 );
#endif

  z = __shfl_xor( x, 2, WARP_GPU_SIZE );
  __Cond_swap__(x, z, !fff);

  y = __shfl_xor( x, 1, WARP_GPU_SIZE );
  __Cond_swap__(x, y, !eee);
  w = __shfl_xor( z, 1, WARP_GPU_SIZE );
  __Cond_swap__(z, w, !eee);
  
#endif
  x_ = x;
  y_ = y;
  z_ = z;
  w_ = w;
}

#if defined(__NVCC__)
template <class T>
__device__ __inline__ void
max_over_warp(T &x_)
{
  T x = x_;
  x = Max(x, __shfl_xor_sync( 0xffffffff, x, 1, 32 ));
  x = Max(x, __shfl_xor_sync( 0xffffffff, x, 2, 32 ));
  x = Max(x, __shfl_xor_sync( 0xffffffff, x, 4, 32 ));
  x = Max(x, __shfl_xor_sync( 0xffffffff, x, 8, 32 ));
  x = Max(x, __shfl_xor_sync( 0xffffffff, x, 16, 32 ));
  x_ = x;
}

template <class T>
__device__ __inline__ void
min_over_warp(T &x_)
{
  T x = x_;
  x = Min(x, __shfl_xor_sync( 0xffffffff, x, 1, 32 ));
  x = Min(x, __shfl_xor_sync( 0xffffffff, x, 2, 32 ));
  x = Min(x, __shfl_xor_sync( 0xffffffff, x, 4, 32 ));
  x = Min(x, __shfl_xor_sync( 0xffffffff, x, 8, 32 ));
  x = Min(x, __shfl_xor_sync( 0xffffffff, x, 16, 32 ));
  x_ = x;
}
#endif

// only neccessary for _small kernel
#if defined(__NVCC__)
template <class T>
__device__ __inline__ void
sum_over_warp(T &x_, const int bit)
{
  T x = x_;
  const int nlz=1<<bit;
  #pragma unroll
  for(int lane=1; lane<nlz; lane<<=1) {
    x += __shfl_xor_sync( 0xffffffff, x, lane, nlz );
  }
  x_ = x;
}
#endif

template <class T>
__device__ __inline__ void
red2_over_warp(T &x_, const T y_)
{
  T x = x_;
  T y = y_;

  const bool eee = (threadIdx.x & 0x1);
  {
    const T t = (eee ? x : y);
    y = (eee ? y : x);
    x = t;
  }

#if defined(__CUDA_ARCH__)
  x = y + __shfl_xor_sync( 0xffffffff, x, 1, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 2, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 4, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 8, 32 );
  x += __shfl_xor_sync( 0xffffffff, x, 16, 32 );
#endif
#if defined(__HIPCC__)
  x = y + __shfl_xor( x,  1, WARP_GPU_SIZE );
  x +=    __shfl_xor( x,  2, WARP_GPU_SIZE );
  x +=    __shfl_xor( x,  4, WARP_GPU_SIZE );
  x +=    __shfl_xor( x,  8, WARP_GPU_SIZE );
  x +=    __shfl_xor( x, 16, WARP_GPU_SIZE );
#if WARP_GPU_SIZE == 64
  x +=    __shfl_xor( x, 32, WARP_GPU_SIZE );
#endif
#endif
  x_ = x;
}

#if defined(__NVCC__)

template <class T>
__device__ __inline__ void
find_maxloc_over_warp(T x_, int &pos)
{
  T x = Abs(x_); int p = threadIdx.x & 0x1f;
  for(int lane=1; lane<32; lane<<=1) {
    T y = __shfl_xor_sync( 0xffffffff, x, lane, 32 );
    if ( y > x ) { x = y; p ^= lane; }
  }
}


template <class T>
__device__ __noinline__ void
matcpy_(const int nm, const int n, T * wk_, T *a_)
{
  const int myid = threadIdx.x % 32 + 1;
  __syncwarp();
#define	a(row,col)	(*(a_+(row-1)+(col-1)*nm))
#define	wk(row,col)	(*(wk_+(row-1)+(col-1)*nm))

  for (int i=1; i<=n%4; i++) {
    T * wki_ptr = &wk(myid,i);
    T * aki_ptr = &a(myid,i);
    for (int k=myid; k<=nm; k+=32, wki_ptr+=32, aki_ptr+=32) {
      *wki_ptr = *aki_ptr;
    }
  }
  for (int i=1+n%4; i<=n; i+=4) {
    T * wki_ptr = &wk(myid,i);
    T * aki_ptr = &a(myid,i);
    for (int k=myid; k<=nm; k+=32, wki_ptr+=32, aki_ptr+=32) {
      for(int I=0; I<4; I++) wki_ptr[I*nm] = aki_ptr[I*nm];
    }
  }

#undef	a
#undef	wk
  __syncwarp();
}
#endif

#endif

