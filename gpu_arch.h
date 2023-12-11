#ifndef __HEADER_GPU_ARCH_H__
#define __HEADER_GPU_ARCH_H__

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_GPU_SIZE 32
//
// for data type renaming
//
typedef cudaStream_t		gpuStream_t;
typedef	struct cudaDeviceProp	gpuDeviceProp;
typedef	cudaError_t		gpuError_t;
//
// for status/error/constants
//
#define gpuSuccess		cudaSuccess
#define gpuErrorInvalidValue	cudaErrorInvalidValue
//
// for function renaming
//
#define gpuSetDevice		cudaSetDevice
#define gpuGetDevice		cudaGetDevice
#define gpuGetDeviceProperties	cudaGetDeviceProperties
#define gpuDriverGetVersion	cudaDriverGetVersion
#define gpuRuntimeGetVersion	cudaRuntimeGetVersion
#define gpuPointerAttributes	cudaPointerAttributes
#define gpuPointerGetAttributes	cudaPointerGetAttributes
#define gpuMalloc		cudaMalloc
#define gpuFree			cudaFree
#define gpuStreamCreate		cudaStreamCreate
#define gpuStreamDestroy	cudaStreamDestroy
#define gpuMemcpy		cudaMemcpy
#define gpuMemcpyDeviceToDevice	cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice	cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost	cudaMemcpyDeviceToHost
#define	gpuDeviceSynchronize	cudaDeviceSynchronize
#endif

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

#define WARP_GPU_SIZE 64
//
// for data type renaming
//
typedef hipStream_t		gpuStream_t;
typedef	hipDeviceProp_t		gpuDeviceProp;
typedef	hipError_t		gpuError_t;
//
// for status/error/constants
//
#define gpuSuccess		hipSuccess
#define gpuErrorInvalidValue	hipErrorInvalidValue
//
// for function renaming
//
#define gpuSetDevice		hipSetDevice
#define gpuGetDevice		hipGetDevice
#define gpuGetDeviceProperties	hipGetDeviceProperties
#define gpuDriverGetVersion	hipDriverGetVersion
#define gpuRuntimeGetVersion	hipRuntimeGetVersion
#define gpuPointerAttributes	hipPointerAttribute_t
#define gpuPointerGetAttributes	hipPointerGetAttributes
#define gpuMalloc		hipMalloc
#define gpuFree			hipFree
#define gpuStreamCreate		hipStreamCreate
#define gpuStreamDestroy	hipStreamDestroy
#define gpuMemcpy		hipMemcpy
#define gpuMemcpyDeviceToDevice	hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice	hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost	hipMemcpyDeviceToHost
#define	gpuDeviceSynchronize	hipDeviceSynchronize
#endif

#endif

