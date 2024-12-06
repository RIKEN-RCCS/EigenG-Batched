#if defined(__NVCC__)

#include <cusolverDn.h>

#if CUSOLVER_VERSION > 11604

void
cusolver64_test(int n, double *A, int lda, double *W, int batchSize)
{
  cusolverDnHandle_t cusolverH;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  cusolverDnParams_t cusolver_params;
  cusolverDnCreateParams(&cusolver_params);

  cusolverStatus_t err;
  cudaError_t status;

  size_t lworkD;
  size_t lworkH;
  err = cusolverDnXsyevBatched_bufferSize(
    cusolverH,
    cusolver_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)n,
    CUDA_R_64F,
    A,
    (int64_t)lda,
    CUDA_R_64F,
    W,
    CUDA_R_64F,
    &lworkD,
    &lworkH,
    (int64_t)batchSize
    );

  if ( err != CUSOLVER_STATUS_SUCCESS ) {
    printf("buffer query fault\n"); return;
  }

  void *d_work;
  status = cudaMalloc((void**)&d_work, lworkD);
  if ( status != cudaSuccess || d_work == NULL ) {
    printf("work allocation fault\n"); return;
  }
  int *d_info;
  status = cudaMalloc((void**)&d_info, sizeof(int)*batchSize);
  if ( status != cudaSuccess || d_info == NULL ) {
    cudaFree(d_work);
    printf("devinfo allocation fault\n"); return;
  }
  void *h_work = (void *)malloc(lworkH);

  err = cusolverDnXsyevBatched(
    cusolverH,
    cusolver_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)n,
    CUDA_R_64F,
    A,
    (int64_t)lda,
    CUDA_R_64F,
    W,
    CUDA_R_64F,
    d_work,
    lworkD,
    h_work,
    lworkH,
    d_info,
    (int64_t)batchSize
    );

  cudaFree(d_work);
  free(h_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  status = cudaMemcpy(info,d_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(d_info);

  if ( err == CUSOLVER_STATUS_INVALID_VALUE ) {
    if ( info[0] < 0 ) {
        printf("[%06d]'s parameter was wrong.\n",-info[0]);
    }
  } else {
    for(int i=0;i<batchSize;i++){
      if(info[i]==n+1) {
        printf("[%06d] Gave up the iteration.\n",i); break;
      }
    }
  }
  free(info);

  cusolverDnDestroyParams(cusolver_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

void
cusolver64_test(int n, float *A, int lda, float *W, int batchSize)
{
  cusolverDnHandle_t cusolverH;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  cusolverDnParams_t cusolver_params;
  cusolverDnCreateParams(&cusolver_params);

  cusolverStatus_t err;
  cudaError_t status;

  size_t lworkD;
  size_t lworkH;
  err = cusolverDnXsyevBatched_bufferSize(
    cusolverH,
    cusolver_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)n,
    CUDA_R_32F,
    A,
    (int64_t)lda,
    CUDA_R_32F,
    W,
    CUDA_R_32F,
    &lworkD,
    &lworkH,
    (int64_t)batchSize
    );

  if ( err != CUSOLVER_STATUS_SUCCESS ) {
    printf("buffer query fault\n"); return;
  }

  void *s_work;
  status = cudaMalloc((void**)&s_work, lworkD);
  if ( status != cudaSuccess || s_work == NULL ) {
    printf("work allocation fault\n"); return;
  }
  int *s_info;
  status = cudaMalloc((void**)&s_info, sizeof(int)*batchSize);
  if ( status != cudaSuccess || s_info == NULL ) {
    cudaFree(s_work);
    printf("devinfo allocation fault\n"); return;
  }
  void *h_work = (void *)malloc(lworkH);

  err = cusolverDnXsyevBatched(
    cusolverH,
    cusolver_params,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    (int64_t)n,
    CUDA_R_32F,
    A,
    (int64_t)lda,
    CUDA_R_32F,
    W,
    CUDA_R_32F,
    s_work,
    lworkD,
    h_work,
    lworkH,
    s_info,
    (int64_t)batchSize
    );

  cudaFree(s_work);
  free(h_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  status = cudaMemcpy(info,s_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(s_info);

  if ( err == CUSOLVER_STATUS_INVALID_VALUE ) {
    if ( info[0] < 0 ) {
        printf("[%06d]'s parameter was wrong.\n",-info[0]);
    }
  } else {
    for(int i=0;i<batchSize;i++){
      if(info[i]==n+1) {
        printf("[%06d] Gave up the iteration.\n",i); break;
      }
    }
  }
  free(info);

  cusolverDnDestroyParams(cusolver_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

#endif
#endif

