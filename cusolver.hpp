#if defined(__NVCC__)

#include <cusolverDn.h>

#if 0
void
cusolver_test(int n, half *A, int lda, half *W, int batchSize)
{
}
#endif

void
cusolver_test(int n, double *A, int lda, double *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  syevjInfo_t syevj_params = NULL;
  cusolverDnCreateSyevjInfo(&syevj_params);
  double constexpr EPS = (double)std::numeric_limits<double>::epsilon();
  cusolverDnXsyevjSetTolerance(syevj_params,EPS*512);
  cusolverDnXsyevjSetMaxSweeps(syevj_params,10);
  cusolverDnXsyevjSetSortEig(syevj_params,0);

  int lwork;
  cusolverDnDsyevjBatched_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
    );

  double *d_work;
  cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  int *d_info;
  cudaMalloc((void**)&d_info, sizeof(int)*batchSize);

  cusolverDnDsyevjBatched(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    d_work,
    lwork,
    d_info,
    syevj_params,
    batchSize
    );

  cudaFree(d_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  cudaMemcpy(info,d_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(d_info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);

  cusolverDnDestroySyevjInfo(syevj_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

void
cusolver_test(int n, float *A, int lda, float *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);
  syevjInfo_t syevj_params = NULL;
  cusolverDnCreateSyevjInfo(&syevj_params);
  double constexpr EPS = (double)std::numeric_limits<float>::epsilon();
  cusolverDnXsyevjSetTolerance(syevj_params,EPS*16);
  cusolverDnXsyevjSetMaxSweeps(syevj_params,10);
  cusolverDnXsyevjSetSortEig(syevj_params,0);

  int lwork;
  cusolverDnSsyevjBatched_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
    );

  float *s_work;
  cudaMalloc((void**)&s_work, sizeof(float)*lwork);
  int *s_info;
  cudaMalloc((void**)&s_info, sizeof(int)*batchSize);

  cusolverDnSsyevjBatched(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_LOWER,
    n,
    A,
    lda,
    W,
    s_work,
    lwork,
    s_info,
    syevj_params,
    batchSize
    );

  cudaFree(s_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  cudaMemcpy(info,s_info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(s_info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);

  cusolverDnDestroySyevjInfo(syevj_params);
  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

#endif

