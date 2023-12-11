#if defined(__NVCC__)

#include <cusolverDn.h>

void
cusolver_evd_test(int n, half *A, int lda, half *W, int batchSize)
{
}

void
cusolver_evd_test(int n, double *A, int lda, double *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);

  int lwork;
  cusolverDnDsyevd_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork
    );

  int *d_Info;
  double *d_work;
  cudaMalloc((void**)&d_Info, sizeof(int));
  cudaMalloc((void**)&d_work, sizeof(double)*lwork);

  for(int i=0; i<batchSize; i++) {
    cusolverDnDsyevd(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      n,
      A+(size_t)i*(n*lda),
      lda,
      W+(size_t)i*(n),
      d_work,
      lwork,
      d_Info
      );
  }

  cudaFree(d_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  cudaMemcpy(info,d_Info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(d_Info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);

  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 


void
cusolver_evd_test(int n, float *A, int lda, float *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);

  int lwork;
  cusolverDnSsyevd_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork
    );

  int *s_Info;
  float *s_work;
  cudaMalloc((void**)&s_Info, sizeof(int));
  cudaMalloc((void**)&s_work, sizeof(float)*lwork);

  for(int i=0; i<batchSize; i++) {
    cusolverDnSsyevd(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      n,
      A+(size_t)i*(n*lda),
      lda,
      W+(size_t)i*(n),
      s_work,
      lwork,
      s_Info
      );
  }

  cudaFree(s_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  cudaMemcpy(info,s_Info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(s_Info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);


  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

#endif

