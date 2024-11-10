#if defined(__NVCC__)

#include <cusolverDn.h>

#if 0
void
cusolver_evd_test(int n, half *A, int lda, half *W, int batchSize)
{
}
#endif

void
cusolver_evd_test(int n, double *A, int lda, double *W, int batchSize)
{
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnCreate(&cusolverH);
  cudaStream_t stream = NULL; 
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cusolverDnSetStream(cusolverH, stream);

  cusolverStatus_t err;
  cudaError_t status;

  int lwork;
  err = cusolverDnDsyevd_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork
    );

  if ( err != CUSOLVER_STATUS_SUCCESS ) { return; }

  int *d_Info;
  double *d_work;
  status = cudaMalloc((void**)&d_Info, sizeof(int)*batchSize);
  if ( status != cudaSuccess || d_Info == NULL ) { return; }
  status = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  if ( status != cudaSuccess || d_work == NULL ) { cudaFree(d_Info); return; }

  for(int i=0; i<batchSize; i++) {
    err = cusolverDnDsyevd(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      n,
      A+(size_t)i*(n*lda),
      lda,
      W+(size_t)i*(n),
      d_work,
      lwork,
      d_Info+i
      );
  }

  cudaFree(d_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  status = cudaMemcpy(info,d_Info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(d_Info);

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

  cusolverStatus_t err;
  cudaError_t status;

  int lwork;
  err = cusolverDnSsyevd_bufferSize(
    cusolverH,
    CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork
    );

  if ( err != CUSOLVER_STATUS_SUCCESS ) { return; }

  int *s_Info;
  float *s_work;
  status = cudaMalloc((void**)&s_Info, sizeof(int)*batchSize);
  if ( status != cudaSuccess || s_Info == NULL ) { return; }
  status = cudaMalloc((void**)&s_work, sizeof(float)*lwork);
  if ( status != cudaSuccess || s_work == NULL ) { cudaFree(s_Info); return; }

  for(int i=0; i<batchSize; i++) {
    err = cusolverDnSsyevd(
      cusolverH,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER,
      n,
      A+(size_t)i*(n*lda),
      lda,
      W+(size_t)i*(n),
      s_work,
      lwork,
      s_Info+i
      );
  }

  cudaFree(s_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  status = cudaMemcpy(info,s_Info,sizeof(int)*batchSize,cudaMemcpyDeviceToHost);
  cudaFree(s_Info);

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

  cusolverDnDestroy(cusolverH);
  cudaStreamDestroy(stream);
} 

#endif

