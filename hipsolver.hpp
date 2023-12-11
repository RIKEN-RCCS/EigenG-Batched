#if defined(__HIPCC__)

#include <hipsolver/hipsolver.h>


#if 0
void
hipsolver_test(int n, half *A, int lda, half *W, int batchSize)
{
}
#endif

void
hipsolver_test(int n, double *A, int lda, double *W, int batchSize)
{
  hipsolverHandle_t hipsolverH = NULL;
  hipsolverCreate(&hipsolverH);
  hipStream_t stream = NULL; 
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipsolverSetStream(hipsolverH, stream);
  hipsolverSyevjInfo_t syevj_params = NULL;
  hipsolverDnCreateSyevjInfo(&syevj_params);
  double constexpr EPS = (double)std::numeric_limits<double>::epsilon();
  hipsolverDnXsyevjSetTolerance(syevj_params,EPS*512);
  hipsolverDnXsyevjSetMaxSweeps(syevj_params,10);
  hipsolverDnXsyevjSetSortEig(syevj_params,0);

  int lwork;
  hipsolverDsyevjBatched_bufferSize(
    hipsolverH,
    HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
    );

  double *d_work;
  hipMalloc((void**)&d_work, sizeof(double)*lwork);
  int *d_info;
  hipMalloc((void**)&d_info, sizeof(int)*batchSize);

  hipsolverDsyevjBatched(
    hipsolverH,
    HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_UPPER,
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

  hipFree(d_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  hipMemcpy(info,d_info,sizeof(int)*batchSize,hipMemcpyDeviceToHost);
  hipFree(d_info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);

  hipsolverDnDestroySyevjInfo(syevj_params);
  hipsolverDnDestroy(hipsolverH);
  hipStreamDestroy(stream);
} 

void
hipsolver_test(int n, float *A, int lda, float *W, int batchSize)
{
  hipsolverHandle_t hipsolverH = NULL;
  hipsolverCreate(&hipsolverH);
  hipStream_t stream = NULL; 
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  hipsolverSetStream(hipsolverH, stream);
  hipsolverSyevjInfo_t syevj_params = NULL;
  hipsolverDnCreateSyevjInfo(&syevj_params);
  double constexpr EPS = (double)std::numeric_limits<float>::epsilon();
  hipsolverDnXsyevjSetTolerance(syevj_params,EPS*16);
  hipsolverDnXsyevjSetMaxSweeps(syevj_params,10);
  hipsolverDnXsyevjSetSortEig(syevj_params,0);

  int lwork;
  hipsolverSsyevjBatched_bufferSize(
    hipsolverH,
    HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_UPPER,
    n,
    A,
    lda,
    W,
    &lwork,
    syevj_params,
    batchSize
    );

  float *s_work;
  hipMalloc((void**)&s_work, sizeof(float)*lwork);
  int *s_info;
  hipMalloc((void**)&s_info, sizeof(int)*batchSize);

  hipsolverSsyevjBatched(
    hipsolverH,
    HIPSOLVER_EIG_MODE_VECTOR,
    HIPSOLVER_FILL_MODE_UPPER,
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

  hipFree(s_work);
  int * info = (int *)malloc(sizeof(int)*batchSize);
  hipMemcpy(info,s_info,sizeof(int)*batchSize,hipMemcpyDeviceToHost);
  hipFree(s_info);

  for(int i=0;i<batchSize;i++){
    if(info[i]!=0) {
      printf("[%06d] Gave up the iteration.\n",i); break;
    }
  }
  free(info);

  hipsolverDnDestroySyevjInfo(syevj_params);
  hipsolverDnDestroy(hipsolverH);
  hipStreamDestroy(stream);
} 

#endif

