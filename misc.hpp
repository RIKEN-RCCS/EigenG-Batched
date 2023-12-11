#ifndef __HEADER_MISC_HPP__
#define __HEADER_MISC_HPP__

#include <time.h>

#if !defined(__HIPCC__)
int min(const int a, const int b) { return (a <= b ? a : b); }
int max(const int a, const int b) { return (a >= b ? a : b); }
#endif

#include "gpu_arch.h"

void
print_header(const char *func_name, int argc, char **argv)
{
  //char	t[1024];
  gpuDeviceProp deviceProp;
  int  id = 0;
  gpuGetDeviceProperties ( &deviceProp, id );
  int driver_Version;
  gpuDriverGetVersion( &driver_Version );
  int runtime_Version;
  gpuRuntimeGetVersion( &runtime_Version );

  printf("<!--/****************************************\n");

  printf("Evaluating the performance on %s\n", func_name==NULL?" ":func_name);
  {
    printf("\n %% ");
    for(int i=0;i<argc;i++) { printf("%s ",argv[i]); }
    printf("\n\n");
  }
  printf("Date : ");
  fflush(stdout);
  system("date");
  fflush(stdout);
  printf("Host : ");
  fflush(stdout);
  system("hostname");
  fflush(stdout);
  printf("Device Name : %s\n", deviceProp.name);
  fflush(stdout);

#if defined(__HIPCC__)
  printf("ROCm Driver : %d.%d\n",
         (driver_Version/1000),
         (driver_Version%100)/10);
//  system("head -1 /proc/driver/nvidia/version");
#endif

#if defined(__NVCC__)
  printf("CUDA Driver : %d.%d\n",
         (driver_Version/1000),
         (driver_Version%100)/10);
  system("head -1 /proc/driver/nvidia/version");
  fflush(stdout);
  printf("Runtime Driver : %d.%d\n",
         (runtime_Version/1000),
         (runtime_Version%100)/10);
  fflush(stdout);
  if ( runtime_Version > driver_Version ) {
    printf("******** CAUTION ********\n");
    printf("The installed video-driver is too old for the runtime.\n");
    fflush(stdout);
  }
#endif

  printf("*****************************************/-->\n");
  fflush(stdout);
}

__host__ double
get_wtime(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (ts.tv_sec+ts.tv_nsec*1e-9);
}

#endif

