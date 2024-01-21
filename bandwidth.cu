#include <stdio.h>
#include <cooperative_groups.h>
#include <math.h>
#include "repeat.h"
namespace cg = cooperative_groups;

__global__ void reduce_basic_warp(double *g_idata, double *g_odata,
                                  unsigned int n, unsigned int basicthread,
                                  unsigned int *time_stamp) {
  cg::thread_block cta = cg::this_thread_block();
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warp_id = tid / 32;
  unsigned int i = tid;
  unsigned int gridSize = basicthread * 2;
  unsigned int start, stop;

  double sum = 0;

  if (tid < basicthread) {
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
    while (i < n) {
      sum += g_idata[i % n];
      sum += g_idata[i + blockDim.x];
      i += gridSize;
    }
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
  }
  g_odata[tid] = sum;
  if (i % 32 == 0) {
    time_stamp[warp_id * 2] = start;
    time_stamp[warp_id * 2 + 1] = stop;
  }
}

void single_warp_test(unsigned int &latency_cycle, unsigned int size,
                      unsigned int threadPerBlock, unsigned int basicthread) {
  unsigned int blockPerSM = 1;
  unsigned int SMPerGPU = 1;

  double *h_input;
  double *d_input;
  double *h_output;
  double *d_output;
  unsigned int *d_time_stamp;
  unsigned int *h_time_stamp =
      (unsigned int *)malloc(sizeof(unsigned int) * threadPerBlock * 2 / 32);

  cudaHostAlloc((void **)&h_input, size * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void **)&h_output, size * sizeof(double),
                cudaHostAllocDefault);
  cudaMalloc((void **)&d_input, size * sizeof(double));
  cudaMalloc((void **)&d_output, size * sizeof(double));
  cudaMalloc((void **)&d_time_stamp,
             threadPerBlock * 2 / 32 * sizeof(unsigned int));

  for (int i = 0; i < size; i++) {
    h_input[i] = 1;
  }

  cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);

  reduce_basic_warp << <blockPerSM *SMPerGPU, threadPerBlock>>>
      (d_input, d_output, size, basicthread, d_time_stamp);

  cudaMemcpy(h_time_stamp, d_time_stamp,
             threadPerBlock * 2 / 32 * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,
           cudaGetErrorString(e));
  }

  latency_cycle = h_time_stamp[1] - h_time_stamp[0];

  free(h_time_stamp);

  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaDeviceReset();
}

double group_warp(unsigned int repeat, unsigned int skip, unsigned int size,
                  unsigned int threadPerBlock, unsigned int basicthread) {
  unsigned int *latency_cycles =
      (unsigned int *)malloc(sizeof(unsigned int) * repeat);

  for (int i = 0; i < repeat; i++) {
    single_warp_test(latency_cycles[i], size, threadPerBlock, basicthread);
  }

  double result = 0;
  for (int i = skip; i < repeat; i++) {
    result += latency_cycles[i];
  }
  result = result / (repeat - skip);
  free(latency_cycles);
  return result;
}

int main() {

  cudaDeviceProp deviceProp;
  cudaSetDevice(0);
  cudaGetDeviceProperties(&deviceProp, 0);
  unsigned int smx_count = deviceProp.multiProcessorCount;
  unsigned int size = 1024 * smx_count * 512;

  unsigned int repeat = 11;
  unsigned int skip = 1;

  printf("test bandwidth per SM with size: %d*%lu \n", size, sizeof(double));
  printf("blck\tthrd\tltc(ccl)\tbdwdth(B/ccl)\n");
  for (int i = 32; i <= 1024; i *= 2) {
    double result = group_warp(repeat, skip, size, 32 , 1024 * smx_count);
    printf("%d\t%d\t%f\t%f\n", 1, i, result / (size / (1024 * smx_count)),
             size * sizeof(double) / result);
  }
}