#include <stdio.h>
#include <cooperative_groups.h>
#include<math.h>
namespace cg = cooperative_groups;



__global__ void reduce_basic(double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;	
   	unsigned int i=blockIdx.x*blockDim.x*2 + threadIdx.x;	
    unsigned int gridSize = blockDim.x*gridDim.x*2;
    // printf("gridsize:%d\tblockdim:%d\tgridDim:%dn",gridSize, blockDim.x, gridDim.x);
    double sum=0;
    while (i < n)
    {
    	sum+=g_idata[i];
    	// if(i+blockDim.x<n)
    	sum+=g_idata[i+blockDim.x];
    	i += gridSize;
    }
    g_odata[tid]=sum;
}


__global__ void copy_basic(double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    unsigned int i=blockIdx.x*blockDim.x*2 + threadIdx.x; 
    unsigned int gridSize = blockDim.x*gridDim.x*2;
    // printf("gridsize:%d\tblockdim:%d\tgridDim:%dn",gridSize, blockDim.x, gridDim.x);
    double tmp1;
    double tmp2;
    while (i < n)
    {
      tmp1=g_idata[i];
      tmp2=g_idata[i+blockDim.x];
      g_odata[i]=tmp1;
      g_odata[tid]=tmp2;
      i += gridSize;
    }
}

 #define single_test(func) \
 do{\
	double* h_input;\
	double* d_input;\
	double* h_output;\
	double* d_output;\
	cudaEvent_t start, end;\
	cudaEventCreate(&start);\
	cudaEventCreate(&end);\
\
	cudaHostAlloc((void**)& h_input, size*sizeof(double), cudaHostAllocDefault);\
	cudaHostAlloc((void**)& h_output, size*sizeof(double), cudaHostAllocDefault);\
	cudaMalloc((void**)&d_input, size*sizeof(double));\
	cudaMalloc((void**)&d_output, size*sizeof(double));\
\
	for(int i=0; i<size; i++)\
	{\
		h_input[i]=1;\
	}\
	cudaMemcpy(d_input, h_input, size*sizeof(double), cudaMemcpyHostToDevice);\
	cudaEventRecord(start);\
	func<<<block*smx_count,thread>>>(d_input,d_output,size);\
	cudaEventRecord(end);\
	cudaDeviceSynchronize();\
	cudaError_t e=cudaGetLastError();\
	if(e!=cudaSuccess) \
    { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    }\
	cudaEventElapsedTime(&millisecond,start,end);\
	cudaFreeHost(h_input);\
	cudaFreeHost(h_output);\
	cudaFree(d_input);\
	cudaFree(d_output);\
	cudaEventDestroy(start);\
	cudaEventDestroy(end);\
	cudaDeviceReset();\
}while(0)\

 #define single_block_test(func) \
 do{\
  double* h_input;\
  double* d_input;\
  double* h_output;\
  double* d_output;\
  cudaEvent_t start, end;\
  cudaEventCreate(&start);\
  cudaEventCreate(&end);\
\
  cudaHostAlloc((void**)& h_input, size*sizeof(double), cudaHostAllocDefault);\
  cudaHostAlloc((void**)& h_output, size*sizeof(double), cudaHostAllocDefault);\
  cudaMalloc((void**)&d_input, size*sizeof(double));\
  cudaMalloc((void**)&d_output, size*sizeof(double));\
\
  for(int i=0; i<size; i++)\
  {\
    h_input[i]=1;\
  }\
  cudaMemcpy(d_input, h_input, size*sizeof(double), cudaMemcpyHostToDevice);\
  cudaEventRecord(start);\
  func<<<1,thread>>>(d_input,d_output,size);\
  cudaEventRecord(end);\
  cudaDeviceSynchronize();\
  cudaError_t e=cudaGetLastError();\
  if(e!=cudaSuccess) \
    { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    }\
  cudaEventElapsedTime(&millisecond,start,end);\
  cudaFreeHost(h_input);\
  cudaFreeHost(h_output);\
  cudaFree(d_input);\
  cudaFree(d_output);\
  cudaEventDestroy(start);\
  cudaEventDestroy(end);\
  cudaDeviceReset();\
}while(0)\

 #define single_warp_test() \
    do{\
    double* h_input;\
    double* d_input;\
    double* h_output;\
    double* d_output;\
    unsigned int* d_time_stamp;\
    unsigned int* h_time_stamp=(unsigned int*)malloc(sizeof(unsigned int)*thread*2/32);\
      \
    cudaHostAlloc((void**)& h_input, smsize*sizeof(double), cudaHostAllocDefault);\
    cudaHostAlloc((void**)& h_output, smsize*sizeof(double), cudaHostAllocDefault);\
    cudaMalloc((void**)&d_input, smsize*sizeof(double));\
    cudaMalloc((void**)&d_output, smsize*sizeof(double));\
    cudaMalloc((void**)&d_time_stamp, thread*2/32*sizeof(unsigned int));\
  \
    for(int i=0; i<smsize; i++)\
    {\
      h_input[i]=1;\
    }\
    cudaMemcpy(d_input, h_input, smsize*sizeof(double), cudaMemcpyHostToDevice);\
    reduce_basic_warp<<<1,thread>>>(d_input,d_output,smsize,executor_num,d_time_stamp);\
    cudaMemcpy(h_time_stamp, d_time_stamp, thread*2/32*sizeof(unsigned int), cudaMemcpyDeviceToHost);\
    cudaDeviceSynchronize();\
    cudaError_t e=cudaGetLastError();\
    if(e!=cudaSuccess) \
      { \
          printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
      }\
      unsigned int start=h_time_stamp[0];\
      unsigned int end=h_time_stamp[1];\
      for(int i=1; i<thread/32; i++)\
      {\
        start=min(start,h_time_stamp[i*2]);\
        end=max(end,h_time_stamp[i*2+1]);\
      }\
      latency_cycle=end-start;\
      free(h_time_stamp);\
    cudaFreeHost(h_input);\
    cudaFree(d_time_stamp);\
    cudaFreeHost(h_output);\
    cudaFree(d_input);\
    cudaFree(d_output);\
    cudaDeviceReset();\
  }while(0);\

#define TEST_TIME 21
#define SKIP 1

int main()
{
	unsigned int size=500000000;
 	cudaDeviceProp deviceProp;
  	cudaSetDevice(0);
  	cudaGetDeviceProperties(&deviceProp, 0);
  	unsigned int smx_count = deviceProp.multiProcessorCount;

    unsigned int base=2000;
    size=base*smx_count*2048*2;
    
    float millisecond;
    float lats[TEST_TIME];

  	
   //  unsigned int block=1;
  	// unsigned int thread = 1024;

    for(int block=1; block<=64; block*=2)
    { 
      for(int thread=32; thread<=1024; thread*=2)
      {
        for(int i=0; i<TEST_TIME; i++)
        {
           single_test(reduce_basic);
           lats[i]=millisecond;
        }
        millisecond=0;
       for(int i=SKIP; i<TEST_TIME; i++)
        {
          millisecond+=lats[i];
        }
        millisecond=millisecond/(TEST_TIME-SKIP);
        printf("block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",
          block,thread, size,
          millisecond, size*sizeof(double)/millisecond/1000/1000);
      }
    }

    size=size/smx_count*2;
      for(int thread=32; thread<=1024; thread*=2)
      {
        for(int i=0; i<TEST_TIME; i++)
        {
           single_block_test(reduce_basic);
           lats[i]=millisecond;
        }
        millisecond=0;
       for(int i=SKIP; i<TEST_TIME; i++)
        {
          millisecond+=lats[i];
        }
        millisecond=millisecond/(TEST_TIME-SKIP);
        printf("block/GPU %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",
          1,thread, size,
          millisecond, size*sizeof(double)/millisecond/1000/1000);
      }



// printf("thread %d, executer %d, smsize %d, time: %d cycle speed: %f GB/s\n",
//       thread, executor_num, smsize, latency, (double)smsize*sizeof(double)/latency);
    }