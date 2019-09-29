#include <stdio.h>
#include <cooperative_groups.h>
#include<math.h>
#include"repeat.h"
namespace cg = cooperative_groups;



__global__ void reduce_basic(double *g_idata, double *g_odata, unsigned int n)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;	
    // unsigned int warp_id=tid/32;
   	unsigned int i=blockIdx.x*blockDim.x*2 + threadIdx.x;	
    unsigned int gridSize = blockDim.x*gridDim.x*2;

    // unsigned int  start,stop;

    double sum=0;

    while (i < n)
    {
    	sum+=g_idata[i];
      sum+=g_idata[i+blockDim.x];
    	i += gridSize;
    }
    g_odata[tid]=sum;
}


__global__ void reduce_basic_warp(double *g_idata, double *g_odata, unsigned int n, unsigned int basicthread, unsigned int *time_stamp)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    unsigned int warp_id=tid/32;
    unsigned int i=tid; 
    unsigned int gridSize = basicthread*2;

    unsigned int  start,stop;
    
    double sum=0;

    if(tid<basicthread)
    {
      asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
      while (i < n)
      {
        sum+=g_idata[i];
        sum+=g_idata[i+basicthread];
        i += gridSize;
      }
      asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
      
    }
    g_odata[tid]=sum;
    if(i%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
}

__global__ void reduce_basic_warp_sm(double *g_idata, double *g_odata, unsigned int n, unsigned int basicthread, unsigned int *time_stamp)
{
    cg::thread_block cta = cg::this_thread_block();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    unsigned int warp_id=tid/32;
    unsigned int i=tid; 
    unsigned int gridSize = basicthread*2;

    unsigned int  start,stop;
    double __shared__ sm[1024];
    sm[tid]=g_idata[tid];
    double sum=0;

    if(tid<basicthread)
    {
      asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
      while (i < n)
      {
        sum+=sm[(i%1024)];
        sum+=sm[(i+basicthread)%1024];
        i += gridSize;
      }
      asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
      
    }
    g_odata[tid]=sum;
    if(i%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
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

void single_test(float& millisecond,
                unsigned int size,
                unsigned int blockPerSM,
                unsigned int threadPerBlock,
                unsigned int SMPerGPU)
{
  double* h_input;
  double* d_input;
  double* h_output;
  double* d_output;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaHostAlloc((void**)& h_input, size*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)& h_output, size*sizeof(double), cudaHostAllocDefault);
  cudaMalloc((void**)&d_input, size*sizeof(double));
  cudaMalloc((void**)&d_output, size*sizeof(double));

  for(int i=0; i<size; i++)
  {
    h_input[i]=1;
  } 

  cudaMemcpy(d_input, h_input, size*sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  reduce_basic<<<blockPerSM*SMPerGPU ,threadPerBlock>>>(d_input,d_output,size);
  cudaEventRecord(end);
  cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) 
    { 
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
    }
  cudaEventElapsedTime(&millisecond,start,end);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaDeviceReset();
}

double group_basic(
                unsigned int repeat,
                unsigned int skip,
                unsigned int size,
                unsigned int blockPerSM,
                unsigned int threadPerBlock,
                unsigned int SMPerGPU)
{
    float* milliseconds = (float*)malloc(sizeof(float)*repeat);
    for(int i=0; i<repeat; i++)
    {
      single_test(milliseconds[i],
                  size,
                  blockPerSM,
                  threadPerBlock,
                  SMPerGPU);
    }
    double result=0;
    for(int i=skip; i<repeat; i++)
    {
      result+=milliseconds[i];
    }
    result=result/(repeat-skip);
    free(milliseconds);
    return result;
}

void single_warp_test(unsigned int & latency_cycle,
                unsigned int size,
                unsigned int threadPerBlock,
                unsigned int basicthread)
{
  unsigned int blockPerSM=1;
  unsigned int SMPerGPU=1;

  double* h_input;
  double* d_input;
  double* h_output;
  double* d_output;
  unsigned int* d_time_stamp;
  unsigned int* h_time_stamp=(unsigned int*)malloc(sizeof(unsigned int)*threadPerBlock*2/32);

  cudaHostAlloc((void**)& h_input, size*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)& h_output, size*sizeof(double), cudaHostAllocDefault);
  cudaMalloc((void**)&d_input, size*sizeof(double));
  cudaMalloc((void**)&d_output, size*sizeof(double));
  cudaMalloc((void**)&d_time_stamp, threadPerBlock*2/32*sizeof(unsigned int));
  
  for(int i=0; i<size; i++)
  {
    h_input[i]=1;
  } 

  cudaMemcpy(d_input, h_input, size*sizeof(double), cudaMemcpyHostToDevice);

  reduce_basic_warp_sm<<<blockPerSM*SMPerGPU ,threadPerBlock>>>(d_input,d_output,size,basicthread,d_time_stamp);

  cudaMemcpy(h_time_stamp, d_time_stamp, threadPerBlock*2/32*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize(); 

  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess) 
  { 
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
  }

  latency_cycle = h_time_stamp[1]-h_time_stamp[0];

  free(h_time_stamp);

  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  // cudaEventDestroy(start);
  // cudaEventDestroy(end);
  cudaDeviceReset();
}

double group_warp(
                unsigned int repeat,
                unsigned int skip,
                unsigned int size,
                unsigned int threadPerBlock,
                unsigned int basicthread)
{
    unsigned int* latency_cycles = (unsigned int*)malloc(sizeof(unsigned int)*repeat);

    for(int i=0; i<repeat; i++)
    {
      single_warp_test(latency_cycles[i],
                  size,
                  threadPerBlock,
                  basicthread);
    }

    double result=0;
    for(int i=skip; i<repeat; i++)
    {
      result+=latency_cycles[i];
    }
    result=result/(repeat-skip);
    free(latency_cycles);
    return result;
}
 // #define single_warp_test() \
 //    do{\
 //    double* h_input;\
 //    double* d_input;\
 //    double* h_output;\
 //    double* d_output;\
 //    unsigned int* d_time_stamp;\
 //    unsigned int* h_time_stamp=(unsigned int*)malloc(sizeof(unsigned int)*thread*2/32);\
 //      \
 //    cudaHostAlloc((void**)& h_input, smsize*sizeof(double), cudaHostAllocDefault);\
 //    cudaHostAlloc((void**)& h_output, smsize*sizeof(double), cudaHostAllocDefault);\
 //    cudaMalloc((void**)&d_input, smsize*sizeof(double));\
 //    cudaMalloc((void**)&d_output, smsize*sizeof(double));\
 //    cudaMalloc((void**)&d_time_stamp, thread*2/32*sizeof(unsigned int));\
 //  \
 //    for(int i=0; i<smsize; i++)\
 //    {\
 //      h_input[i]=1;\
 //    }\
 //    cudaMemcpy(d_input, h_input, smsize*sizeof(double), cudaMemcpyHostToDevice);\
 //    reduce_basic_warp<<<1,thread>>>(d_input,d_output,smsize,executor_num,d_time_stamp);\
 //    cudaMemcpy(h_time_stamp, d_time_stamp, thread*2/32*sizeof(unsigned int), cudaMemcpyDeviceToHost);\
 //    cudaDeviceSynchronize();\
 //    cudaError_t e=cudaGetLastError();\
 //    if(e!=cudaSuccess) \
 //      { \
 //          printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
 //      }\
 //      unsigned int start=h_time_stamp[0];\
 //      unsigned int end=h_time_stamp[1];\
 //      for(int i=1; i<thread/32; i++)\
 //      {\
 //        start=min(start,h_time_stamp[i*2]);\
 //        end=max(end,h_time_stamp[i*2+1]);\
 //      }\
 //      latency_cycle=end-start;\
 //      free(h_time_stamp);\
 //    cudaFreeHost(h_input);\
 //    cudaFree(d_time_stamp);\
 //    cudaFreeHost(h_output);\
 //    cudaFree(d_input);\
 //    cudaFree(d_output);\
 //    cudaDeviceReset();\
 //  }while(0);\

#define TEST_TIME 1
#define SKIP 0


#define single_size_test(func) \
{\
  unsigned int thread=1024;\
        for(int i=0; i<TEST_TIME; i++)\
        {\
           single_block_test(func);\
           lats[i]=millisecond;\
        }\
        millisecond=0;\
       for(int i=SKIP; i<TEST_TIME; i++)\
        {\
          millisecond+=lats[i];\
        }\
        millisecond=millisecond/(TEST_TIME-SKIP);\
        printf("func %s, block %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
          #func, block,thread, size,\
          millisecond, size*sizeof(double)/millisecond/1000/1000);\
}

int main()
{
	unsigned int size=500000000;
 	cudaDeviceProp deviceProp;
  	cudaSetDevice(0);
  	cudaGetDeviceProperties(&deviceProp, 0);
  	unsigned int smx_count = deviceProp.multiProcessorCount;

    unsigned int base=2048;
    

    // unsigned int block=1;
    // unsigned int thread=1024;
    unsigned int repeat=11;
    unsigned int skip=1;

    size=base*32;
    printf("test warp level bandwidth with size: %d*%lu \n", size,sizeof(double));
    printf("blck\tthrd\tltc(ccl)\tbdwdth(B/ccl)\n");
    for(int i=1; i<=32; i*=2)
    {
        double result = group_warp(repeat,skip,
                size,
                32,
                i);
        printf("%d\t%d\t%f\t%f\n",1,i,result,size*sizeof(double)/result);
    }    

    for(int i=32; i<=1024; i*=2)
    {
        double result = group_warp(repeat,skip,
                size,
                i,
                i);
        printf("%d\t%d\t%f\t%f\n",i,i,result,size*sizeof(double)/result);
    }    
    // size=base*1024;
    // printf("test block level bandwidth with size: %d*%lu \n", size,sizeof(double));
    // printf("blck\tthrd\tltc(ms)\tbdwdth(GB/s)\n");
    // for(int thread=32; thread<=1024; thread*=2)
    // {
    //     double result = group_basic(repeat,skip,size,1, thread,1);
    //     printf("%d\t%d\t%f\t%f\n",1,thread,result,(double)size*sizeof(double)/1000/1000/1000/(result/1000));      
    // }

    //  size=base*smx_count*1024*2;
    // printf("test multi block level bandwidth with size: %d*%lu \n", size,sizeof(double));
    // printf("blck\tthrd\tltc(ms)\tbdwdth(GB/s)\n");
    // for(int sm=1; sm<=smx_count; sm+=1)
    // {
    //     double result = group_basic(repeat,skip,size,1, 1024, sm);
    //     printf("%d\t%d\t%f\t%f\n",sm, 1024,result,(double)size*sizeof(double)/1000/1000/1000/(result/1000));     
    // }
    // double result = group_basic(repeat,skip,size,2, 1024, smx_count);
    // printf("%d\t%d\t%f\t%f\n",smx_count*2, 1024,result,(double)size*sizeof(double)/1000/1000/1000/(result/1000));   


    // single_test(millisecond,
    //             size,
    //             block,
    //             thread,
    //             1);

    // single_warp_test(latency_cycle,size,
    //             32,32);
    // double result = group_warp(11,
    //             1,
    //             size,
    //             32,
    //             32);
    // printf("%f\n",(double)result);
    // printf("%f\n",millisecond);
  
    }