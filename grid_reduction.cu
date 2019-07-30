/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/



#include <stdio.h>
#include <cooperative_groups.h>
#include "repeat.h"
namespace cg = cooperative_groups;


template <class T>
__device__ __forceinline__ T warp_reduce(T mySum, cg::thread_block_tile<32> group)
{
    mySum+=group.shfl_down(mySum,16);
    mySum+=group.shfl_down(mySum,8);
    mySum+=group.shfl_down(mySum,4);
    mySum+=group.shfl_down(mySum,2);
    mySum+=group.shfl_down(mySum,1);
    return mySum;
}


template <class T, unsigned int blockSize>
__device__ __forceinline__ T block_reduce_cuda_sample_opt(T mySum, T* sdata, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);    
    unsigned int tid=cta.thread_rank();
    if(blockSize >= 64)
    {
        sdata[tid] = mySum;
        cg::sync(cta);
    }
    // do reduction in shared mem
    if(blockSize >= 1024) 
    {
        if (tid < 512)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }
        cg::sync(cta);
    }

    if(blockSize >= 512) 
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }
        cg::sync(cta);
    }
    if(blockSize >= 256){
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }
        cg::sync(cta);
    }

    if(blockSize >= 128)
    {
        if ( tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }
        cg::sync(cta);
    }

    if (blockSize >=  64){
        if (tid < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            mySum=warp_reduce(mySum,tile32);
        }
    }
    if(blockSize==32)
    {
        mySum=warp_reduce(mySum,tile32);
    }
    return mySum;
}


template <class T, unsigned int blockSize>
__device__ __forceinline__ T block_reduce_warpserial(T mySum, T* sdata, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);    
    unsigned int tid=cta.thread_rank() ;

    if(blockSize>=64)
    {
        sdata[tid] = mySum;
        cg::sync(cta);
        if(tid<32)
        {
            if(blockSize>=1024){mySum+=sdata[tid+512];mySum+=sdata[tid+544];mySum+=sdata[tid+576];mySum+=sdata[tid+608];
                                mySum+=sdata[tid+640];mySum+=sdata[tid+672];mySum+=sdata[tid+704];mySum+=sdata[tid+736];
                                mySum+=sdata[tid+768];mySum+=sdata[tid+800];mySum+=sdata[tid+832];mySum+=sdata[tid+864];
                                mySum+=sdata[tid+896];mySum+=sdata[tid+928];mySum+=sdata[tid+960];mySum+=sdata[tid+992];}
            if(blockSize>=512){mySum+=sdata[tid+256];mySum+=sdata[tid+288];mySum+=sdata[tid+320];mySum+=sdata[tid+352];
                                mySum+=sdata[tid+384];mySum+=sdata[tid+416];mySum+=sdata[tid+448];mySum+=sdata[tid+480];}
            if(blockSize>=256){mySum+=sdata[tid+128];mySum+=sdata[tid+160];mySum+=sdata[tid+192];mySum+=sdata[tid+224];}
            if(blockSize>=128){mySum+=sdata[tid+64];mySum+=sdata[tid+96];}
            if(blockSize>=64)mySum+=sdata[tid+32];
            mySum=warp_reduce(mySum,tile32);
        }
    }
    if(blockSize==32)
    {
        mySum=warp_reduce(mySum,tile32);
    }

    return mySum;
}

template <class T, bool nIsPow2>
__device__ __forceinline__ T serial_reduce( unsigned int n, 
    unsigned int threadId, unsigned int blockId,
    unsigned int blockSize, unsigned int gridSizeMul2,
     T* idata)
/*
n is the size of array
blockSize is the size of a block 
*/
{
    unsigned int i=blockId*blockSize*2+threadId;
    T mySum=0;
    while (i < n)
    {
        mySum += idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += idata[i+blockSize];

        i += gridSizeMul2;
    }
    return mySum; 
}

template <class T, unsigned int BlockSize>
__device__ __forceinline__ T serial_reduce_final( unsigned int n, 
    unsigned int threadId, 
     T* idata)
/*
n is the size of array
blockSize is the size of a block 
*/
{
    unsigned int i=threadId;
    T mySum=0;
    while (i < n)
    {
        if(BlockSize<=128)
        {
            repeat16(mySum += idata[i];i+=BlockSize;);
        }
        if(BlockSize==256)
        {
            repeat8(mySum += idata[i];i+=BlockSize;);
        }
        if(BlockSize==512)
        {
            repeat4(mySum += idata[i];i+=BlockSize;);
        }
        if(BlockSize==1024)
        {
            repeat2(mySum += idata[i];i+=BlockSize;);
        }
    }
    return mySum; 
}
typedef double(*binary_reduction)(double, double*, cg::thread_block cta); 

template <binary_reduction block_reduction, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_grid(double *g_idata, double *g_odata, unsigned int n, bool useSM=true)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group gg = cg::this_grid();

    // double __shared__ sdata[blockSize];
    extern double  __shared__ sm[];
    double* sdata;
    if(useSM==true)
        sdata=sm;
    else
        sdata=g_odata;
    double  mySum = 0;
    if(n>2048)
    {
        mySum = serial_reduce<double,nIsPow2>(n, 
            threadIdx.x, blockIdx.x,
            blockDim.x, blockDim.x*gridDim.x*2,
                g_idata);
        g_odata[gg.thread_rank()] = mySum;
        cg::sync(gg);
    
        // write result for this block to global mem
        if(blockIdx.x==0)
        {
            mySum=0;
            mySum = serial_reduce_final<double,blockSize>(blockSize*gridDim.x, 
                threadIdx.x,
                g_odata);

            mySum=block_reduction(mySum, sdata, cta);
            if (threadIdx.x == 0) g_odata[blockIdx.x] = mySum;
        }
    }
    else
    {
        // use fewer threads is more profitable
        if(blockIdx.x==0)
        {
            mySum=0;
            serial_reduce<double,nIsPow2>(n, 
            threadIdx.x, 0,
            blockDim.x, blockDim.x*1*2,
                g_idata);
            mySum=block_reduction(mySum, sdata, cta);
            if (threadIdx.x == 0) g_odata[blockIdx.x] = mySum;
        }
    }
}


template <bool nIsPow2>
__global__ void
reduce_kernel1(double *g_idata, double *g_odata, unsigned int n)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // double mySum = g_idata[i];
    double  mySum = 0;

    mySum = serial_reduce<double,nIsPow2>(n, 
        threadIdx.x, blockIdx.x,
        blockDim.x, blockDim.x*gridDim.x*2,
            g_idata);
    g_odata[blockIdx.x*blockDim.x+threadIdx.x] = mySum;
}

template <binary_reduction block_reduction, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_kernel2(double *g_idata, double *g_odata, unsigned int n, bool useSM=true)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // double __shared__ sdata[blockSize];
    extern double  __shared__ sm[];
    double* sdata;
    if(useSM==true)
        sdata=sm;
    else
        sdata=g_odata;
    // load shared mem
    unsigned int tid = threadIdx.x;

    // double mySum = g_idata[i];
    double  mySum = 0;

    if(blockIdx.x==0)
    {
        mySum=0;
        mySum = serial_reduce_final<double,blockSize>(n, 
            tid,
            g_odata);

        mySum=block_reduction(mySum, sdata, cta);
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }

}
double cpu_reduce(double* array, unsigned int array_size)
{
    double sum=0;
    for(int i=0; i<array_size; i++)
    {
        sum+=array[i];
    }
    return sum;
}

template <binary_reduction block_reduction, unsigned int blockSize, bool nIsPow2,bool useSM>
void __forceinline__ launchKernelBasedReduction(double *g_idata, double *g_odata, unsigned int gridSize,  unsigned int n)
{
        reduce_kernel1<nIsPow2><<<gridSize,blockSize>>>(g_idata,g_odata,n); 
        if(useSM==true)
            reduce_kernel2<block_reduction,blockSize,false><<<1,blockSize,blockSize*sizeof(double)>>>(g_odata,g_odata,n,true); 
        else
            reduce_kernel2<block_reduction,blockSize,false><<<1,blockSize>>>(g_odata,g_odata,n,false); 

}
template <binary_reduction block_reduction, unsigned int blockSize, bool nIsPow2,bool useSM>
void __forceinline__ gridBasedReduction(double *g_idata, double *g_odata, unsigned int gridSize,  unsigned int n)
{
    bool l_useSM=useSM;
    void* KernelArgs[] = {(void**)&g_idata,(void**)&g_odata,(void*)&n,(void*)&l_useSM}; 

    if( useSM==true)
    {
        cudaLaunchCooperativeKernel((void*)reduce_grid<block_reduction,blockSize,nIsPow2>, gridSize,blockSize, KernelArgs,blockSize*sizeof(double),0);
    }   
    else
    {
        cudaLaunchCooperativeKernel((void*)reduce_grid<block_reduction,blockSize,nIsPow2>, gridSize,blockSize, KernelArgs,0,0);
    }
}

//        reduce_kernel1<true><<<2,BLOCKSIZE>>>(d_input,d_output,array_size); \
        reduce_kernel2<block_reduce_warpserial<double, BLOCKSIZE>,BLOCKSIZE,true><<<2,BLOCKSIZE,BLOCKSIZE*sizeof(double)>>>(d_output,d_output,array_size,false); \

//        strategy<block_reduce_kernel<double, BLOCKSIZE>,BLOCKSIZE,true,true>(d_input, d_output, GRIDSIZE,  array_size);\


#define single_test(BLOCKSIZE,GRIDSIZE, SIZE ,strategy, block_reduce_kernel,useSM) \
    do{\
        unsigned int array_size=SIZE;\
        double* h_output = (double*)malloc(sizeof(double)*array_size); \
        double* d_input; \
        double* d_output; \
        cudaEvent_t start, end;\
        cudaEventCreate(&start);\
        cudaEventCreate(&end);\
        cudaMalloc((void**)&d_input, sizeof(double)*array_size); \
        cudaMalloc((void**)&d_output, sizeof(double)*array_size); \
        cudaMemcpy(d_input, h_input, sizeof(double)*array_size, cudaMemcpyHostToDevice); \
        cudaEventRecord(start);\
        strategy<block_reduce_kernel<double, BLOCKSIZE>,BLOCKSIZE,true,useSM>(d_input, d_output, GRIDSIZE,  array_size);\
        cudaEventRecord(end);\
        \
        cudaMemcpy(h_output, d_output, sizeof(double)*array_size, cudaMemcpyDeviceToHost); \
        gpu_result=h_output[0];\
        cudaDeviceSynchronize(); \
        cudaError_t e=cudaGetLastError(); \
        if(e!=cudaSuccess) \
        { \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
        }\
        cudaEventElapsedTime(&millisecond,start,end);\
        free(h_output);\
        cudaFree(d_input);\
        cudaFree(d_output);\
        cudaEventDestroy(start);\
        cudaEventDestroy(end);\
    }while(0);

#define BASIC 1024
int main()
{
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int smx_count = deviceProp.multiProcessorCount;
    double gpu_result;

    float millisecond;
    // float lats[TEST_TIME];
    unsigned int thread_per_block=256;
    unsigned int block_per_sm=4;
    unsigned int data_per_thread=40960*1.2;
    unsigned long size = thread_per_block*smx_count*data_per_thread;
    double* h_input = (double*)malloc(sizeof(double)*size);
    for(int i=0; i<size; i++) 
    {
        h_input[i]=1;
    }
    double cpu_result=cpu_reduce(h_input,size);
    switch(thread_per_block)
    {
        case 32:
            single_test(32,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
        case 64:
            single_test(64,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
        case 128:
            single_test(128,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
        case 256:
            single_test(256,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
        case 512:
            single_test(512,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
        case 1024:
            single_test(1024,smx_count*block_per_sm,size,gridBasedReduction,block_reduce_warpserial,false);
            break;
    }
    
    printf("%f-%f=%f\n",cpu_result,gpu_result,cpu_result-gpu_result);   
    printf("block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
          1,thread_per_block, size,\
          (double)millisecond, (double)size*sizeof(double)/1000/1000/1000/(millisecond/1000));\

    switch(thread_per_block)
    {
        case 32:
            single_test(32,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
        case 64:
            single_test(64,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
        case 128:
            single_test(128,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
        case 256:
            single_test(256,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
        case 512:
            single_test(512,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
        case 1024:
            single_test(1024,smx_count*block_per_sm,size,launchKernelBasedReduction,block_reduce_warpserial,false);
            break;
    }
    printf("%f-%f=%f\n",cpu_result,gpu_result,cpu_result-gpu_result);
    printf("block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
          1,thread_per_block, size,\
          (double)millisecond, (double)size*sizeof(double)/1000/1000/1000/(millisecond/1000));\

    free(h_input);
//     single_test(32,reduce0);
//     single_test(64,reduce0);
//     single_test(128,reduce0);
//     single_test(256,reduce0);
//     single_test(512,reduce0);
//     single_test(1024,reduce0);


    // printf("funcname\tthread\tcorrect\tresult\tlatency\n");

    // repeat_test(32,reduce_opt,repeattime);
    // repeat_test(64,reduce_opt,repeattime);
    // repeat_test(128,reduce_opt,repeattime);
    // repeat_test(256,reduce_opt,repeattime);
    // repeat_test(512,reduce_opt,repeattime);
    // repeat_test(1024,reduce_opt,repeattime);


    // repeat_test(32,reduce_reduce_sync,repeattime);
    // repeat_test(64,reduce_reduce_sync,repeattime);
    // repeat_test(128,reduce_reduce_sync,repeattime);
    // repeat_test(256,reduce_reduce_sync,repeattime);
    // repeat_test(512,reduce_reduce_sync,repeattime);
    // repeat_test(1024,reduce_reduce_sync,repeattime);
 }

