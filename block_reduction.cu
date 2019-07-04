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
__device__ __forceinline__ T block_reduce_cuda_sample(T mySum, T* sdata, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);    
    unsigned int tid=cta.thread_rank() ;
    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 1024) && (tid < 512))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 512];
    }

    cg::sync(cta);


    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    cg::sync(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             mySum += tile32.shfl_down(mySum, offset);
        }
    }
    return mySum;
}


template <class T, unsigned int blockSize>
__device__ __forceinline__ T block_reduce_cuda_sample_opt(T mySum, T* sdata, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);    
    unsigned int tid=cta.thread_rank() ;
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
        if (cta.thread_rank() < 32)
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
__device__ __forceinline__ T block_reduce_tilebase(T mySum, T* sdata, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);    
    unsigned int tid=cta.thread_rank() ;
    unsigned int warp_id=tid/32;
    mySum=warp_reduce(mySum,tile32);

    if(blockSize>=64)
    {
        if(tile32.thread_rank()==0)
            sdata[warp_id]=mySum;
        cg::sync(cta);
        if(tid<32)
        {
            mySum=sdata[tid];
            if(blockSize>=1024) mySum+=tile32.shfl_down(mySum,16);
            if(blockSize>=512) mySum+=tile32.shfl_down(mySum,8);
            if(blockSize>=256) mySum+=tile32.shfl_down(mySum,4);
            if(blockSize>=128) mySum+=tile32.shfl_down(mySum,2);
            if(blockSize>=64) mySum+=tile32.shfl_down(mySum,1);
         }
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
            if(blockSize>=512){mySum+=sdata[tid+256];mySum+=sdata[tid+288];mySum+=sdata[tid+320];mySum+=sdata[tid+352];mySum+=sdata[tid+384];mySum+=sdata[tid+416];mySum+=sdata[tid+448];mySum+=sdata[tid+480];}
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

template <unsigned int blockSize>
__global__ void
reduce0(double *g_idata, double *g_odata,  unsigned int *time_stamp)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    double __shared__ sdata[blockSize];
    unsigned int  start,stop;

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    double mySum = g_idata[i];
    unsigned int warp_id=i/32;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    sdata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(i%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__global__ void
reduce_basic(double *g_idata, double *g_odata, unsigned int *time_stamp)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    double __shared__ sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int warp_id=tid/32;

    unsigned int  start,stop;
    double mySum = 0;

    mySum += g_idata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    mySum=block_reduce_cuda_sample<double,blockSize>(mySum,sdata,cta);
    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


template <unsigned int blockSize>
__global__ void
reduce_opt(double *g_idata, double *g_odata, unsigned int *time_stamp)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    double __shared__ sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int warp_id=tid/32;

    // unsigned int gridSize = blockSize*2*gridDim.x;
    unsigned int  start,stop;
    double mySum = 0;

    mySum += g_idata[tid];

    // each thread puts its local sum into shared memory

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    mySum=block_reduce_cuda_sample_opt<double,blockSize>(mySum,sdata,cta);
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}
template <unsigned int blockSize>
__global__ void
reduce_small_share(double *g_idata, double *g_odata, unsigned int *time_stamp)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    double __shared__ sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int warp_id=tid/32;

    unsigned int  start,stop;
    double mySum = 0;

    mySum += g_idata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    mySum=block_reduce_tilebase<double,blockSize>(mySum,sdata,cta);

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <unsigned int blockSize>
__global__ void
reduce_reduce_sync(double *g_idata, double *g_odata, unsigned int *time_stamp)
{
    cg::thread_block cta = cg::this_thread_block();
    double __shared__ sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int warp_id=tid/32;

    unsigned int  start,stop;
    double mySum = 0;

    mySum += g_idata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    mySum=block_reduce_warpserial<double,blockSize>(mySum,sdata,cta);
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


#include<cmath>


#define single_test(THREADSIZE,func) \
    do{double* h_input = (double*)malloc(sizeof(double)*THREADSIZE); \
    double* h_output = (double*)malloc(sizeof(double)*THREADSIZE); \
    unsigned int * h_time_stamp = (unsigned int*)malloc(sizeof(unsigned int)*THREADSIZE/32*2); \
    double* d_input; \
    double* d_output; \
    unsigned int* d_time_stamp; \
    cudaMalloc((void**)&d_input, sizeof(double)*THREADSIZE); \
    cudaMalloc((void**)&d_output, sizeof(double)*THREADSIZE); \
    cudaMalloc((void**)&d_time_stamp , sizeof(unsigned int)*THREADSIZE/32*2); \
    for(int i=0; i<THREADSIZE; i++) \
    { \
        h_input[i]=i; \
    } \
    cudaMemcpy(d_input, h_input, sizeof(double)*THREADSIZE, cudaMemcpyHostToDevice); \
    func<THREADSIZE><<<1,THREADSIZE>>>(d_input,d_output,d_time_stamp); \
    cudaMemcpy(h_output, d_output, sizeof(double)*THREADSIZE, cudaMemcpyDeviceToHost); \
 \
    cudaMemcpy(h_time_stamp, d_time_stamp, sizeof(unsigned int)*THREADSIZE/32*2, cudaMemcpyDeviceToHost); \
    cudaDeviceSynchronize(); \
  \
 \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) \
    { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
    }\
    unsigned int begin, end;\
    begin=h_time_stamp[0];\
    end=h_time_stamp[1];\
    for(int i=0; i<THREADSIZE/32; i++)\
    {\
        begin=min(begin,h_time_stamp[i*2]);\
        end=max(begin,h_time_stamp[i*2+1]);\
    }\
    printf("%s\t%d\t%f\t",\
        #func,THREADSIZE,h_output[0]);\
    printf("%d\n",\
        end-begin);\
    free(h_input);\
    free(h_output);\
    free(h_time_stamp);\
    cudaFree(d_input);\
    cudaFree(d_output);\
    cudaFree(d_time_stamp);}while(0)\


int main()
{

//     single_test(32,reduce0);
//     single_test(64,reduce0);
//     single_test(128,reduce0);
//     single_test(256,reduce0);
//     single_test(512,reduce0);
//     single_test(1024,reduce0);

    single_test(32,reduce_basic);
    single_test(64,reduce_basic);
    single_test(128,reduce_basic);
    single_test(256,reduce_basic);
    single_test(512,reduce_basic);
    single_test(1024,reduce_basic);

    single_test(32,reduce_opt);
    single_test(64,reduce_opt);
    single_test(128,reduce_opt);
    single_test(256,reduce_opt);
    single_test(512,reduce_opt);
    single_test(1024,reduce_opt);

    single_test(32,reduce_small_share);
    single_test(64,reduce_small_share);
    single_test(128,reduce_small_share);
    single_test(256,reduce_small_share);
    single_test(512,reduce_small_share);
    single_test(1024,reduce_small_share);

    single_test(32,reduce_reduce_sync);
    single_test(64,reduce_reduce_sync);
    single_test(128,reduce_reduce_sync);
    single_test(256,reduce_reduce_sync);
    single_test(512,reduce_reduce_sync);
    single_test(1024,reduce_reduce_sync);
 }

