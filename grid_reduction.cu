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

template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_opt(double *g_idata, double *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group gg = cg::this_grid();

    double __shared__ sdata[blockSize];

    // load shared mem
    unsigned int tid = threadIdx.x;

    // double mySum = g_idata[i];
    double  mySum = 0;
    // if(n>threashold)
    {
        mySum = serial_reduce<double,nIsPow2>(n, 
            tid, blockIdx.x,
            blockDim.x, blockDim.x*gridDim.x*2,
                g_idata);
        g_odata[blockIdx.x*blockDim.x+threadIdx.x] = mySum;
        cg::sync(gg);
    }
    // write result for this block to global mem
    if(blockIdx.x==0)
    {
        mySum=0;
        mySum = serial_reduce_final<double,blockSize>(blockSize*gridDim.x, 
            tid,
            g_odata);

        mySum=block_reduce_cuda_sample_opt<double,blockSize>(mySum, sdata, cta);
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
}


template <unsigned int blockSize>
__global__ void
reduce_reduce_sync(double *g_idata, double *g_odata, unsigned int *time_stamp)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group gg = cg::this_grid();
    // double __shared__ sdata[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int warp_id=tid/32;

    unsigned int  start,stop;
    double mySum = 0;

    mySum += g_idata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    mySum=block_reduce_warpserial<double,blockSize>(mySum,g_idata,cta);
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid%32==0)
    {
        time_stamp[warp_id*2]=start;
        time_stamp[warp_id*2+1]=stop;
    }
    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;

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

// void __forceinline__ cooperative_launch(nKernel func,
//     unsigned int blockPerGPU,unsigned int threadPerBlock, 
//     unsigned int GPU_count=1, cudaLaunchParams *launchParamsList=NULL)
// {
//     void* KernelArgs[] = {};
//     cudaLaunchCooperativeKernel((void*)func, blockPerGPU,threadPerBlock,KernelArgs,32,0);
// }

#define BASIC 1024
int main()
{
    do{
        unsigned int array_size=BASIC*56;
        double* h_input = (double*)malloc(sizeof(double)*array_size); 
        double* h_output = (double*)malloc(sizeof(double)*array_size); 
        double* d_input; 
        double* d_output; 
        cudaMalloc((void**)&d_input, sizeof(double)*array_size); 
        cudaMalloc((void**)&d_output, sizeof(double)*array_size); 
        for(int i=0; i<array_size; i++) 
        { 
            h_input[i]=i; 
        }
        void* KernelArgs[] = {(void**)&d_input,(void**)&d_output,(void*)&array_size}; 
        cudaMemcpy(d_input, h_input, sizeof(double)*array_size, cudaMemcpyHostToDevice); 

        // reduce_opt<BASIC,true><<<2,BASIC>>>(d_input,d_output,array_size); 
        
        cudaLaunchCooperativeKernel((void*)reduce_opt<BASIC,true>, 2,BASIC, KernelArgs,0,0);

        cudaMemcpy(h_output, d_output, sizeof(double)*array_size, cudaMemcpyDeviceToHost); 
        cudaDeviceSynchronize(); 
        double cpu_result=cpu_reduce(h_input,array_size);
        printf("%f:%f\n",cpu_result,h_output[0]);
        // for(int i=0; i<4; i++)
        // {
        //     printf("%f\t",h_output[i]);
        // }
        cudaError_t e=cudaGetLastError(); 
        if(e!=cudaSuccess) 
        { 
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
        }
        // result=h_output[0];
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }while(0);

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

