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


template <class T, unsigned int blockSize, bool nIsPow2, bool useSM, bool useWarpSerial>
__global__ void
reduce_grid(T *g_idata, T *g_odata, unsigned int n)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group gg = cg::this_grid();
    T (* block_reduction) (T, T*, cg::thread_block cta); 
    if(useWarpSerial==true)
    {
        block_reduction=block_reduce_warpserial<T, blockSize>;
    }
    else
    {
        block_reduction=block_reduce_cuda_sample_opt<T, blockSize>;
    }
    extern T  __shared__ sm[];
    T* sdata;
    if(useSM==true)
        sdata=sm;
    else
        sdata=g_odata;
    T  mySum = 0;
    if(n>2048)
    {
        mySum = serial_reduce<T,nIsPow2>(n, 
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


template <class T, bool nIsPow2>
__global__ void
reduce_kernel1(T *g_idata, T *g_odata, unsigned int n)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // double mySum = g_idata[i];
    T  mySum = 0;

    mySum = serial_reduce<T,nIsPow2>(n, 
        threadIdx.x, blockIdx.x,
        blockDim.x, blockDim.x*gridDim.x*2,
            g_idata);
    g_odata[blockIdx.x*blockDim.x+threadIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool useSM, bool useWarpSerial>
__global__ void
reduce_kernel2(T *g_idata, T *g_odata, unsigned int n)
/*
size of g_odata no smaller than n; equal to the multiply of blockSize; 
value index larger than n should be setted to 0 in advance;
*/
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    extern T  __shared__ sm[];
    T* sdata;
    if(useSM==true)
        sdata=sm;
    else
        sdata=g_odata;
    
    T (* block_reduction) (T, T*, cg::thread_block cta); 
    if(useWarpSerial==true)
    {
        block_reduction=block_reduce_warpserial<T, blockSize>;
    }
    else
    {
        block_reduction=block_reduce_cuda_sample_opt<T, blockSize>;
    }

    unsigned int tid = threadIdx.x;

    double  mySum = 0;

    if(blockIdx.x==0)
    {
        mySum=0;
        mySum = serial_reduce_final<T,blockSize>(n, 
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

template <class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial>
void __forceinline__ launchKernelBasedReduction(T *g_idata, T *g_odata, unsigned int gridSize,  unsigned int n)
{
        reduce_kernel1<T, nIsPow2><<<gridSize,blockSize>>>(g_idata,g_odata,n); 
        if(useSM==true)
            reduce_kernel2<T,blockSize,true, useWarpSerial><<<1,blockSize,blockSize*sizeof(T)>>>(g_odata,g_odata,n); 
        else
            reduce_kernel2<T,blockSize,false, useWarpSerial><<<1,blockSize>>>(g_odata,g_odata,n); 

}
template <class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial>
void __forceinline__ gridBasedReduction(T *g_idata, T *g_odata, unsigned int gridSize,  unsigned int n)
{


    void* KernelArgs[] = {(void**)&g_idata,(void**)&g_odata,(void*)&n}; 

    if( useSM==true)
    {
        cudaLaunchCooperativeKernel((void*)reduce_grid<T,blockSize,nIsPow2,true,useWarpSerial>, gridSize,blockSize, KernelArgs,blockSize*sizeof(T),0);
    }   
    else
    {
        cudaLaunchCooperativeKernel((void*)reduce_grid<T,blockSize,nIsPow2,false,useWarpSerial>, gridSize,blockSize, KernelArgs,0,0);
    }
}


template<class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial, bool useKernelLaunch>
void single_test(float& millisecond, T&gpu_result, unsigned int grid_size, unsigned int array_size ,T* h_input) 
{    
    T* h_output = (T*)malloc(sizeof(T)*array_size); 
    T* d_input; 
    T* d_output; 
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaMalloc((void**)&d_input, sizeof(T)*array_size); 
    cudaMalloc((void**)&d_output, sizeof(T)*array_size); 
    cudaMemcpy(d_input, h_input, sizeof(T)*array_size, cudaMemcpyHostToDevice); 
    cudaEventRecord(start);
    if(useKernelLaunch==true)
    {
        launchKernelBasedReduction<T, blockSize,true,useSM,useWarpSerial>(d_input, d_output, grid_size,  array_size);
    }
    else
    {
        gridBasedReduction<T, blockSize,true,useSM,useWarpSerial>(d_input, d_output, grid_size,  array_size);
    }
    cudaEventRecord(end);
    cudaMemcpy(h_output, d_output, sizeof(T)*array_size, cudaMemcpyDeviceToHost); 
    gpu_result=h_output[0];
    cudaDeviceSynchronize(); 
    cudaError_t e=cudaGetLastError(); 
    if(e!=cudaSuccess) 
    { 
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
    }
    cudaEventElapsedTime(&millisecond,start,end);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

#define my_single_test(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch) \
 single_test<type,threadcount,isPow2, useSM, useWarpSerial, useKernelLaunch>(millisecond, gpu_result, smx_count*block_per_sm,size, h_input);

#define switchuseSM(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
    if(useSM==true){my_single_test(type,threadcount,isPow2,true,useWarpSerial,useKernelLaunch);}\
    if(useSM==false){my_single_test(type,threadcount,isPow2,false,useWarpSerial,useKernelLaunch);}\

#define switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
    if(useWarpSerial==true){switchuseSM(type,threadcount,isPow2,useSM,true,useKernelLaunch);}\
    if(useWarpSerial==false){switchuseSM(type,threadcount,isPow2,useSM,false,useKernelLaunch);}    

#define switchuseKernelLaunch(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
    if(useKernelLaunch==true){switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,true);}\
    if(useKernelLaunch==false){switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,false);}    

#define switchisPow2(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
    if(isPow2==true){switchuseKernelLaunch(type,threadcount,true,useSM,useWarpSerial,useKernelLaunch);}\
    if(isPow2==false){switchuseKernelLaunch(type,threadcount,false,useSM,useWarpSerial,useKernelLaunch);}

// #define switchall(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     switchisPow2(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch);

#define switchall(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
    switch(threadcount) \
    {\
        case 32:\
            switchisPow2(double, 32, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
        case 64:\
            switchisPow2(double, 64, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
        case 128:\
            switchisPow2(double, 128, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
        case 256:\
            switchisPow2(double, 256, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
        case 512:\
            switchisPow2(double, 512, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
        case 1024:\
            switchisPow2(double, 1024, true, useSM,useWarpSerial,useKernelLaunch);\
            break;\
    }

void PrintHelp()
{
    printf(
            "--thread <n>:           thread per block\n \
            --block <n>:            block per sm\n \
            --base_array <n>:       average array per thread\n \
            --sharememory:          use shared memory at block level reduction (default false)\n \
            --warpserial:           use warpserial implementation (default false)\n\
            --kernellaunch:         use kernel launch as an implicit barrier (default false)\n");
    exit(1);
}


#include <getopt.h>
#include<iostream>
int main(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int smx_count = deviceProp.multiProcessorCount;
    double gpu_result;

    float millisecond;
    unsigned int thread_per_block=32;
    unsigned int block_per_sm=1;
    unsigned int data_per_thread=4;
    
    bool useSM=false;
    bool useWarpSerial=false;
    bool useKernelLaunch=false;

    unsigned int size = thread_per_block*smx_count*data_per_thread;

    const char* const short_opts = "t:b:n:swk";
    const option long_opts[] = {
            {"thread", required_argument, nullptr, 't'},
            {"block", required_argument, nullptr, 'b'},
            {"base_array", required_argument, nullptr, 'n'},
            {"sharememory", no_argument, nullptr, 's'},
            {"warpserial", no_argument, nullptr, 'w'},
            {"kernellaunch", no_argument, nullptr, 'k'},
            {nullptr, no_argument, nullptr, 0}
    };

    while (true)
    {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt)
        {
        case 't':
            thread_per_block = std::stoi(optarg);
            printf("thread set to: %d\n",thread_per_block);
            break;

        case 'b':
            block_per_sm = std::stoi(optarg);
            printf("block set to: %d\n",block_per_sm);
            break;

        case 'n':
            data_per_thread = std::stoi(optarg);
            printf("data per thread set to: %d\n",data_per_thread);
            break;

        case 's':
            useSM = true;
            printf("useSM is set to true\n");
            break;

        case 'w':
            useWarpSerial = true;
            printf("useWarpSerial is set to true\n");
            break;

        case 'k':
            useKernelLaunch = true;
            printf("useKernelLaunch is set to true\n");
            break;

        default:
            PrintHelp();
            break;
        }
    }

    double* h_input = (double*)malloc(sizeof(double)*size);
    for(int i=0; i<size; i++) 
    {
        h_input[i]=1;
    }
    double cpu_result=cpu_reduce(h_input,size);

    switchall(thread_per_block, 32, true, useSM,useWarpSerial,useKernelLaunch);
    
    printf("%f-%f=%f\n",cpu_result,gpu_result,cpu_result-gpu_result);   
    printf("block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
          1,thread_per_block, size,\
          (double)millisecond, (double)size*sizeof(double)/1000/1000/1000/(millisecond/1000));\

  
    free(h_input);

 }

