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

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}

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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

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
    T* sdata;
    if(useSM==true)
        sdata=SharedMemory<T>();
    else
        sdata=g_odata;
    T  mySum = 0;
    if(gridDim.x>1)
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
            mySum = serial_reduce_final<T,blockSize>(blockSize*gridDim.x, 
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
            mySum = serial_reduce<T,nIsPow2>(n, 
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

    T* sdata;
    if(useSM==true)
        sdata=SharedMemory<T>();
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
            g_idata);

        mySum=block_reduction(mySum, sdata, cta);
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }

}
template <class T>
T cpu_reduce(T* array, unsigned int array_size)
{
    T sum=0;
    for(int i=0; i<array_size; i++)
    {
        sum+=array[i];
    }
    return sum;
}

template <class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial>
void __forceinline__ launchKernelBasedReduction(T *g_idata, T *g_odata, unsigned int gridSize,  unsigned int n)
{  
    void* KernelArgs1[] = {(void**)&g_idata,(void**)&g_odata,(void*)&n}; 
    unsigned int size_k2=blockSize*gridSize;
    void* KernelArgs2[] = {(void**)&g_odata,(void**)&g_odata,(void*)&size_k2}; 

    {
        cudaLaunchCooperativeKernel((void*)reduce_kernel1<T,nIsPow2>, gridSize,blockSize, KernelArgs1,blockSize*sizeof(T),0);
        if( useSM==true)
        {
            cudaLaunchCooperativeKernel((void*)reduce_kernel2<T,blockSize,true,useWarpSerial>, 1,blockSize, KernelArgs2,blockSize*sizeof(T),0);
        }   
        else
        {
            cudaLaunchCooperativeKernel((void*)reduce_kernel2<T,blockSize,false,useWarpSerial>, 1,blockSize, KernelArgs2,0,0);
        }
    }
}
template <class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial>
void __forceinline__ gridBasedReduction(T *g_idata, T *g_odata, unsigned int gridSize,  unsigned int n)
{
    void* KernelArgs[] = {(void**)&g_idata,(void**)&g_odata,(void*)&n}; 

    {
        if( useSM==true)
        {
            cudaLaunchCooperativeKernel((void*)reduce_grid<T,blockSize,nIsPow2,true,useWarpSerial>, gridSize,blockSize, KernelArgs,blockSize*sizeof(T),0);
        }   
        else
        {
            cudaLaunchCooperativeKernel((void*)reduce_grid<T,blockSize,nIsPow2,false,useWarpSerial>, gridSize,blockSize, KernelArgs,0,0);
        }
    }
}

//though there are no peer access in single gpu. I set it as CAN peer access here.
void getAceessMatrix(unsigned int *access, unsigned int gpu_count)
{
    for(int s=0; s<gpu_count; s++)
    {
        cudaSetDevice(s);
        access[s*gpu_count+s]=1;
        for(int d=s+1; d<gpu_count; d++)
        {
            int l_access = 0;
            cudaDeviceCanAccessPeer(&l_access, s, d);
            if (l_access) {
                cudaDeviceEnablePeerAccess(d, 0);
                cudaCheckError();
                cudaSetDevice(d);
                cudaDeviceEnablePeerAccess(s, 0);
                cudaSetDevice(s);
                cudaCheckError();
            }
            access[s*gpu_count+d]=l_access;      
            access[d*gpu_count+s]=l_access;      
        }
    }
}



// int main()
// {
//     int numGPUs;
//     cudaGetDeviceCount(&numGPUs);
//     unsigned int * access=(unsigned int *)malloc(sizeof(unsigned int)*numGPUs*numGPUs);
//     getAceessMatrix(access, numGPUs);
//     for(int i=0; i<numGPUs; i++)
//     {
//         for(int j=0; j<numGPUs; j++)
//         {
//             printf("%d\t",access[i*numGPUs+j]);
//         }
//         printf("\n");
//     }
// }

template<class T>
void basic_transfer(T**g_idata, T*g_odata, unsigned int size_gpu, unsigned int gpu_count, cudaStream_t *mstream)
{
    for(int deviceid=0; deviceid<gpu_count;deviceid++)
    {
            // cudaSetDevice(deviceid);
        cudaMemcpyPeerAsync(g_odata+deviceid*size_gpu, 0, g_idata[deviceid], deviceid, size_gpu*sizeof(T), mstream[deviceid]);
            // cudaMemcpyAsync(g_odata+deviceid*size_gpu, g_idata[deviceid], size_gpu*sizeof(T), cudaMemcpyDeviceToDevice, mstream[deviceid]);
        cudaCheckError();
    }
}

template<class T>
void basic_transfer_alter
    (   T****source_ptr, T**** destinate_ptr, //[step][source][destinate]*
        unsigned int ***size,//[step][source gpu][destinate gpu]
        // unsigned int ***size,//[step][source gpu][destinate gpu]
        unsigned int gpu_count,
        unsigned int steps,
        cudaStream_t *mstream)//for synchronization from source
{
    for(unsigned int step=0; step<steps; step++)
    {
        //async transfer
        if(step>1)
        {
            for(unsigned int src_gpu=0; src_gpu<gpu_count; src_gpu++)
            {
                cudaSetDevice(src_gpu);
                for(int dst_gpu=0; dst_gpu<gpu_count; dst_gpu++)
                {
                    if(size[step-1][src_gpu][dst_gpu]==0)continue;
                    cudaStreamSynchronize(mstream[src_gpu]);
                    break;
                }
            }
        }

        for(unsigned int src_gpu=0; src_gpu<gpu_count; src_gpu++)
        {
            // cudaSetDevice(src_gpu);
            for(int dst_gpu=0; dst_gpu<gpu_count; dst_gpu++)
            {
                if(size[step][src_gpu][dst_gpu]==0)continue;
                cudaMemcpyPeer(destinate_ptr[step][src_gpu][dst_gpu], dst_gpu, source_ptr[step][src_gpu][dst_gpu], src_gpu, size[step][src_gpu][dst_gpu]);
            }
            cudaCheckError();
        }
    }

    for(unsigned int src_gpu=0; src_gpu<gpu_count; src_gpu++)
    {
        cudaSetDevice(src_gpu);
        cudaStreamSynchronize(mstream[src_gpu]);
    }
}

template<class T>
void mc_transfer(T**g_idata, T*g_odata, unsigned int size_gpu, unsigned int gpu_count, cudaStream_t *mstream)
{
    for(int deviceid=0; deviceid<gpu_count;deviceid++)
    {
            // cudaSetDevice(deviceid);
        // cudaMemcpyPeerAsync(g_odata+deviceid*size_gpu, 0, g_idata[deviceid], deviceid, size_gpu*sizeof(T), mstream[deviceid]);
        cudaMemcpyAsync(g_odata+deviceid*size_gpu, g_idata[deviceid], size_gpu*sizeof(T), cudaMemcpyDeviceToDevice, mstream[deviceid]);
        cudaCheckError();
    }
}

template <class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial>
void __forceinline__ launchMultiKernelBasedReduction(double&millisecond, T **g_idata, T *g_odata, unsigned int gridSize,  unsigned int data_per_gpu, unsigned int gpu_count=1)
{  

        cudaStream_t *mstream = (cudaStream_t*)malloc(sizeof(cudaStream_t)*gpu_count);
        
        timespec tsstart,tsend;
        long time_elapsed_ns;

        //first compute
        void***packedKernelArgs = (void***)malloc(sizeof(void**)*gpu_count); 
        cudaLaunchParams *launchParamsList = (cudaLaunchParams *)malloc(
            sizeof(cudaLaunchParams)*gpu_count);

        for(int deviceid=0; deviceid<gpu_count;deviceid++)
        {
            cudaSetDevice(deviceid);
            packedKernelArgs[deviceid]=(void**)malloc(sizeof(void*)*3);
            packedKernelArgs[deviceid][0]=(void*)&g_idata[deviceid];
            packedKernelArgs[deviceid][1]=(void*)&g_idata[deviceid];
            packedKernelArgs[deviceid][2]=(void*)&data_per_gpu;
            cudaStreamCreate(&mstream[deviceid]);
            
            launchParamsList[deviceid].func=(void*)reduce_kernel1<T,nIsPow2>;
            launchParamsList[deviceid].gridDim=gridSize;
            launchParamsList[deviceid].blockDim=blockSize;
            launchParamsList[deviceid].sharedMem=0;
            launchParamsList[deviceid].stream=mstream[deviceid];
            launchParamsList[deviceid].args=packedKernelArgs[deviceid];
        }
        cudaCheckError();
        unsigned int size_gpu=blockSize*gridSize;
        //initialize transfer strategy
        unsigned int step=1;

        T**** source_ptr=(T****)malloc(sizeof(T***)*step);
        T**** destinate_ptr=(T****)malloc(sizeof(T***)*step);
        // unsigned int size[1][2][2];
        unsigned int ***size=(unsigned int***)malloc(sizeof(unsigned int**)*step);
        for(int s=0; s<step; s++)
        {
            source_ptr[s]=(T***)malloc(sizeof(T**)*gpu_count);
            destinate_ptr[s]=(T***)malloc(sizeof(T**)*gpu_count);
            size[s]=(unsigned int **)malloc(sizeof(unsigned int *)*gpu_count);
            for(int src_gpu=0; src_gpu<gpu_count; src_gpu++)
            {
                source_ptr[s][src_gpu]=(T**)malloc(sizeof(T*)*gpu_count);
                destinate_ptr[s][src_gpu]=(T**)malloc(sizeof(T*)*gpu_count);
                size[s][src_gpu]=(unsigned int *)malloc(sizeof(unsigned int )*gpu_count);
                for(int dst_gpu=0; dst_gpu<gpu_count; dst_gpu++)
                {
                    source_ptr[s][src_gpu][dst_gpu]=NULL;
                    destinate_ptr[s][src_gpu][dst_gpu]=NULL;
                    size[s][src_gpu][dst_gpu]=0;
                }
            }
        }

        size[0][0][0]=size_gpu;
        source_ptr[0][0][0]=g_idata[0];
        destinate_ptr[0][0][0]=g_odata;

        size[0][1][0]=size_gpu;
        source_ptr[0][1][0]=g_idata[1];
        destinate_ptr[0][1][0]=g_odata+size_gpu;

        //compute time
        /************************************/
        clock_gettime(CLOCK_REALTIME, &tsstart);

        cudaLaunchCooperativeKernelMultiDevice(launchParamsList, gpu_count);
        

        //data transfer
        // mc_transfer<T>(g_idata, g_odata, size_gpu, gpu_count, mstream);
        basic_transfer_alter<T>(
            source_ptr, destinate_ptr, size,
            gpu_count,
            step,
            mstream);
        
        // cudaSetDevice(0);
        // for(int deviceid=0; deviceid<gpu_count;deviceid++)
        // {
        //     cudaSetDevice(deviceid);
        //     cudaDeviceSynchronize();
        //     cudaCheckError();
        // }

        //last compute
        cudaSetDevice(0);
        launchKernelBasedReduction<T,blockSize,true,useSM,useWarpSerial>(g_odata,g_odata,gridSize,size_gpu*gpu_count);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_REALTIME, &tsend);
        /************************************/
        time_elapsed_ns = (tsend.tv_nsec-tsstart.tv_nsec);
        time_elapsed_ns += 1000000000*(tsend.tv_sec-tsstart.tv_sec);

        millisecond = (double)time_elapsed_ns/1000;

        cudaCheckError();

        for(int deviceid=0; deviceid<gpu_count;deviceid++)
        {
            cudaSetDevice(deviceid);    
            cudaCheckError();
            cudaStreamDestroy(mstream[deviceid]);
        }

        cudaCheckError();

        free(mstream);
        free(packedKernelArgs);
        free(launchParamsList);

        for(int s=0; s<step; s++)
        {
            for(int src_gpu=0; src_gpu<gpu_count; src_gpu++)
            {
                free(source_ptr[s][src_gpu]);
                free(destinate_ptr[s][src_gpu]);
                free(size[s][src_gpu]);       
            }
            free(source_ptr[s]);
            free(destinate_ptr[s]);
            free(size[s]);
        }
}

template<class T, unsigned int blockSize, bool nIsPow2,bool useSM, bool useWarpSerial, bool useKernelLaunch>
void single_test(double& millisecond, T&gpu_result, unsigned int gridSize, unsigned int g_array_size ,T* h_input, unsigned int gpu_count=1) 
{   
    cudaSetDevice(0);
    unsigned int size_gpu=blockSize*gridSize;
    unsigned int l_array_size=g_array_size>size_gpu?g_array_size:size_gpu;
    T* h_output = (T*)malloc(sizeof(T)*size_gpu*gpu_count); 
    T** d_input = (T**)malloc(sizeof(T*)*gpu_count); 
    T* d_output; 
    


    for(int deviceid=0; deviceid<gpu_count; deviceid++)
    {
        printf("here is devide %d\n",deviceid);
	cudaSetDevice(deviceid);
    	cudaCheckError();
        cudaMalloc((void**)&d_input[deviceid], sizeof(T)*l_array_size); 
        cudaMemcpy(d_input[deviceid], h_input, sizeof(T)*g_array_size, cudaMemcpyHostToDevice); 
    }
    cudaSetDevice(0);
    cudaCheckError();
    cudaMalloc((void**)&d_output, sizeof(T)*size_gpu*gpu_count); 
    
    
    cudaCheckError();

    launchMultiKernelBasedReduction<T, blockSize,true,useSM,useWarpSerial>(millisecond, d_input, d_output, gridSize,  l_array_size, gpu_count);
    cudaCheckError();

    
    cudaCheckError();
    cudaMemcpy(h_output, d_output, sizeof(T)*size_gpu*gpu_count, cudaMemcpyDeviceToHost); 
    gpu_result=h_output[0];
    cudaDeviceSynchronize(); 
    cudaError_t e=cudaGetLastError(); 
    if(e!=cudaSuccess) 
    { 
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
    }

    
    free(h_output);
    cudaFree(d_input);
    for(int deviceid=0; deviceid<gpu_count; deviceid++)
    {
        cudaSetDevice(deviceid);
        cudaFree(d_input[deviceid]);
    }
    free(d_input);
    cudaFree(d_output);
}

#define my_single_test(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch,gpu_count) \
 do{\
    double* lats=(double*)malloc(sizeof(double)*repeat);\
    for(int i=0; i<repeat; i++)\
    {\
        single_test<type,threadcount,isPow2, useSM, useWarpSerial, useKernelLaunch>(millisecond, gpu_result, smx_count*block_per_sm,size, h_input,gpu_count);\
        lats[i]=millisecond;\
    }\
    millisecond=0;\
    for(int i=skip; i<repeat; i++)\
    {\
        millisecond+=lats[i];\
    }\
    millisecond=millisecond/(repeat-skip);\
    free(lats);\
    }while(0)

 


// #define thread_per_block 1024

int main()
{
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned int smx_count = deviceProp.multiProcessorCount;

    unsigned int thread_per_block=1024;
    unsigned int block_per_sm=2;
    unsigned int data_per_thread=4;
    unsigned int type=0; 

    unsigned int size=data_per_thread*thread_per_block*block_per_sm;

    unsigned int repeat=1;
    unsigned int skip=0;

    bool useSM=false;
    bool useWarpSerial=false;
    bool useKernelLaunch=false;
    double millisecond;
    bool isPow2=false;
    double gpu_result;
    double* h_input = (double*)malloc(sizeof(double)*size);
    for(int i=0; i<size; i++) 
    {
        h_input[i]=1;
    }
    double cpu_result=cpu_reduce<double>(h_input,size);
    if(size%(thread_per_block*smx_count*2)==0)
    { 
        isPow2=true;
    }
    else
    {
        isPow2=false;
    }
    unsigned int gpu_count=8;
    // switchall(T, thread_per_block, isPow2, useSM,useWarpSerial,useKernelLaunch);
    my_single_test(double,1024,true,false,true,true,gpu_count);

    fprintf(stderr,"%f-%f=%f\n",cpu_result,(double)gpu_result,cpu_result*gpu_count-gpu_result);   
    printf("useSM: %d, use warp serial:%d, use kernel launch:%d, block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
          useSM, useWarpSerial,useKernelLaunch,\
          block_per_sm,thread_per_block, size,\
          (double)millisecond, (double)size*sizeof(double)/1000/1000/1000/(millisecond/1000));\

  
    free(h_input);   
}


// #define switchuseSM(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     if(useSM==true){my_single_test(type,threadcount,isPow2,true,useWarpSerial,useKernelLaunch);}\
//     if(useSM==false){my_single_test(type,threadcount,isPow2,false,useWarpSerial,useKernelLaunch);}\

// #define switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     if(useWarpSerial==true){switchuseSM(type,threadcount,isPow2,useSM,true,useKernelLaunch);}\
//     if(useWarpSerial==false){switchuseSM(type,threadcount,isPow2,useSM,false,useKernelLaunch);}    

// #define switchuseKernelLaunch(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     if(useKernelLaunch==true){switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,true);}\
//     if(useKernelLaunch==false){switchuseWarpSerial(type,threadcount,isPow2,useSM,useWarpSerial,false);}    

// #define switchisPow2(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     if(isPow2==true){switchuseKernelLaunch(type,threadcount,true,useSM,useWarpSerial,useKernelLaunch);}\
//     if(isPow2==false){switchuseKernelLaunch(type,threadcount,false,useSM,useWarpSerial,useKernelLaunch);}

// #define switchall(type,threadcount,isPow2,useSM,useWarpSerial,useKernelLaunch)\
//     switch(threadcount) \
//     {\
//         case 32:\
//             switchisPow2(type, 32, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//         case 64:\
//             switchisPow2(type, 64, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//         case 128:\
//             switchisPow2(type, 128, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//         case 256:\
//             switchisPow2(type, 256, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//         case 512:\
//             switchisPow2(type, 512, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//         case 1024:\
//             switchisPow2(type, 1024, isPow2, useSM,useWarpSerial,useKernelLaunch);\
//             break;\
//     }



// template <class T>
// void runTest(unsigned int thread_per_block, unsigned int block_per_sm,
//     unsigned int smx_count,
//     unsigned int size,
//     unsigned int repeat,
//     unsigned int skip,
//     bool useSM,
//     bool useWarpSerial,
//     bool useKernelLaunch
//     )
// {
//     float millisecond;
//     bool isPow2=false;
//     T gpu_result;
//     T* h_input = (T*)malloc(sizeof(T)*size);
//     for(int i=0; i<size; i++) 
//     {
//         h_input[i]=1;
//     }
//     double cpu_result=cpu_reduce<T>(h_input,size);
//     if(size%(thread_per_block*smx_count*2)==0)
//     { 
//         isPow2=true;
//     }
//     else
//     {
//         isPow2=false;
//     }
    
//     switchall(T, thread_per_block, isPow2, useSM,useWarpSerial,useKernelLaunch);

//     fprintf(stderr,"%f-%f=%f\n",cpu_result,(double)gpu_result,cpu_result-gpu_result);   
//     printf("useSM: %d, use warp serial:%d, use kernel launch:%d, block/SM %d thread %d totalsize %d time: %f ms speed: %f GB/s\n",\
//           useSM, useWarpSerial,useKernelLaunch,\
//           block_per_sm,thread_per_block, size,\
//           (double)millisecond, (double)size*sizeof(T)/1000/1000/1000/(millisecond/1000));\

  
//     free(h_input); 
// }

// void PrintHelp()
// {
//     printf(
//             "--thread <n>(t):           thread per block\n \
//              --block <n>(b):            block per sm\n \
//              --base_array <n>(a):       average array per thread\n \
//              --array <n>(n):            total array size\n \
//              --repeat <n>(r):           time of experiment (larger than 2)\n \
//              --type <n>(v):             type of experiment (0:int 1:float 2:double)\n \
//              --sharememory(s):          use shared memory at block level reduction (default false)\n \
//              --warpserial(w):           use warpserial implementation (default false)\n \
//              --kernellaunch(k):         use kernel launch as an implicit barrier (default false)\n");
//     exit(1);
// }

// #include <getopt.h>
// #include<iostream>
// int main(int argc, char **argv)
// {
//     cudaDeviceProp deviceProp;
//     cudaSetDevice(0);
//     cudaGetDeviceProperties(&deviceProp, 0);
//     unsigned int smx_count = deviceProp.multiProcessorCount;

//     unsigned int thread_per_block=1024;
//     unsigned int block_per_sm=2;
//     unsigned int data_per_thread=4;
//     unsigned int type=0; 

//     bool useSM=false;
//     bool useWarpSerial=false;
//     bool useKernelLaunch=false;
//     unsigned int size = 0;

//     unsigned int repeat=11;
//     unsigned int skip=1;

//     const char* const short_opts = "t:b:a:n:r:v:swk";
//     const option long_opts[] = {
//             {"thread", required_argument, nullptr, 't'},
//             {"block", required_argument, nullptr, 'b'},
//             {"base_array", required_argument, nullptr, 'a'},
//             {"array", required_argument, nullptr, 'n'},
//             {"repeat", required_argument, nullptr, 'r'},
//             {"type", required_argument, nullptr, 'v'},
//             {"sharememory", no_argument, nullptr, 's'},
//             {"warpserial", no_argument, nullptr, 'w'},
//             {"kernellaunch", no_argument, nullptr, 'k'},
//             {nullptr, no_argument, nullptr, 0}
//     };

//     while (true)
//     {
//         const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

//         if (-1 == opt)
//             break;

//         switch (opt)
//         {
//         case 't':
//             thread_per_block = std::stoi(optarg);
//             fprintf(stderr,"thread set to: %d\n",thread_per_block);
//             break;

//         case 'b':
//             block_per_sm = std::stoi(optarg);
//             fprintf(stderr,"block set to: %d\n",block_per_sm);
//             break;

//         case 'a':
//             data_per_thread = std::stoi(optarg);
//             fprintf(stderr,"data per thread set to: %d\n",data_per_thread);
//             break;

//         case 'n':
//             size = std::stoi(optarg);
//             fprintf(stderr,"array size set to: %d\n",size);
//             break;

//         case 'r':
//             repeat = std::stoi(optarg);
//             if(repeat<=2)
//             {
//                 repeat=1;
//                 skip=0;
//                 fprintf(stderr,"repeat set to: %d and skip 0 experiment\n",repeat);
//             }
//             else
//             {
//                 fprintf(stderr,"repeat set to: %d\n",repeat);
//             }
//             break;
//         case 'v':
//             type = std::stoi(optarg);
//             type=type>=3?0:type;
//             fprintf(stderr,"type set to (0:int 1:float 2:double): %d\n",type);
//             break;
//         case 's':
//             useSM = true;
//             fprintf(stderr,"useSM is set to true\n");
//             break;

//         case 'w':
//             useWarpSerial = true;
//             fprintf(stderr,"useWarpSerial is set to true\n");
//             break;

//         case 'k':
//             useKernelLaunch = true;
//             fprintf(stderr,"useKernelLaunch is set to true\n");
//             break;

//         default:
//             PrintHelp();
//             break;
//         }
//     }

//     size = size==0?block_per_sm*thread_per_block*smx_count*data_per_thread:size;
//     switch(type)
//     {
//         case 0:
//         runTest<int>(thread_per_block, block_per_sm,smx_count,
//                 size,repeat,skip,
//                 useSM,useWarpSerial,useKernelLaunch);
//         break;
//         case 1:
//         runTest<float>(thread_per_block, block_per_sm,smx_count,
//                 size,repeat,skip,
//                 useSM,useWarpSerial,useKernelLaunch);
//         break;
//         case 2:
//         runTest<double>(thread_per_block, block_per_sm,smx_count,
//                 size,repeat,skip,
//                 useSM,useWarpSerial,useKernelLaunch);
//         break;

//     }

//  }

