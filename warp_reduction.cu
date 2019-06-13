
//assume 32 threads here
//tile sync
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ void reduceTile(double *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    sdata[tid]+=sdata[tid+16];cg::sync(tile32);
    sdata[tid]+=sdata[tid+8];cg::sync(tile32);
    sdata[tid]+=sdata[tid+4];cg::sync(tile32);
    sdata[tid]+=sdata[tid+2];cg::sync(tile32);
    sdata[tid]+=sdata[tid+1];cg::sync(tile32);
}
//block sync
__device__ __forceinline__ void reduceBlock(double *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    
    sdata[tid]+=sdata[tid+16];cg::sync(cta);
    sdata[tid]+=sdata[tid+8];cg::sync(cta);
    sdata[tid]+=sdata[tid+4];cg::sync(cta);
    sdata[tid]+=sdata[tid+2];cg::sync(cta);
    sdata[tid]+=sdata[tid+1];cg::sync(cta);
}
//shuffle
__device__ __forceinline__ void reduceshuffle(double *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    double mySum = sdata[tid];
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    mySum+=tile32.shfl_down(mySum,16);
    mySum+=tile32.shfl_down(mySum,8);
    mySum+=tile32.shfl_down(mySum,4);
    mySum+=tile32.shfl_down(mySum,2);
    mySum+=tile32.shfl_down(mySum,1);
    if(tid==0)
        sdata[tid]=mySum;
}

__global__ void reduceBlock_BLOCK(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;
    if(tid<32)
    {
        sdata[tid]=idata[tid];
        sdata[tid+32]=0;
    }

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    if(tid<32)
    {   
        sdata[tid]+=sdata[tid+16];cg::sync(block);
        sdata[tid]+=sdata[tid+8];cg::sync(block);
        sdata[tid]+=sdata[tid+4];cg::sync(block);
        sdata[tid]+=sdata[tid+2];cg::sync(block);
        sdata[tid]+=sdata[tid+1];cg::sync(block);
    }  
    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_TILE(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    sdata[tid]+=sdata[tid+16];cg::sync(tile32);
    sdata[tid]+=sdata[tid+8];cg::sync(tile32);
    sdata[tid]+=sdata[tid+4];cg::sync(tile32);
    sdata[tid]+=sdata[tid+2];cg::sync(tile32);
    sdata[tid]+=sdata[tid+1];cg::sync(tile32);

    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_SHUFFLE(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    double mySum = sdata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    
    mySum+=tile32.shfl_down(mySum,16);
    mySum+=tile32.shfl_down(mySum,8);
    mySum+=tile32.shfl_down(mySum,4);
    mySum+=tile32.shfl_down(mySum,2);
    mySum+=tile32.shfl_down(mySum,1);
    if(tid==0)
        sdata[tid]=mySum;
    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_SERIAL(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    double mySum = sdata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    if(tid==0)
    {
        mySum+=sdata[1];
        mySum+=sdata[2];
        mySum+=sdata[3];
        mySum+=sdata[4];
        mySum+=sdata[5];
        mySum+=sdata[6];
        mySum+=sdata[7];
        mySum+=sdata[8];
        mySum+=sdata[9];
        mySum+=sdata[10];
        mySum+=sdata[11];
        mySum+=sdata[12];
        mySum+=sdata[13];
        mySum+=sdata[14];
        mySum+=sdata[15];
        mySum+=sdata[16];
        mySum+=sdata[17];
        mySum+=sdata[18];
        mySum+=sdata[19];
        mySum+=sdata[20];
        mySum+=sdata[21];
        mySum+=sdata[22];
        mySum+=sdata[23];
        mySum+=sdata[24];
        mySum+=sdata[25];
        mySum+=sdata[26];
        mySum+=sdata[27];
        mySum+=sdata[28];
        mySum+=sdata[29];
        mySum+=sdata[30];
        mySum+=sdata[31];
        sdata[tid]=mySum;
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}
int main()
{
    double* h_input = (double*)malloc(sizeof(double)*32);
    double* h_output = (double*)malloc(sizeof(double)*4);
    unsigned int * h_time_stamp = (unsigned int*)malloc(sizeof(unsigned int)*8);
    double* d_input;
    double* d_output;
    unsigned int* d_time_stamp;
    cudaMalloc((void**)&d_input, sizeof(double)*32);
    cudaMalloc((void**)&d_output, sizeof(double)*4);
    cudaMalloc((void**)&d_time_stamp , sizeof(unsigned int)*8);
    for(int i=0; i<32; i++)
    {
        h_input[i]=i;
    }
    cudaMemcpy(d_input, h_input, sizeof(double)*32, cudaMemcpyHostToDevice);
    reduceBlock_TILE<<<1,32>>>(d_input,d_output,d_time_stamp);
    reduceBlock_SHUFFLE<<<1,32>>>(d_input,d_output+1,d_time_stamp+2);
    reduceBlock_BLOCK<<<1,64>>>(d_input,d_output+2,d_time_stamp+4);
    reduceBlock_SERIAL<<<1,32>>>(d_input,d_output+3,d_time_stamp+6);
    cudaMemcpy(h_output, d_output, sizeof(double)*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time_stamp, d_time_stamp, sizeof(unsigned int)*8, cudaMemcpyDeviceToHost);
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess)
    {
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
    }
    printf("result: %f %f %f %f\n",h_output[0],h_output[1],h_output[2],h_output[3]);
    printf("time: %d %d %d %d\n",h_time_stamp[1]-h_time_stamp[0],
        h_time_stamp[3]-h_time_stamp[2],
        h_time_stamp[5]-h_time_stamp[4],
        h_time_stamp[7]-h_time_stamp[6]);
 }



