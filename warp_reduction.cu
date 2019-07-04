
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
    // if(tid<32)
    // {
        sdata[tid]=idata[tid];
        sdata[tid+32]=0;
    // }

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    // if(tid<32)
    // {   
        sdata[tid]+=sdata[tid+16];cg::sync(block);
        sdata[tid]+=sdata[tid+8];cg::sync(block);
        sdata[tid]+=sdata[tid+4];cg::sync(block);
        sdata[tid]+=sdata[tid+2];cg::sync(block);
        sdata[tid]+=sdata[tid+1];cg::sync(block);
    // }  
    
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

    sdata[tid]+=sdata[tid+16];cg::sync(cg::tiled_partition(block,32));
    sdata[tid]+=sdata[tid+8];cg::sync(cg::tiled_partition(block,16));
    sdata[tid]+=sdata[tid+4];cg::sync(cg::tiled_partition(block,8));
    sdata[tid]+=sdata[tid+2];cg::sync(cg::tiled_partition(block,4));
    sdata[tid]+=sdata[tid+1];cg::sync(cg::tiled_partition(block,2));

    // sdata[tid]+=sdata[tid+16];cg::sync(cg::tiled_partition<32>(block));
    // sdata[tid]+=sdata[tid+8];cg::sync(cg::tiled_partition<16>(block));
    // sdata[tid]+=sdata[tid+4];cg::sync(cg::tiled_partition<8>(block));
    // sdata[tid]+=sdata[tid+2];cg::sync(cg::tiled_partition<4>(block));
    // sdata[tid]+=sdata[tid+1];cg::sync(cg::tiled_partition<2>(block));
    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_TILEW(double*idata,double*output,unsigned int *time_stamp)
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
    
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid==0)
        sdata[tid]=mySum;
    
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}




__global__ void reduceBlock_COA(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");


    if(tid<16){ sdata[tid]+=sdata[tid+16];cg::sync(cg::coalesced_threads());}
    if(tid<8){sdata[tid]+=sdata[tid+8];cg::sync(cg::coalesced_threads());}
    if(tid<4){sdata[tid]+=sdata[tid+4];cg::sync(cg::coalesced_threads());}
    if(tid<2){sdata[tid]+=sdata[tid+2];cg::sync(cg::coalesced_threads());}
    if(tid<1){sdata[tid]+=sdata[tid+1];cg::sync(cg::coalesced_threads());}
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_COAW(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;
    cg::coalesced_group csg = cg::coalesced_threads();

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");


    sdata[tid]+=sdata[tid+16];cg::sync(csg);
    sdata[tid]+=sdata[tid+8];cg::sync(csg);
    sdata[tid]+=sdata[tid+4];cg::sync(csg);
    sdata[tid]+=sdata[tid+2];cg::sync(csg);
    sdata[tid]+=sdata[tid+1];cg::sync(csg);
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}



__global__ void reduceBlock_COA_SHUFFLE(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;
    
    double mySum = sdata[tid];
    cg::coalesced_group csg = cg::coalesced_threads();
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    // if(tid<32){
    //     mySum+=cg::coalesced_threads().shfl_down(mySum,16);
    // } 
    // if(tid<16){
    //     mySum+=cg::coalesced_threads().shfl_down(mySum,8);
    // } 
    // if(tid<8){
    //     mySum+=cg::coalesced_threads().shfl_down(mySum,4);
    // } 
    // if(tid<4){
    //     mySum+=cg::coalesced_threads().shfl_down(mySum,2);
    // } 
    // if(tid<2){
    //     mySum+=cg::coalesced_threads().shfl_down(mySum,1);
    // } 
    
    mySum+=csg.shfl_down(mySum,16);
    mySum+=csg.shfl_down(mySum,8);
    mySum+=csg.shfl_down(mySum,4);
    mySum+=csg.shfl_down(mySum,2);
    mySum+=csg.shfl_down(mySum,1);

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    if(tid==0)
        sdata[tid]=mySum;
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_NOSYNC(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    double __shared__ volatile sdata[64];
    unsigned int tid = block.thread_rank();
    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    sdata[tid]+=sdata[tid+16];
    sdata[tid]+=sdata[tid+8];
    sdata[tid]+=sdata[tid+4];
    sdata[tid]+=sdata[tid+2];
    sdata[tid]+=sdata[tid+1];

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

    double mySum=0;// = sdata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    // if(tid==0)
    // {
        mySum+=sdata[0];
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
    // }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void reduceBlock_SERIAL_BASIC(double*idata,double*output,unsigned int *time_stamp)
{
    cg::thread_block block = cg::this_thread_block();

    double __shared__ sdata[64];
    unsigned int tid = block.thread_rank();
    
    cg::thread_group tile32 = cg::tiled_partition(block,32);

    unsigned int  start,stop;

    sdata[tid]=idata[tid];
    sdata[tid+32]=0;

    double mySum=0;// = sdata[tid];

    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
    // if(tid==0)
    // {
        mySum+=sdata[1];
        sdata[0]=mySum;
        // sync(tile32);
        // sync(tile32);
    // }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    time_stamp[0]=start;
    time_stamp[1]=stop;
    if(tid==0)
        output[tid]=sdata[tid];
}

__global__ void k_base_kernel (float q, float p, double *out, unsigned int *time_stamp=NULL, unsigned int tile=32){
        unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
        cg::thread_group block = cg::this_thread_block();
        cg::thread_group tg = cg::tiled_partition(block, tile);
        
        unsigned int  start,end;
        asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
        cg::sync(tg);
        // cg::sync(tg);
        asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");
        unsigned int warp_id = id/32;
        if(id%32==0)
        {
            if(NULL!=time_stamp){
                time_stamp[warp_id*2]=start;
                time_stamp[warp_id*2+1]=end;
            }
        }
        out[id]=(double)q;
    }

int main()
{
    double* h_input = (double*)malloc(sizeof(double)*32);
    double* h_output = (double*)malloc(sizeof(double)*20);
    unsigned int * h_time_stamp = (unsigned int*)malloc(sizeof(unsigned int)*40);
    double* d_input;
    double* d_output;
    unsigned int* d_time_stamp;
    cudaMalloc((void**)&d_input, sizeof(double)*32);
    cudaMalloc((void**)&d_output, sizeof(double)*20);
    cudaMalloc((void**)&d_time_stamp , sizeof(unsigned int)*40);
    for(int i=0; i<32; i++)
    {
        h_input[i]=i;
    }
    cudaMemcpy(d_input, h_input, sizeof(double)*32, cudaMemcpyHostToDevice);
    reduceBlock_BLOCK<<<1,32>>>(d_input,d_output+8,d_time_stamp+16);
    reduceBlock_SERIAL<<<1,32>>>(d_input,d_output+0,d_time_stamp+0);
    reduceBlock_NOSYNC<<<1,32>>>(d_input,d_output+1,d_time_stamp+2);
    reduceBlock_TILE<<<1,32>>>(d_input,d_output+2,d_time_stamp+4);
    reduceBlock_TILEW<<<1,32>>>(d_input,d_output+3,d_time_stamp+6);
    reduceBlock_SHUFFLE<<<1,32>>>(d_input,d_output+4,d_time_stamp+8);
    reduceBlock_COA<<<1,32>>>(d_input,d_output+5,d_time_stamp+10);
    reduceBlock_COAW<<<1,32>>>(d_input,d_output+6,d_time_stamp+12);
    reduceBlock_COA_SHUFFLE<<<1,32>>>(d_input,d_output+7,d_time_stamp+14);
    reduceBlock_SERIAL_BASIC<<<1,32>>>(d_input,d_output+9,d_time_stamp+18);
    k_base_kernel<<<1,32>>>(2,4,d_output+10,d_time_stamp+20);
    cudaMemcpy(h_output, d_output, sizeof(double)*20, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time_stamp, d_time_stamp, sizeof(unsigned int)*40, cudaMemcpyDeviceToHost);
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess)
    {
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
    }
    printf("type\tserial\tnosync\ttile\ttilewarp\ttile_shuffle\tcoa\tcoawarp\tcoa_shuffle\tblock\tbasic\n");
    printf("result:\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
        h_output[0],
        h_output[1],
        h_output[2],
        h_output[3],
        h_output[4],
        h_output[5],
        h_output[6],
        h_output[7],
        h_output[8],
        h_output[9],
        h_output[10]);
    printf("time:\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
        h_time_stamp[1]-h_time_stamp[0],
        h_time_stamp[3]-h_time_stamp[2],
        h_time_stamp[5]-h_time_stamp[4],
        h_time_stamp[7]-h_time_stamp[6],
        h_time_stamp[9]-h_time_stamp[8],
        h_time_stamp[11]-h_time_stamp[10],
        h_time_stamp[13]-h_time_stamp[12],
        h_time_stamp[15]-h_time_stamp[14],
        h_time_stamp[17]-h_time_stamp[16],
        h_time_stamp[19]-h_time_stamp[18],
        h_time_stamp[21]-h_time_stamp[20]);
 }



