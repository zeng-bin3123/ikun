#ifndef INFCCL_KERNELS_H_
#define INFCCL_KERNELS_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define INFCCL_THREADS 256
#define INFCCL_BLOCKS(n) (((n)+INFCCL_THREADS-1)/INFCCL_THREADS)

typedef uint64_t Pack64;

template<typename T> struct FuncSum  { __device__ __forceinline__ T operator()(T a,T b)const{return a+b;} };
template<typename T> struct FuncProd { __device__ __forceinline__ T operator()(T a,T b)const{return a*b;} };
template<typename T> struct FuncMax  { __device__ __forceinline__ T operator()(T a,T b)const{return a>b?a:b;} };
template<typename T> struct FuncMin  { __device__ __forceinline__ T operator()(T a,T b)const{return a<b?a:b;} };
template<> struct FuncSum<half>  { __device__ __forceinline__ half operator()(half a,half b)const{return __hadd(a,b);} };
template<> struct FuncProd<half> { __device__ __forceinline__ half operator()(half a,half b)const{return __hmul(a,b);} };

template<class F,typename T> struct PackOp;
template<class F> struct PackOp<F,float> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;float f[2];}ua,ub,ur;ua.p=a;ub.p=b;
        ur.f[0]=F()(ua.f[0],ub.f[0]);ur.f[1]=F()(ua.f[1],ub.f[1]);return ur.p;}
};
template<class F> struct PackOp<F,half> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;half h[4];}ua,ub,ur;ua.p=a;ub.p=b;
        ur.h[0]=F()(ua.h[0],ub.h[0]);ur.h[1]=F()(ua.h[1],ub.h[1]);
        ur.h[2]=F()(ua.h[2],ub.h[2]);ur.h[3]=F()(ua.h[3],ub.h[3]);return ur.p;}
};
template<class F> struct PackOp<F,int> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;int i[2];}ua,ub,ur;ua.p=a;ub.p=b;
        ur.i[0]=F()(ua.i[0],ub.i[0]);ur.i[1]=F()(ua.i[1],ub.i[1]);return ur.p;}
};
template<class F> struct PackOp<F,char> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;char c[8];}ua,ub,ur;ua.p=a;ub.p=b;
        #pragma unroll
        for(int i=0;i<8;i++)ur.c[i]=F()(ua.c[i],ub.c[i]);return ur.p;}
};
template<class F> struct PackOp<F,long long> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;long long l;}ua,ub,ur;ua.p=a;ub.p=b;ur.l=F()(ua.l,ub.l);return ur.p;}
};
template<class F> struct PackOp<F,unsigned long long> {
    __device__ __forceinline__ Pack64 operator()(Pack64 a,Pack64 b)const{
        union{Pack64 p;unsigned long long l;}ua,ub,ur;ua.p=a;ub.p=b;ur.l=F()(ua.l,ub.l);return ur.p;}
};

template<typename T,class F>
__global__ void reduce_local_kernel(T*__restrict__ dst,const T*__restrict__ staged,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int epp=sizeof(Pack64)/sizeof(T);
    int np=n/epp;
    if(tid<np){
        Pack64 a=((Pack64*)dst)[tid];
        Pack64 b=((const Pack64*)staged)[tid];
        ((Pack64*)dst)[tid]=PackOp<F,T>()(a,b);
    }
    int ts=np*epp;
    int tt=tid-np;
    if(tt>=0&&tt<(n-ts))
        dst[ts+tt]=F()(dst[ts+tt],staged[ts+tt]);
}

#endif
