#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

__global__ void rgb_to_gray(
    const unsigned char* rgb,
    unsigned char* gray,
    int w,int h,int c)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=w || y>=h) return;
    int idx=(y*w+x)*c;
    float r=rgb[idx], g=c>1?rgb[idx+1]:r, b=c>2?rgb[idx+2]:r;
    gray[y*w+x]=(unsigned char)(0.299f*r+0.587f*g+0.114f*b);
}

template<int TILE,int MAXK>
__global__ void conv_shared(
    const unsigned char* in,
    unsigned char* out,
    const float* k,
    int w,int h,int K)
{
    int tx=threadIdx.x, ty=threadIdx.y;
    int x=blockIdx.x*TILE+tx;
    int y=blockIdx.y*TILE+ty;

    int S=TILE+K-1;
    extern __shared__ unsigned char s[];

    for(int sy=ty;sy<S;sy+=blockDim.y)
        for(int sx=tx;sx<S;sx+=blockDim.x){
            int gx=clampi(blockIdx.x*TILE+sx-(K-1)/2,0,w-1);
            int gy=clampi(blockIdx.y*TILE+sy-(K-1)/2,0,h-1);
            s[sy*S+sx]=in[gy*w+gx];
        }
    __syncthreads();
    if(x>=w||y>=h) return;

    float sum=0;
    int off=(K-1)/2;
    int bx=tx+off, by=ty+off;
    for(int ky=0;ky<K;ky++)
        for(int kx=0;kx<K;kx++)
            sum+=s[(by+ky-off)*S+(bx+kx-off)]*k[ky*K+kx];

    sum=fminf(fmaxf(sum,0.0f),255.0f);
    out[y*w+x]=(unsigned char)(sum);
}
float process_image(const char* in,const char* outp,const char* filter,int K)
{
    int w,h,c;
    unsigned char* h_rgb=stbi_load(in,&w,&h,&c,0);
    if(!h_rgb) return -1.0f;
    size_t rgbB=w*h*c, grayB=w*h;
    unsigned char *d_rgb,*d_g,*d_o;
    CHECK(cudaMalloc(&d_rgb,rgbB));
    CHECK(cudaMalloc(&d_g,grayB));
    CHECK(cudaMalloc(&d_o,grayB));
    CHECK(cudaMemcpy(d_rgb,h_rgb,rgbB,cudaMemcpyHostToDevice));
    dim3 B(16,16), G((w+15)/16,(h+15)/16);
    rgb_to_gray<<<G,B>>>(d_rgb,d_g,w,h,c);
    cudaDeviceSynchronize();

    float* h_k=(float*)malloc(K*K*sizeof(float));
    for(int i=0;i<K*K;i++) h_k[i]=(strcmp(filter,"resize")==0)?1.0f/(K*K):-1.0f;
    if(strcmp(filter,"edge")==0)
        h_k[(K/2)*K+(K/2)]=(float)(K*K-1);

    float* d_k;
    cudaMalloc(&d_k,K*K*sizeof(float));
    cudaMemcpy(d_k,h_k,K*K*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t e0,e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    int TILE=16;
    int S=TILE+K-1;
    size_t sh=S*S*sizeof(unsigned char);
    cudaEventRecord(e0);
    conv_shared<16,31><<<G,B,sh>>>(d_g,d_o,d_k,w,h,K);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms,e0,e1);

    unsigned char* h_out=(unsigned char*)malloc(grayB);
    cudaMemcpy(h_out,d_o,grayB,cudaMemcpyDeviceToHost);
    char name[256];
    stbi_write_png(name,w,h,1,h_out,w);
    cudaFree(d_rgb); cudaFree(d_g); cudaFree(d_o); cudaFree(d_k);
    stbi_image_free(h_rgb); free(h_out); free(h_k);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

