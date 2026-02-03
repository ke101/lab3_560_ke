#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


__global__ void rgb_to_gray_kernel(
    const unsigned char *rgb, unsigned char *gray, int w, int h, int c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = (y * w + x) * c;
    float r = rgb[idx + 0];
    float g = (c > 1) ? rgb[idx + 1] : r;
    float b = (c > 2) ? rgb[idx + 2] : r;
    float lum = 0.299f*r + 0.587f*g + 0.114f*b;
    gray[y * w + x] = (unsigned char)(lum + 0.5f);
}

// tile
template <int TILE, int MAX_K>
__global__ void conv_shared_kernel(
    const unsigned char *in, unsigned char *out,
    const float *kernel, int w, int h, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;
    int x = bx + tx;
    int y = by + ty;

    const int pad = (MAX_K - 1) / 2; 
    extern __shared__ unsigned char sdata[]; 

    int S = TILE + K - 1;
    int s_idx = (tx + (K-1)/2) + (ty + (K-1)/2) * S; 

    for (int sy = ty; sy < S; sy += blockDim.y) {
        for (int sx = tx; sx < S; sx += blockDim.x) {
            int gx = bx + (sx - (K-1)/2);
            int gy = by + (sy - (K-1)/2);
            gx = clamp_i(gx, 0, w - 1);
            gy = clamp_i(gy, 0, h - 1);
            sdata[sy * S + sx] = in[gy * w + gx];
        }
    }
    __syncthreads();

    if (x >= w || y >= h) return;

    float sum = 0.0f;
    int Sstride = S;
    int start_x = tx;
    int start_y = ty;
    int offx = (K - 1) / 2;
    int offy = (K - 1) / 2;
    int base_sx = tx + offx;
    int base_sy = ty + offy;

    for (int ky = 0; ky < K; ++ky) {
        int srow = base_sy + (ky - offy);
        const float *krow = kernel + ky * K;
        int srow_off = srow * Sstride;
        for (int kx = 0; kx < K; ++kx) {
            int scol = base_sx + (kx - offx);
            unsigned char v = sdata[srow_off + scol];
            sum += v * krow[kx];
        }
    }
    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;
    out[y * w + x] = (unsigned char)(sum + 0.5f);
}

// normalization
__global__ void normalize_apply_kernel(unsigned char *img, int w, int h, float mean, float stddev, float minv, float maxv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = w * h;
    if (idx >= N) return;
    float z = (img[idx] - mean) / stddev;
    float s = (z - minv) / (maxv - minv);
    if (s < 0.0f) s = 0.0f;
    if (s > 1.0f) s = 1.0f;
    img[idx] = (unsigned char)(s * 255.0f + 0.5f);
}

// helper function
static float *alloc_and_copy_kernel_to_device(const float *hkernel, int K)
{
    float *d_kernel = nullptr;
    CHECK_CUDA(cudaMalloc(&d_kernel, sizeof(float) * K * K));
    CHECK_CUDA(cudaMemcpy(d_kernel, hkernel, sizeof(float) * K * K, cudaMemcpyHostToDevice));
    return d_kernel;
}

static void make_laplacian_kernel(float *hkernel, int K)
{
    int N = K*K;
    for (int i=0;i<N;i++) hkernel[i] = -1.0f;
    hkernel[(K/2)*K + (K/2)] = (float)(N - 1);
}

static void make_box_kernel(float *hkernel, int K)
{
    int N = K*K;
    float v = 1.0f / (float)N;
    for (int i=0;i<N;i++) hkernel[i] = v;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("Usage: %s input.jpg outprefix filter N\n", argv[0]);
        printf(" filter: edge | resize | normalize\n");
        return 1;
    }

    const char *input = argv[1];
    const char *outprefix = argv[2];
    const char *filter = argv[3];
    int K = atoi(argv[4]);
    if (K < 1) K = 3;
    if ((K % 2) == 0) K += 1; 

    int w,h,c;
    unsigned char *h_rgb = stbi_load(input, &w, &h, &c, 0);
    if (!h_rgb) {
        fprintf(stderr, "Failed to load %s\n", input);
        return 2;
    }

    // if not M*M, crop to square
    if (w != h) {
        int M = w < h ? w : h;
        unsigned char *crop = (unsigned char*)malloc(M * M * c);
        int xoff = (w - M) / 2;
        int yoff = (h - M) / 2;
        for (int y0=0; y0<M; ++y0) {
            memcpy(crop + (size_t)y0 * M * c,
                   h_rgb + ((yoff + y0) * w + xoff) * c,
                   (size_t)M * c);
        }
        free(h_rgb);
        h_rgb = crop;
        w = h = M;
    }

    size_t rgb_bytes = (size_t)w * h * c * sizeof(unsigned char);
    size_t gray_bytes = (size_t)w * h * sizeof(unsigned char);

    unsigned char *d_rgb = nullptr, *d_gray=nullptr, *d_out=nullptr;
    CHECK_CUDA(cudaMalloc(&d_rgb, rgb_bytes));
    CHECK_CUDA(cudaMalloc(&d_gray, gray_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, gray_bytes));

    CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));
    dim3 blk(16,16);
    dim3 grid((w + blk.x -1)/blk.x, (h + blk.y -1)/blk.y);
    rgb_to_gray_kernel<<<grid, blk>>>(d_rgb, d_gray, w, h, c);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    float gpu_milliseconds = 0.0f;

    if (strcmp(filter,"edge")==0 || strcmp(filter,"resize")==0) {
        float *hkernel = (float*)malloc(sizeof(float) * K * K);
        if (strcmp(filter,"resize")==0) make_box_kernel(hkernel, K);
        else make_laplacian_kernel(hkernel, K);

        float *d_kernel = alloc_and_copy_kernel_to_device(hkernel, K);
        const int TILE = 16;
        int S = TILE + K - 1;
        size_t shm_bytes = (size_t)S * S * sizeof(unsigned char);

        dim3 grid2((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
        dim3 blk2(TILE, TILE);
        const int MAX_K = 31;
        if (K <= MAX_K) {
            CHECK_CUDA(cudaEventRecord(ev0));
            conv_shared_kernel<TILE, MAX_K><<<grid2, blk2, shm_bytes/sizeof(unsigned char)>>>(d_gray, d_out, d_kernel, w, h, K);
            CHECK_CUDA(cudaEventRecord(ev1));
            CHECK_CUDA(cudaEventSynchronize(ev1));
            CHECK_CUDA(cudaEventElapsedTime(&gpu_milliseconds, ev0, ev1));
        } else {
            fprintf("error");
            CHECK_CUDA(cudaMemcpy(d_out, d_gray, gray_bytes, cudaMemcpyDeviceToDevice));
        }
        // resizer
        if (strcmp(filter,"resize")==0) {
            int scale = K/2 >= 1 ? K/2 : 1;
            if (scale > 1) {
                int nw = w / scale;
                int nh = h / scale;
                if (nw < 1) nw = 1;
                if (nh < 1) nh = 1;
                unsigned char *htmp = (unsigned char*)malloc(gray_bytes);
                CHECK_CUDA(cudaMemcpy(htmp, d_out, gray_bytes, cudaMemcpyDeviceToHost));
                unsigned char *hdown = (unsigned char*)malloc((size_t)nw * nh);
                for (int yy=0; yy<nh; ++yy) {
                    for (int xx=0; xx<nw; ++xx) {
                        hdown[yy*nw + xx] = htmp[(yy*scale)*w + (xx*scale)];
                    }
                }
                char outname[512];
                snprintf(outname, sizeof(outname), "%s_%s_N%d.png", outprefix, filter, K);
                stbi_write_png(outname, nw, nh, 1, hdown, nw);
                free(htmp); free(hdown);
                printf("GPU_kernel_ms: %.3f\n", gpu_milliseconds);
                CHECK_CUDA(cudaFree(d_kernel));
                free(hkernel);
                CHECK_CUDA(cudaFree(d_rgb)); CHECK_CUDA(cudaFree(d_gray)); CHECK_CUDA(cudaFree(d_out));
                stbi_image_free(h_rgb);
                return 0;
            }
        }
        unsigned char *h_out = (unsigned char*)malloc(gray_bytes);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, gray_bytes, cudaMemcpyDeviceToHost));
        char outname[512];
        snprintf(outname, sizeof(outname), "%s_%s_N%d.png", outprefix, filter, K);
        stbi_write_png(outname, w, h, 1, h_out, w);
        free(h_out);
        CHECK_CUDA(cudaFree(d_kernel));
        free(hkernel);
    }
    else if (strcmp(filter,"normalize")==0) {
        unsigned char *hgray = (unsigned char*)malloc(gray_bytes);
        CHECK_CUDA(cudaMemcpy(hgray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
        double sum = 0.0;
        int Npix = w * h;
        for (int i=0;i<Npix;i++) sum += hgray[i];
        double mean = sum / Npix;
        double var = 0.0;
        for (int i=0;i<Npix;i++) { double d = hgray[i] - mean; var += d*d; }
        var /= Npix;
        double stddev = sqrt(var);
        if (stddev < 1e-6) stddev = 1.0;

        double minz = 1e9, maxz = -1e9;
        for (int i=0;i<Npix;i++) {
            double z = ((double)hgray[i] - mean) / stddev;
            if (z < minz) minz = z;
            if (z > maxz) maxz = z;
        }
        int threads = 256;
        int blocks = (Npix + threads - 1) / threads;
        CHECK_CUDA(cudaEventRecord(ev0));
        normalize_apply_kernel<<<blocks, threads>>>(d_gray, w, h, (float)mean, (float)stddev, (float)minz, (float)maxz);
        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        CHECK_CUDA(cudaEventElapsedTime(&gpu_milliseconds, ev0, ev1));

        unsigned char *h_out = (unsigned char*)malloc(gray_bytes);
        CHECK_CUDA(cudaMemcpy(h_out, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
        char outname[512];
        snprintf(outname, sizeof(outname), "%s_%s_N%d.png", outprefix, filter, K);
        stbi_write_png(outname, w, h, 1, h_out, w);
        free(h_out);
        free(hgray);
    }
    else {
        fprintf(stderr, "Unknown filter '%s'\n", filter);
    }
    printf("GPU_kernel_ms: %.3f\n", gpu_milliseconds);
    CHECK_CUDA(cudaFree(d_rgb));
    CHECK_CUDA(cudaFree(d_gray));
    CHECK_CUDA(cudaFree(d_out));
    stbi_image_free(h_rgb);
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    return 0;
}
