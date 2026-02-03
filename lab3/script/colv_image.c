#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static inline int clamp(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

// read image
unsigned char *load_and_to_gray(const char *filename, int *width, int *height) {
    int n;
    unsigned char *data = stbi_load(filename, width, height, &n, 0);
    if (!data) {
        fprintf(stderr, "ERROR: failed to load %s\n", filename);
        return NULL;
    }
    int w = *width, h = *height;
    unsigned char *gray = (unsigned char*)malloc(w * h);
    if (!gray) { stbi_image_free(data); return NULL; }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * n;
            float r = data[idx + 0];
            float g = (n > 1) ? data[idx + 1] : data[idx + 0];
            float b = (n > 2) ? data[idx + 2] : data[idx + 0];
            // luminosity method
            float lum = 0.299f * r + 0.587f * g + 0.114f * b;
            gray[y * w + x] = (unsigned char)clamp((int)roundf(lum), 0, 255);
        }
    }
    stbi_image_free(data);
    return gray;
}

// change to grayscale
int write_gray_png(const char *filename, unsigned char *gray, int w, int h) {
    return stbi_write_png(filename, w, h, 1, gray, w) ? 0 : -1;
}

// generic convolution (kernel is floats of size k x k)
unsigned char *convolve_gray(unsigned char *in, int w, int h, const float *kernel, int k) {
    int pad = k / 2;
    unsigned char *out = (unsigned char*)malloc(w * h);
    if (!out) return NULL;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float accum = 0.0f;
            for (int ky = 0; ky < k; ++ky) {
                for (int kx = 0; kx < k; ++kx) {
                    int ix = x + kx - pad;
                    int iy = y + ky - pad;
                    ix = clamp(ix, 0, w - 1);
                    iy = clamp(iy, 0, h - 1);
                    float v = in[iy * w + ix];
                    accum += v * kernel[ky * k + kx];
                }
            }
            int val = (int)roundf(accum);
            out[y * w + x] = (unsigned char)clamp(val, 0, 255);
        }
    }
    return out;
}

// mean and stv computation for normalization
void compute_mean_std(unsigned char *in, int w, int h, double *mean, double *stddev) {
    long long sum = 0;
    int N = w * h;
    for (int i = 0; i < N; ++i) sum += in[i];
    *mean = (double)sum / N;
    double var = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = in[i] - *mean;
        var += d * d;
    }
    var /= N;
    *stddev = sqrt(var);
}

// normalization
unsigned char *normalize_gray(unsigned char *in, int w, int h) {
    double mean, stddev;
    compute_mean_std(in, w, h, &mean, &stddev);
    if (stddev < 1e-6) stddev = 1.0;
    unsigned char *out = (unsigned char*)malloc(w * h);
    if (!out) return NULL;
    double minv = 1e9, maxv = -1e9;
    int N = w * h;
    double *zbuf = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i) {
        double z = (in[i] - mean) / stddev;
        zbuf[i] = z;
        if (z < minv) minv = z;
        if (z > maxv) maxv = z;
    }
    double range = maxv - minv;
    if (range < 1e-9) range = 1.0;
    for (int i = 0; i < N; ++i) {
        double s = (zbuf[i] - minv) / range;
        out[i] = (unsigned char)clamp((int)roundf((float)(s * 255.0)), 0, 255);
    }
    free(zbuf);
    return out;
}

// build simple Laplacian-like kernel
float *make_laplacian_kernel(int k) {
    int N = k*k;
    float *kernel = (float*)malloc(sizeof(float) * N);
    if (!kernel) return NULL;
    for (int i = 0; i < N; ++i) kernel[i] = -1.0f;
    kernel[(k/2)*k + (k/2)] = (float)(N - 1);
    return kernel;
}

float *make_box_kernel(int k) {
    int N = k*k;
    float *kernel = (float*)malloc(sizeof(float) * N);
    if (!kernel) return NULL;
    float val = 1.0f / N;
    for (int i = 0; i < N; ++i) kernel[i] = val;
    return kernel;
}

// resize
unsigned char *box_filter_and_downsample(unsigned char *in, int w, int h, int k, int scale, int *out_w, int *out_h) {
    float *kernel = make_box_kernel(k);
    unsigned char *blur = convolve_gray(in, w, h, kernel, k);
    free(kernel);
    if (!blur) return NULL;
    if (scale <= 1) {
        *out_w = w; *out_h = h;
        return blur; 
    }
    int nw = w / scale;
    int nh = h / scale;
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    unsigned char *out = (unsigned char*)malloc(nw * nh);
    for (int y = 0; y < nh; ++y) {
        for (int x = 0; x < nw; ++x) {
            out[y * nw + x] = blur[(y * scale) * w + (x * scale)];
        }
    }
    free(blur);
    *out_w = nw; *out_h = nh;
    return out;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }
    const char *infile = argv[1];
    const char *outprefix = argv[2];
    const char *filter = argv[3];
    int N = atoi(argv[4]);
    if (N < 1) N = 3;
    int opt_scale = 0;
    if (argc >= 6) opt_scale = atoi(argv[5]);

    int w, h;
    unsigned char *gray = load_and_to_gray(infile, &w, &h);
    if (!gray) return 2;
    if (w != h) {
        int M = w < h ? w : h;
        unsigned char *crop = (unsigned char*)malloc(M * M);
        int xoff = (w - M) / 2;
        int yoff = (h - M) / 2;
        for (int y = 0; y < M; ++y) {
            memcpy(crop + y * M, gray + (y + yoff) * w + xoff, M);
        }
        free(gray);
        gray = crop;
        w = h = M;
    }
    // convolution
    unsigned char *outimg = NULL;
    int outw = w, outh = h;
    clock_t t0 = clock();
    if (strcmp(filter, "edge") == 0) {
        if (N == 3) {
            float gx[9] = {-1,0,1,-2,0,2,-1,0,1};
            float gy[9] = {-1,-2,-1,0,0,0,1,2,1};
            signed short *tmpx = (signed short*)malloc(sizeof(signed short) * w * h);
            signed short *tmpy = (signed short*)malloc(sizeof(signed short) * w * h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int sumx = 0, sumy = 0;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int ix = clamp(x + kx, 0, w - 1);
                            int iy = clamp(y + ky, 0, h - 1);
                            int kval = gray[iy * w + ix];
                            int kidx = (ky+1)*3 + (kx+1);
                            sumx += kval * (int)gx[kidx];
                            sumy += kval * (int)gy[kidx];
                        }
                    }
                    tmpx[y*w + x] = (signed short)sumx;
                    tmpy[y*w + x] = (signed short)sumy;
                }
            }
            outimg = (unsigned char*)malloc(w*h);
            for (int i = 0; i < w*h; ++i) {
                float mag = sqrtf((float)(tmpx[i]*tmpx[i] + tmpy[i]*tmpy[i]));
                int v = (int)roundf(mag);
                if (v > 255) v = 255;
                outimg[i] = (unsigned char)v;
            }
            free(tmpx); free(tmpy);
        } else {
            float *kernel = make_laplacian_kernel(N);
            unsigned char *conv = convolve_gray(gray, w, h, kernel, N);
            free(kernel);
            outimg = conv;
        }
    } else if (strcmp(filter, "resize") == 0) {
        int scale = opt_scale > 0 ? opt_scale : (N / 2 >= 1 ? N / 2 : 1);
        outimg = box_filter_and_downsample(gray, w, h, N, scale, &outw, &outh);
    } else if (strcmp(filter, "normalize") == 0) {
        outimg = normalize_gray(gray, w, h);
    } else {
        //   fprintf(stderr, "Unknown filter: %s\n", filter);
        free(gray);
        return 3;
    }
    clock_t t1 = clock();
    double seconds = (double)(t1 - t0) / CLOCKS_PER_SEC;
    char outname[1024];
    snprintf(outname, sizeof(outname), "%s_%s_N%d.png", outprefix, filter, N);
    if (outimg) {
        if (write_gray_png(outname, outimg, outw, outh) != 0) {
            // fprintf(stderr, "error", outname);
        } else {
            printf("wrote %s (size %dx%d) in %.6f s\n", outname, outw, outh, seconds);
        }
        free(outimg);
    }
    free(gray);
    return 0;
}
