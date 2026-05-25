/*
 * cuda_modifier_real_kernels.cu
 *
 * Tests whether user-accessible cache modifier selection improves performance
 * in realistic (non-microbenchmark) CUDA kernels, in contrast to Triton
 * where modifiers are either structurally blocked (LDGSTS) or already optimal (.nc).
 *
 * Three access patterns:
 *   A) Streaming scan      — read large array once        → expect .cs/.nc wins
 *   B) Temporal reuse      — read small buffer N times    → expect .ca wins
 *   C) Gather (random)     — random-indexed reads         → expect .cg wins (no L1 pollute)
 *
 * Each pattern has 4 kernels: .ca / .cg / .nc / .cs
 * Measured with clock64, 120 trials, reported as median GB/s.
 * Saves results to artifacts/runs/cuda_modifier_real_kernels/results.json
 */

#include <cuda_runtime.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NTRIALS   120
#define NTHREADS  256
#define REPS      8

/* ── LDG helpers with explicit cache modifiers ─────────────────────────── */
__device__ __forceinline__ float ldg_ca(const float *p) {
    float v;
    asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ldg_cg(const float *p) {
    float v;
    asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ldg_nc(const float *p) {
    float v;
    asm volatile("ld.global.nc.f32 %0,[%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ldg_cs(const float *p) {
    float v;
    asm volatile("ld.global.cs.f32 %0,[%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

/* ── Pattern A: Streaming scan ─────────────────────────────────────────── */
/* Each thread reads a strided element across the full array.              */
/* Access is sequential → no temporal reuse within a kernel launch.        */
/* Optimal policy: .cs (streaming, evict-first) or .nc (read-only cache)   */
#define STREAM_KERNEL(NAME, LDG_FN)                                        \
__global__ void stream_##NAME(const float * __restrict__ src,              \
                               float * __restrict__ dst,                    \
                               int n, uint64_t *t_out) {                    \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                       \
    int stride = gridDim.x * blockDim.x;                                   \
    uint64_t t0, t1;                                                        \
    float acc = 0.0f;                                                       \
    /* prime */                                                             \
    for (int i = tid; i < n; i += stride) acc += LDG_FN(src + i);         \
    if (acc == -9999.f) dst[0] = acc;                                       \
    __syncthreads();                                                        \
    acc = 0.0f;                                                             \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t0) :: "memory");         \
    for (int r = 0; r < REPS; r++)                                         \
        for (int i = tid; i < n; i += stride) acc += LDG_FN(src + i);    \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t1) :: "memory");         \
    if (threadIdx.x == 0 && blockIdx.x == 0) *t_out = t1 - t0;           \
    if (acc == -9999.f) dst[0] = acc;                                      \
}

STREAM_KERNEL(ca, ldg_ca)
STREAM_KERNEL(cg, ldg_cg)
STREAM_KERNEL(nc, ldg_nc)
STREAM_KERNEL(cs, ldg_cs)

/* ── Pattern B: Temporal reuse ─────────────────────────────────────────── */
/* All threads read from small buffer (fits in L1) N_REUSE times.          */
/* High temporal reuse → .ca (keep in L1) should win.                      */
#define REUSE_KERNEL(NAME, LDG_FN)                                         \
__global__ void reuse_##NAME(const float * __restrict__ src,               \
                              float * __restrict__ dst,                     \
                              int buf_len, int n_reuse, uint64_t *t_out) { \
    int lane = threadIdx.x % 32;                                            \
    uint64_t t0, t1;                                                        \
    float acc = 0.0f;                                                       \
    for (int k = lane; k < buf_len; k += 32) acc += LDG_FN(src + k);     \
    if (acc == -9999.f) dst[0] = acc;                                       \
    acc = 0.0f;                                                             \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t0) :: "memory");         \
    for (int r = 0; r < n_reuse; r++)                                      \
        for (int k = lane; k < buf_len; k += 32) acc += LDG_FN(src + k); \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t1) :: "memory");         \
    if (threadIdx.x == 0 && blockIdx.x == 0) *t_out = t1 - t0;           \
    if (acc == -9999.f) dst[0] = acc;                                      \
}

REUSE_KERNEL(ca, ldg_ca)
REUSE_KERNEL(cg, ldg_cg)
REUSE_KERNEL(nc, ldg_nc)
REUSE_KERNEL(cs, ldg_cs)

/* ── Pattern C: Gather (random index) ──────────────────────────────────── */
/* Threads read from random indices across a large table.                   */
/* No spatial/temporal locality → .cg (bypass L1, go to L2) should win:   */
/* avoids polluting L1 with data that won't be reused.                     */
#define GATHER_KERNEL(NAME, LDG_FN)                                        \
__global__ void gather_##NAME(const float * __restrict__ src,              \
                               const int  * __restrict__ idx,              \
                               float * __restrict__ dst,                   \
                               int n_gather, int src_len, uint64_t *t_out){\
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                      \
    int stride = gridDim.x * blockDim.x;                                   \
    uint64_t t0, t1;                                                        \
    float acc = 0.0f;                                                       \
    for (int i = tid; i < n_gather; i += stride)                           \
        acc += LDG_FN(src + (idx[i] % src_len));                           \
    if (acc == -9999.f) dst[0] = acc;                                      \
    acc = 0.0f;                                                             \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t0) :: "memory");         \
    for (int r = 0; r < REPS; r++)                                         \
        for (int i = tid; i < n_gather; i += stride)                       \
            acc += LDG_FN(src + (idx[i] % src_len));                       \
    asm volatile("mov.u64 %0, %%clock64;": "=l"(t1) :: "memory");         \
    if (threadIdx.x == 0 && blockIdx.x == 0) *t_out = t1 - t0;           \
    if (acc == -9999.f) dst[0] = acc;                                      \
}

GATHER_KERNEL(ca, ldg_ca)
GATHER_KERNEL(cg, ldg_cg)
GATHER_KERNEL(nc, ldg_nc)
GATHER_KERNEL(cs, ldg_cs)

/* ── Utilities ──────────────────────────────────────────────────────────── */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x>y)-(x<y);
}
static uint64_t median64(uint64_t *a, int n) {
    qsort(a, n, sizeof(uint64_t), cmp_u64);
    return a[n/2];
}
/* cycles → GB/s: total_bytes accessed in REPS passes */
static double to_gbs(uint64_t cy, long long bytes, int reps, double freq_ghz) {
    double ns = cy / freq_ghz;
    double total_bytes = (double)bytes * reps;
    return total_bytes / ns; /* GB/s = bytes/ns */
}

/* ── Main ───────────────────────────────────────────────────────────────── */
int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double freq_ghz = prop.clockRate / 1e6;
    printf("Device: %s  freq=%.3f GHz\n", prop.name, freq_ghz);

    /* ── Sizes ── */
    /* Pattern A: 16 MB array → well beyond L2 (3 MB), pure DRAM streaming */
    int stream_n = 16*1024*1024/4; /* floats */
    int nblocks_stream = 128;

    /* Pattern B: 16 KB buffer → fits entirely in L1 (128 KB SM86) */
    int reuse_buf = 16*1024/4; /* 4096 floats */
    int n_reuse   = 256;

    /* Pattern C: gather 1M indices from 4 MB table (> L1, < L2) */
    int gather_n   = 1*1024*1024;
    int gather_src = 4*1024*1024/4;
    int nblocks_gather = 64;

    /* Allocate device buffers */
    float *d_src_stream, *d_src_reuse, *d_src_gather;
    int   *d_idx;
    float *d_dst;
    uint64_t *d_t;

    cudaMalloc(&d_src_stream, stream_n   * sizeof(float));
    cudaMalloc(&d_src_reuse,  reuse_buf  * sizeof(float));
    cudaMalloc(&d_src_gather, gather_src * sizeof(float));
    cudaMalloc(&d_idx,        gather_n   * sizeof(int));
    cudaMalloc(&d_dst, 4 * sizeof(float));
    cudaMalloc(&d_t,  sizeof(uint64_t));

    /* Init host arrays */
    float *h_src = (float*)malloc(stream_n * sizeof(float));
    int   *h_idx = (int*)  malloc(gather_n * sizeof(int));
    for (int i = 0; i < stream_n; i++) h_src[i] = 1.0f;
    srand(42);
    for (int i = 0; i < gather_n; i++) h_idx[i] = rand();
    cudaMemcpy(d_src_stream, h_src, stream_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_reuse,  h_src, reuse_buf * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_gather, h_src, gather_src * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, gather_n * sizeof(int), cudaMemcpyHostToDevice);
    free(h_src); free(h_idx);

    uint64_t samples[NTRIALS];
    const char *mods[] = {"ca","cg","nc","cs"};

    /* JSON output */
    mkdir("artifacts/runs/cuda_modifier_real_kernels", 0755);
    FILE *fp = fopen("artifacts/runs/cuda_modifier_real_kernels/results.json","w");
    fprintf(fp,"{\n  \"device\":\"%s\",\n  \"freq_ghz\":%.4f,\n",prop.name,freq_ghz);
    fprintf(fp,"  \"ntrials\":%d, \"reps\":%d,\n",NTRIALS,REPS);

    /* ══ Pattern A: Streaming ══ */
    printf("\n=== Pattern A: Streaming scan  (16 MB, pure DRAM) ===\n");
    printf("  Expected winner: .cs or .nc  (no temporal reuse → L1 pollution hurts .ca)\n");
    printf("  %-6s  %8s  %6s\n","mod","GB/s","vs .ca");
    fprintf(fp,"  \"streaming_16mb\":{\n");

    long long stream_bytes = (long long)stream_n * 4;
    double baseline_stream = 0;
    for (int m = 0; m < 4; m++) {
        for (int t = 0; t < NTRIALS; t++) {
            if (m==0) stream_ca<<<nblocks_stream,NTHREADS>>>(d_src_stream,d_dst,stream_n,d_t);
            if (m==1) stream_cg<<<nblocks_stream,NTHREADS>>>(d_src_stream,d_dst,stream_n,d_t);
            if (m==2) stream_nc<<<nblocks_stream,NTHREADS>>>(d_src_stream,d_dst,stream_n,d_t);
            if (m==3) stream_cs<<<nblocks_stream,NTHREADS>>>(d_src_stream,d_dst,stream_n,d_t);
            cudaDeviceSynchronize();
            cudaMemcpy(&samples[t], d_t, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }
        uint64_t med = median64(samples, NTRIALS);
        double gbs = to_gbs(med, stream_bytes, REPS, freq_ghz);
        if (m==0) baseline_stream = gbs;
        double pct = (gbs / baseline_stream - 1.0) * 100.0;
        printf("  .%-5s  %8.2f  %+6.1f%%\n", mods[m], gbs, pct);
        fprintf(fp,"    \"%s\":{\"med_cy\":%llu,\"gbs\":%.3f}%s\n",
                mods[m],(unsigned long long)med,gbs,m<3?",":"");
    }
    fprintf(fp,"  },\n");

    /* ══ Pattern B: Temporal reuse ══ */
    printf("\n=== Pattern B: Temporal reuse  (16 KB buf, 256x reload, sub-L1) ===\n");
    printf("  Expected winner: .ca  (hot data in L1, repeated access)\n");
    printf("  %-6s  %8s  %6s\n","mod","GB/s","vs .ca");
    fprintf(fp,"  \"temporal_reuse_16kb\":{\n");

    long long reuse_bytes = (long long)reuse_buf * 4;
    double baseline_reuse = 0;
    for (int m = 0; m < 4; m++) {
        for (int t = 0; t < NTRIALS; t++) {
            if (m==0) reuse_ca<<<1,32>>>(d_src_reuse,d_dst,reuse_buf,n_reuse,d_t);
            if (m==1) reuse_cg<<<1,32>>>(d_src_reuse,d_dst,reuse_buf,n_reuse,d_t);
            if (m==2) reuse_nc<<<1,32>>>(d_src_reuse,d_dst,reuse_buf,n_reuse,d_t);
            if (m==3) reuse_cs<<<1,32>>>(d_src_reuse,d_dst,reuse_buf,n_reuse,d_t);
            cudaDeviceSynchronize();
            cudaMemcpy(&samples[t], d_t, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }
        uint64_t med = median64(samples, NTRIALS);
        double gbs = to_gbs(med, reuse_bytes, n_reuse, freq_ghz);
        if (m==0) baseline_reuse = gbs;
        double pct = (gbs / baseline_reuse - 1.0) * 100.0;
        printf("  .%-5s  %8.2f  %+6.1f%%\n", mods[m], gbs, pct);
        fprintf(fp,"    \"%s\":{\"med_cy\":%llu,\"gbs\":%.3f}%s\n",
                mods[m],(unsigned long long)med,gbs,m<3?",":"");
    }
    fprintf(fp,"  },\n");

    /* ══ Pattern C: Random gather ══ */
    printf("\n=== Pattern C: Random gather  (1M indices, 4 MB table, L2-bound) ===\n");
    printf("  Expected winner: .cg  (no reuse, L1 pollution hurts .ca/.nc)\n");
    printf("  %-6s  %8s  %6s\n","mod","GB/s","vs .ca");
    fprintf(fp,"  \"random_gather_4mb\":{\n");

    long long gather_bytes = (long long)gather_n * 4;
    double baseline_gather = 0;
    for (int m = 0; m < 4; m++) {
        for (int t = 0; t < NTRIALS; t++) {
            if (m==0) gather_ca<<<nblocks_gather,NTHREADS>>>(d_src_gather,d_idx,d_dst,gather_n,gather_src,d_t);
            if (m==1) gather_cg<<<nblocks_gather,NTHREADS>>>(d_src_gather,d_idx,d_dst,gather_n,gather_src,d_t);
            if (m==2) gather_nc<<<nblocks_gather,NTHREADS>>>(d_src_gather,d_idx,d_dst,gather_n,gather_src,d_t);
            if (m==3) gather_cs<<<nblocks_gather,NTHREADS>>>(d_src_gather,d_idx,d_dst,gather_n,gather_src,d_t);
            cudaDeviceSynchronize();
            cudaMemcpy(&samples[t], d_t, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        }
        uint64_t med = median64(samples, NTRIALS);
        double gbs = to_gbs(med, gather_bytes, REPS, freq_ghz);
        if (m==0) baseline_gather = gbs;
        double pct = (gbs / baseline_gather - 1.0) * 100.0;
        printf("  .%-5s  %8.2f  %+6.1f%%\n", mods[m], gbs, pct);
        fprintf(fp,"    \"%s\":{\"med_cy\":%llu,\"gbs\":%.3f}%s\n",
                mods[m],(unsigned long long)med,gbs,m<3?",":"");
    }
    fprintf(fp,"  }\n}\n");
    fclose(fp);

    printf("\nSaved → artifacts/runs/cuda_modifier_real_kernels/results.json\n");

    cudaFree(d_src_stream); cudaFree(d_src_reuse); cudaFree(d_src_gather);
    cudaFree(d_idx); cudaFree(d_dst); cudaFree(d_t);
    return 0;
}
