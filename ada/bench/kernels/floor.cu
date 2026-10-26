// The per-kernel floor in a CUDA graph on this GPU: chains of dependent tiny kernels, by grid size, and a 3.3 MB read.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void empty(int* p) { if (p && threadIdx.x == 9999) *p = 1; }
__global__ void readk(const int4* w, int n, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { int4 v = __ldg(w + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345) *out = 1;
}
int main() {
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    int* o; cudaMalloc(&o, 4);
    const int L = 48;
    int4* w; const size_t bytes = 3276800; cudaMalloc(&w, bytes * L); cudaMemset(w, 1, bytes * L);
    auto run = [&](const char* name, auto body) {
        cudaGraph_t g; cudaGraphExec_t x;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        for (int r = 0; r < 20 * L; ++r) body(r % L);
        cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&x, g, 0);
        cudaGraphLaunch(x, s); cudaStreamSynchronize(s);
        float best = 1e9;
        for (int k = 0; k < 5; ++k) { cudaEventRecord(a, s); cudaGraphLaunch(x, s); cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best; }
        printf("%-40s %7.2f us a kernel\n", name, best * 1e3 / (20 * L));
    };
    run("empty <<<1, 32>>>", [&](int) { empty<<<1, 32, 0, s>>>(o); });
    run("empty <<<160, 256>>>", [&](int) { empty<<<160, 256, 0, s>>>(o); });
    run("empty <<<1640, 256>>>", [&](int) { empty<<<1640, 256, 0, s>>>(o); });
    for (int gsz : {80, 160, 328, 656})
        for (int bs : {256, 512}) {
            char n[64]; snprintf(n, 64, "read 3.3 MB <<<%d, %d>>>", gsz, bs);
            run(n, [&](int l) { readk<<<gsz, bs, 0, s>>>(w + l * (bytes / 16), (int) (bytes / 16), o); });
        }
    return 0;
}
