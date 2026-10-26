// Decode's latency-bound kernels alone, at the model's shapes, on synthetic weights: no spin kernels, so Nsight Compute
// can profile it (the engine's decode deadlocks under ncu's serialized replay).  Each op runs over 48 layers' worth of
// distinct weights (none left in L2 from the layer before, as in a window), captured in a graph like the verifier's.
//   bench/gk [T=4] [reps=20]          prints us per op (a layer's launch), per op alone and as the HC read + router chain
// Build: bench/gk.sh
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, NE = 512, HV = 48, L = 48;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

std::mt19937 rng(1);
template <class T> T* dev_fill(size_t n, std::function<T()> f) {
    std::vector<T> h(n);
    for (auto& v : h) v = f();
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> T* dev_zero(size_t n) {
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
uint16_t f16(float f) { __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float unif(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }

struct Layer {
    int8_t *qd, *qu;
    uint16_t *sd, *su, *wi, *wr, *wa, *wb;
    float *wn, *dt, *sa;
};

}  // namespace

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 4;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    std::vector<Layer> ly(L);
    for (Layer& l : ly) {
        l.qd = dev_fill<int8_t>((size_t) LR * D, [] { return (int8_t) (rng() % 255 - 127); });
        l.qu = dev_fill<int8_t>((size_t) D * LR, [] { return (int8_t) (rng() % 255 - 127); });
        l.sd = dev_fill<uint16_t>((size_t) LR * D / 32, [] { return f16(1e-3f); });
        l.su = dev_fill<uint16_t>((size_t) D * LR / 32, [] { return f16(1e-3f); });
        l.wi = dev_fill<uint16_t>((size_t) HC * D, [] { return bf16(unif(0.02f)); });
        l.wn = dev_fill<float>(D, [] { return 1.0f + unif(0.1f); });
        l.wr = dev_fill<uint16_t>((size_t) NE * N, [] { return bf16(unif(0.05f)); });
        l.wa = dev_fill<uint16_t>((size_t) HV * N, [] { return bf16(unif(0.05f)); });
        l.wb = dev_fill<uint16_t>((size_t) HV * N, [] { return bf16(unif(0.05f)); });
        l.dt = dev_fill<float>(HV, [] { return unif(1.0f); });
        l.sa = dev_fill<float>(HV, [] { return -1.0f + unif(0.5f); });
    }
    float* R = dev_fill<float>((size_t) 8 * D, [] { return unif(1.0f); });
    float* bo = dev_fill<float>((size_t) 8 * N, [] { return unif(0.1f); });
    float* inj_prev = dev_fill<float>((size_t) 8 * HC, [] { return unif(1.0f); });
    float* inj_out = dev_zero<float>(8 * HC);
    float* lo = dev_zero<float>(8 * LR);
    float* rs = dev_zero<float>(8 * HC);
    float* mixed = dev_fill<float>((size_t) 8 * N, [] { return unif(1.0f); });
    float* scratch = dev_zero<float>(fused_gr_scratch_bytes() / 4 + 1);
    float* logits = dev_zero<float>(8 * NE);
    int32_t* ids = dev_zero<int32_t>(8 * 10);
    float* wts = dev_zero<float>(8 * 10);
    uint8_t* xq1 = dev_zero<uint8_t>(8 * (N / 32) * 36);
    uint32_t* seq = dev_zero<uint32_t>(1);
    unsigned* counter = dev_zero<unsigned>(64);
    float* gate = dev_zero<float>(8 * HV);
    float* beta = dev_zero<float>(8 * HV);

    auto hc_read = [&](const Layer& l, cudaStream_t s) {
        FusedGrArgs fa[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            FusedGrArgs& a = fa[t];
            a.R = R + (size_t) t * D; a.R_out = R + (size_t) t * D; a.apply = true;
            a.bo_prev = bo + (size_t) t * N; a.inj_prev = inj_prev + t * HC;
            a.w_norm = l.wn; a.w_inject = l.wi; a.q_down = l.qd; a.s_down = l.sd; a.q_up = l.qu; a.s_up = l.su;
            a.lo = lo + t * LR; a.rs = rs + t * HC; a.inject_out = inj_out + t * HC; a.mixed = mixed + (size_t) t * N;
        }
        fused_gr_read_multi(fa, T, scratch, s);
    };
    auto router = [&](const Layer& l, cudaStream_t s) {
        VerifyRouterArgs a;
        a.x = mixed; a.w = l.wr; a.logits = logits; a.ids = ids; a.weights = wts; a.xq1 = xq1;
        a.seq = seq; a.ring = 1; a.counter = counter; a.n_tok = T; a.n_embd = N; a.n_expert = NE;
        verify_router(a, s);
    };
    auto ab = [&](const Layer& l, cudaStream_t s) {
        gdn_ab_multi(mixed, l.wa, l.wb, l.dt, l.sa, gate, beta, N, HV, T, s);
    };

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    // one graph of reps x 48 layers of `body`; us per layer
    auto time = [&](const char* name, std::function<void(const Layer&, cudaStream_t)> body) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int r = 0; r < reps; ++r)
            for (const Layer& l : ly) body(l, s);
        CK(cudaStreamEndCapture(s, &g));
        cudaGraphExec_t x;
        CK(cudaGraphInstantiate(&x, g, 0));
        CK(cudaGraphLaunch(x, s));   // warm
        CK(cudaStreamSynchronize(s));
        float best = 1e30f;
        for (int k = 0; k < 5; ++k) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(x, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        std::printf("%-28s %8.2f us a layer\n", name, best * 1e3f / (reps * L));
        CK(cudaGraphExecDestroy(x));
        CK(cudaGraphDestroy(g));
    };
    if (argc > 3) {   // one read of layer 0 from these inputs, every output to argv[3] (compare two builds' bitwise)
        hc_read(ly[0], s);
        hc_read(ly[1], s);   // a second, reading R as the first left it
        CK(cudaStreamSynchronize(s));
        FILE* f = std::fopen(argv[3], "wb");
        auto dump = [&](const float* d, size_t n) {
            std::vector<float> h(n);
            CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
            std::fwrite(h.data(), 4, n, f);
        };
        dump(R, (size_t) T * D); dump(mixed, (size_t) T * N); dump(lo, (size_t) T * LR); dump(rs, (size_t) T * HC);
        dump(inj_out, (size_t) T * HC);
        std::fclose(f);
        return 0;
    }
    std::printf("T = %d, %d reps of 48 layers\n", T, reps);
    time("HC read (down + up)", hc_read);
    time("router", router);
    time("gdn alpha/beta", ab);
    time("HC read + router", [&](const Layer& l, cudaStream_t st) { hc_read(l, st); router(l, st); });
    return 0;
}
