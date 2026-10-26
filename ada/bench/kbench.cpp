// CPU expert kernel microbenchmark: the pool's own kernels (native_gu_rows, native_down_rows, q2_rows_any) on
// synthetic blobs, per format, with the weights in L2, in L3, or streamed from RAM, on 1..8 pinned threads.
// Build: see bench/kbench.sh.  Output: GB/s of weights per thread and in total.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels::cpu;

struct Case { const char* name; int gu, d; };

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

int main(int argc, char** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 0.4;
    const Case cases[] = {{"IQ2_S gate/up", 22, 20}, {"IQ3_XXS gate/up", 18, 20}, {"IQ3_S gate/up", 21, 20},
                          {"IQ4_NL down", 21, 20}, {"Q2_0 down", 21, 42}};
    const int threads_list[] = {1, 2, 4, 6, 7, 8, 14, 16};   // 14: the pool's 7 cores with their SMT siblings
    const size_t stream_bytes = 6ull << 30;   // far past the 32 MB L3
    std::vector<uint8_t> big(stream_bytes);
    std::mt19937_64 rng(1);
    for (size_t i = 0; i < stream_bytes; i += 8) { uint64_t v = rng(); std::memcpy(&big[i], &v, 8); }
    std::printf("%-16s %-6s %2s %3s %9s %9s\n", "kernel", "data", "nt", "thr", "GB/s/thr", "GB/s all");
    for (const Case& c : cases) {
        NativeFmt f;
        std::string err;
        if (!native_fmt(c.gu, c.d, H, FF, f, err)) { std::printf("%s: %s\n", c.name, err.c_str()); continue; }
        const bool down = std::strstr(c.name, "down") != nullptr;
        // one call = what a pool task does: a range of rows of one expert
        const size_t part_off = down ? f.down_off : 0;
        const size_t part_bytes = down ? (size_t) H * f.d_row : f.bytes - (f.bytes - f.down_off) ;   // gate+up
        const size_t n_exp = stream_bytes / f.bytes;
        for (const char* mode : {"L2", "L3", "RAM"}) {
            for (int nt : {1, 2}) {
                for (int T : threads_list) {
                    std::atomic<int> go{0};
                    std::vector<double> gbs(T);
                    std::vector<std::thread> th;
                    for (int t = 0; t < T; ++t)
                        th.emplace_back([&, t] {
                            // logical 0-7 are eight physical cores (siblings are i, i+8); 14 threads: cores 1-7 and
                            // their siblings 9-15 (the pool's cores), 16: all
                            pin(T == 14 ? 1 + t % 7 + 8 * (t / 7) : t);
                            std::vector<float> x(H), h(FF);
                            for (auto& v : x) v = (float) ((rng() % 2001) - 1000) / 1000.f;
                            for (auto& v : h) v = (float) ((t * 7 + 3) % 11) / 10.f;
                            alignas(64) static thread_local uint8_t act[MAXT][16384], hq[MAXT][4096];
                            static thread_local ActQ a2[MAXT];
                            for (int k = 0; k < nt; ++k) {
                                native_quant_act(f, x.data(), act[k]);
                                native_quant_h(f, h.data(), hq[k]);
                                act_quant_any(h.data(), FF, a2[k]);
                            }
                            const void* ap[MAXT];
                            const void* hp[MAXT];
                            const ActQ* a2p[MAXT];
                            alignas(64) static thread_local float ff[MAXT][FF], out[MAXT][H];
                            float* ffp[MAXT];
                            float* outp[MAXT];
                            for (int k = 0; k < MAXT; ++k) { ap[k] = act[k]; hp[k] = hq[k]; a2p[k] = &a2[k]; ffp[k] = ff[k]; outp[k] = out[k]; }
                            // rows per call: L2 = ~160 KB of weights, else the whole part of an expert
                            const int all_rows = down ? H : FF;
                            const size_t row_bytes = down ? f.d_row : 2 * f.gu_row;
                            const int rows = std::string(mode) == "L2" ? (int) std::max<size_t>(16, (160u << 10) / row_bytes) : all_rows;
                            size_t e = (size_t) t * 9973 % n_exp;
                            const uint8_t* fixed = big.data() + (size_t) t * f.bytes;
                            while (go.load() == 0) {}
                            double bytes = 0;
                            const double t0 = now();
                            double t1 = t0;
                            int calls = 0;
                            while (t1 - t0 < secs) {
                                const uint8_t* blob;
                                if (std::string(mode) == "RAM") { blob = big.data() + e * f.bytes; e = (e + 7919) % n_exp; }
                                else blob = fixed;
                                if (!down) native_gu_rows(f, blob, ap, nt, ffp, 0, rows);
                                else if (f.d_type == 42) q2_rows_any(blob + f.down_off, f.d_row, (int) (f.n_ff / 64), a2p, nt, outp, 0, rows);
                                else native_down_rows(f, blob, hp, nt, outp, 0, rows);
                                bytes += (double) rows * row_bytes;
                                if ((++calls & 7) == 0) t1 = now();
                            }
                            t1 = now();
                            gbs[t] = bytes / (t1 - t0) / 1e9;
                        });
                    go.store(1);
                    for (auto& x : th) x.join();
                    double sum = 0;
                    for (double g : gbs) sum += g;
                    std::printf("%-16s %-6s %2d %3d %9.2f %9.2f\n", c.name, mode, nt, T, sum / T, sum);
                    std::fflush(stdout);
                }
            }
        }
        (void) part_off; (void) part_bytes;
    }
    return 0;
}
