// src/core/second_gpu.cpp - see include/strata/core/second_gpu.hpp.
#include "strata/core/second_gpu.hpp"
#include "strata/core/capture_gate.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {

// the plan block: [groups of part 0, of part 1, ring, pad] [starts, part 0: cap+1] [part 1: cap+1] [dst: cap]
// [tok: cap] (8-byte aligned) [slot addresses, part 0: cap] [part 1: cap] [out] [flag after part 0] [after part 1]
struct PlanLayout {
    int64_t starts[2], dst, tok, ptr;   // int32 offsets; `ptr` even
    size_t bytes;
    explicit PlanLayout(int64_t cap)
        : starts{4, 4 + cap + 1}, dst(4 + 2 * (cap + 1)), tok(dst + cap), ptr((tok + cap + 1) & ~1ll),
          bytes((size_t) ptr * 4 + (size_t) (2 * cap + 3) * 8) {}
};

// the grouped kernels' grids are sized for the bucket a layer's group count falls in
int bucket(int groups) {
    for (int b : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) if (groups <= b) return b;
    return groups <= 48 ? 48 : groups;
}

struct DeviceScope {   // the second GPU current for one call, the main one again afterwards
    int main;
    DeviceScope(int dev, int main_dev) : main(main_dev) { cudaSetDevice(dev); }
    ~DeviceScope() { cudaSetDevice(main); }
};

// STRATA_GPU2_PROF=1 (diagnostics): per layer share, the host's time in submit() and this GPU's time for the
// graphs (timing events around them), printed when the second GPU is destroyed.
struct Gpu2Prof {
    bool on = [] { const char* v = std::getenv("STRATA_GPU2_PROF"); return v != nullptr && v[0] == '1'; }();
    cudaEvent_t a = nullptr, b = nullptr;
    cudaEvent_t k[4] = {};   // inside the graphs: start, after the copies in, after the experts, after the rows out
    bool pending = false;
    int64_t n = 0, ng = 0, nk = 0;
    double host_ms = 0, gpu_ms = 0, wait_ms = 0, kms[3] = {};
    cudaEvent_t wa = nullptr, wb = nullptr;   // armed: after the wait kernel, after the graph
    bool apend = false;
    int64_t na = 0;
    double a_gap = 0, a_body = 0, a_tail = 0;
    std::chrono::steady_clock::time_point t0;
    double to_go = 0;
    int64_t ngo = 0;
};
Gpu2Prof g2p;

}  // namespace

SecondGpu::~SecondGpu() {
    if (g2p.on && g2p.n > 0)
        std::fprintf(stderr, "strata second GPU profile: %lld shares, host submit %.1f us (of it waiting for the last "
                             "share %.1f us), GPU graphs %.1f us per share\n", (long long) g2p.n,
                     1000.0 * g2p.host_ms / (double) g2p.n, 1000.0 * g2p.wait_ms / (double) g2p.n,
                     g2p.ng > 0 ? 1000.0 * g2p.gpu_ms / (double) g2p.ng : 0.0);
    if (g2p.on && g2p.ngo > 0)
        std::fprintf(stderr, "strata second GPU profile: from the layer's dispatch to the share's go %.1f us (%lld)\n",
                     1000.0 * g2p.to_go / (double) g2p.ngo, (long long) g2p.ngo);
    if (g2p.on && g2p.na > 0)
        std::fprintf(stderr, "strata second GPU profile: armed, from the go: to the graph's first kernel %.1f us, the "
                             "graph %.1f us, to its end %.1f us (%lld shares)\n", 1000.0 * g2p.a_gap / (double) g2p.na,
                     1000.0 * g2p.a_body / (double) g2p.na, 1000.0 * g2p.a_tail / (double) g2p.na, (long long) g2p.na);
    if (g2p.on && g2p.nk > 0)
        std::fprintf(stderr, "strata second GPU profile: per share with copies in: copies %.1f us, experts %.1f us, "
                             "rows out %.1f us (%lld shares)\n", 1000.0 * g2p.kms[0] / (double) g2p.nk,
                     1000.0 * g2p.kms[1] / (double) g2p.nk, 1000.0 * g2p.kms[2] / (double) g2p.nk, (long long) g2p.nk);
    if (dev_ < 0) return;
    disarm();
    if (arm_enabled_)
        std::fprintf(stderr, "strata second GPU: armed shares %lld taken, %lld released empty (another layer %lld, "
                             "prefetched part %lld, groups %lld, format %lld; the rest at the windows' tails)\n",
                     (long long) arm_fast, (long long) arm_missed, (long long) miss_layer, (long long) miss_pre,
                     (long long) miss_groups, (long long) miss_fmt);
    DeviceScope scope(dev_, main_);
    if (s_) cudaStreamSynchronize(s_);
    if (pre_s_) cudaStreamSynchronize(pre_s_);
    for (auto& gr : graphs_) cudaGraphExecDestroy(gr.exec);
    cache_.close();
    void* dev[] = {d_xq_, d_plan_, d_scratch_, d_rows_, d_count_, d_pre_};
    for (void* p : dev) if (p) cudaFree(p);
    void* host[] = {h_x_, h_plan_, h_go_};
    for (void* p : host) if (p) cudaFreeHost(p);
    if (ev_) cudaEventDestroy(ev_);
    if (arm_ev_) cudaEventDestroy(arm_ev_);
    if (pre_ev_) cudaEventDestroy(pre_ev_);
    if (s_) cudaStreamDestroy(s_);
    if (pre_s_) cudaStreamDestroy(pre_s_);
}

bool SecondGpu::init(int device, int main_device, int64_t n_embd, int64_t n_ff, int max_t, int64_t k, std::string& err) {
    dev_ = device;
    main_ = main_device;
    max_t_ = max_t;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    cap_ = (int64_t) max_t * k;
    DeviceScope scope(dev_, main_);
    const size_t xb = (size_t) max_t * (std::max)((size_t) (n_embd / 32) * 36, (size_t) (n_embd / 256) * 292);
    const size_t ob = (size_t) cap_ * (size_t) n_embd * sizeof(float);
    const size_t pb = PlanLayout(cap_).bytes;
    const unsigned pm = cudaHostAllocPortable | cudaHostAllocMapped;
    const char* step = nullptr;
    cudaError_t e = cudaSuccess;
    int unified = 0;
    auto run = [&](const char* what, cudaError_t r) { if (e == cudaSuccess && r != cudaSuccess) { e = r; step = what; } };
    run("context", cudaFree(nullptr));
    run("unified addressing", cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, dev_));
    run("stream", cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
    run("event", cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming));
    run("pinned staging", cudaHostAlloc((void**) &h_x_, xb, pm));
    run("pinned staging", cudaHostAlloc((void**) &h_plan_, pb, pm));
    run("pinned staging", cudaHostAlloc((void**) &h_go_, 64, pm));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_go_, h_go_, 0));
    run("event", cudaEventCreateWithFlags(&arm_ev_, cudaEventDisableTiming));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_x_, h_x_, 0));
    run("mapped staging", cudaHostGetDevicePointer((void**) &m_plan_, h_plan_, 0));
    run("device buffers", cudaMalloc((void**) &d_xq_, xb));
    run("device buffers", cudaMalloc((void**) &d_plan_, pb));
    run("device buffers", cudaMalloc((void**) &d_scratch_, strata::kernels::native_expert_scratch_bytes(cap_, n_ff)));
    run("device buffers", cudaMalloc((void**) &d_rows_, ob));
    run("device buffers", cudaMalloc((void**) &d_count_, sizeof(unsigned)));
    run("device buffers", cudaMemset(d_count_, 0, sizeof(unsigned)));
    if (e != cudaSuccess) {
        err = std::string("second GPU: ") + step + ": " + cudaGetErrorString(e);
        return false;
    }
    if (!unified) {   // it writes its rows through the host pointers of the first GPU's mapped memory
        err = "second GPU: no unified addressing";
        return false;
    }
    std::memset(h_x_, 0, xb);
    std::memset(h_plan_, 0, pb);
    std::memset(h_go_, 0, 64);
    n_layers_ = (int64_t) strata::kernels::cpu::expert_layout().fmt.size();
    { const char* v = std::getenv("STRATA_GPU2_ARM"); arm_enabled_ = v != nullptr && v[0] == '1'; }
    return true;
}

bool SecondGpu::init_prefetch(int slots, uint64_t blob_bytes, std::string& err) {
    slots = (std::min)(slots, kPrefetchMax);
    if (slots <= 0) return true;
    DeviceScope scope(dev_, main_);
    pre_cap_ = (blob_bytes + 255) / 256 * 256;
    if (cudaStreamCreateWithFlags(&pre_s_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&pre_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc((void**) &d_pre_, (size_t) slots * pre_cap_) != cudaSuccess) {
        err = std::string("second GPU: prefetch slots: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    pre_max_ = slots;
    return true;
}

bool SecondGpu::prefetch(int64_t layer, const int32_t* ids, const uint8_t* const* src, int n, uint64_t bytes,
                         std::string& err) {
    if (n <= 0) return true;
    if (n > pre_max_ || bytes > pre_cap_) { err = "second GPU: a prefetch does not fit its slots"; return false; }
    DeviceScope scope(dev_, main_);
    void* dst[kPrefetchMax];
    size_t size[kPrefetchMax];
    for (int i = 0; i < n; ++i) {
        dst[i] = d_pre_ + (size_t) i * pre_cap_;
        size[i] = (size_t) bytes;
    }
    if (cudaStreamWaitEvent(pre_s_, ev_, 0) != cudaSuccess ||   // the last submitted layer may read the slots
        !copy_blobs(dst, (const void* const*) src, size, (size_t) n, pre_s_) ||
        cudaEventRecord(pre_ev_, pre_s_) != cudaSuccess) {
        err = std::string("second GPU: prefetch: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(pre_s_);   // submit now, not at the next driver call
    pre_layer_ = layer;
    pre_n_ = n;
    for (int i = 0; i < n; ++i) pre_ids_[i] = ids[i];
    prefetch_copied += n;
    return true;
}

bool SecondGpu::graph_for(int gu_type, int d_type, int gu_act, int d_act, int groups, int part, bool prep,
                          cudaGraphExec_t& exec,
                          std::string& err) {
    for (const auto& gr : graphs_)
        if (gr.gu == gu_type && gr.d == d_type && gr.gu_act == gu_act && gr.d_act == d_act &&
            gr.groups == groups && gr.part == part && gr.prep == prep) {
            exec = gr.exec;
            return true;
        }
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(
        gu_type, d_type, n_embd_, n_ff_, gu_act, d_act);
    const PlanLayout P(cap_);
    const auto* pi = (const int32_t*) d_plan_;
    const auto* p64 = (const unsigned long long*) (pi + P.ptr);
    cudaGraph_t graph = nullptr;
    strata::core::CaptureHold capture_hold;
    if (cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "second GPU: cannot capture";
        return false;
    }
    if (g2p.on && g2p.k[0] == nullptr)
        for (auto& ev : g2p.k) cudaEventCreate(&ev);
    auto stamp = [&](int i) { if (g2p.on && prep) cudaEventRecordWithFlags(g2p.k[i], s_, cudaEventRecordExternal); };
    try {
        stamp(0);
        if (prep) {
            // kernels copy the plan and the activations (a copy node costs ~15 us on this link); the grouped kernels
            // read the group count from the plan and skip the groups past it
            strata::kernels::copy_i32_from_mapped((int32_t*) d_plan_, (const int32_t*) m_plan_, (int64_t) P.bytes / 4,
                                                  s_);
            const int64_t act_bytes = L.gu_act == 15 ? (n_embd_ / 256) * 292 : (n_embd_ / 32) * 36;
            strata::kernels::copy_from_mapped((float*) d_xq_, (const float*) m_x_, max_t_ * act_bytes / 4, s_);
        }
        stamp(1);
        const int64_t xq_stride = L.gu_act == 15 ? (n_embd_ / 256) * 292 : (n_embd_ / 32) * 36;
        strata::kernels::native_expert_grouped(L, p64 + part * cap_, pi + P.starts[part], pi + part, pi + P.dst,
                                               pi + P.tok, groups, cap_, d_xq_, xq_stride, d_scratch_, d_rows_, s_);
        stamp(2);
        strata::kernels::native_expert_rows_out(d_rows_, pi + part, pi + P.starts[part], pi + P.dst, n_embd_, cap_,
                                                (float* const*) (p64 + 2 * cap_),
                                                (uint32_t* const*) (p64 + 2 * cap_ + 1 + part), pi + 2, d_count_, s_);
        stamp(3);
    } catch (const std::exception& ex) {
        cudaStreamEndCapture(s_, &graph);
        if (graph) cudaGraphDestroy(graph);
        err = std::string("second GPU: ") + ex.what();
        return false;
    }
    const cudaError_t ce = cudaStreamEndCapture(s_, &graph);
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("second GPU: graph capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    cudaGraphUpload(exec, s_);
    cudaStreamSynchronize(s_);
    graphs_.push_back({gu_type, d_type, gu_act, d_act, groups, part, prep, exec});
    return true;
}

void SecondGpu::write_plan(int64_t layer, int n_tok, int64_t k, const int32_t* slots, const int32_t* starts,
                           const int32_t* entries, int n_groups, float* out, uint32_t* flag, uint32_t ring, int ng[2]) {
    (void) layer;
    (void) n_tok;
    // the plan: the experts in its cache first (part 0), then the prefetched ones (part 1), their entries in that order
    const PlanLayout P(cap_);
    auto* pi = (int32_t*) h_plan_;
    auto* p64 = (unsigned long long*) (pi + P.ptr);
    ng[0] = ng[1] = 0;
    int e = 0;
    for (int part = 0; part < 2; ++part) {
        int32_t* st = pi + P.starts[part];
        for (int g = 0; g < n_groups; ++g) {
            const bool pre = slots[g] <= -2;
            if (pre != (part == 1)) continue;
            p64[(size_t) part * (size_t) cap_ + (size_t) ng[part]] =
                pre ? (unsigned long long) (d_pre_ + (size_t) (-2 - slots[g]) * pre_cap_)
                    : (unsigned long long) cache_.device_slot(slots[g]);
            st[ng[part]++] = e;
            for (int j = starts[g]; j < starts[g + 1]; ++j, ++e) {
                pi[P.dst + e] = entries[j];
                pi[P.tok + e] = (int32_t) (entries[j] / k);
            }
        }
        st[ng[part]] = e;
        pi[part] = ng[part];
    }
    pi[2] = (int32_t) ring;
    p64[2 * cap_] = (unsigned long long) out;
    p64[2 * cap_ + 1] = ng[1] == 0 ? (unsigned long long) flag : 0;   // the last launch raises it
    p64[2 * cap_ + 2] = ng[1] > 0 ? (unsigned long long) flag : 0;
}

namespace {
constexpr int kArmBucket = 16;   // the armed graph's group bucket: a layer's share with more groups takes the slow path
}

bool SecondGpu::arm(int64_t layer, std::string& err) {
    const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    const int gu_act = strata::kernels::gpu_expert_act(f.gu_act), d_act = strata::kernels::gpu_expert_act(f.d_act);
    DeviceScope scope(dev_, main_);
    cudaGraphExec_t ex = nullptr;
    if (!graph_for(f.gu_type, f.d_type, gu_act, d_act, kArmBucket, 0, true, ex, err)) return false;
    if (++seq_ == 0) ++seq_;   // the word starts at 0
    strata::kernels::wait_mapped_u32(m_go_, seq_, s_);
    if (g2p.on) {
        if (g2p.wa == nullptr) { cudaEventCreate(&g2p.wa); cudaEventCreate(&g2p.wb); }
        cudaEventRecord(g2p.wa, s_);
    }
    if (cudaGraphLaunch(ex, s_) != cudaSuccess || (g2p.on && cudaEventRecord(g2p.wb, s_) != cudaSuccess) ||
        cudaEventRecord(arm_ev_, s_) != cudaSuccess) {
        err = std::string("second GPU: arming: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(s_);   // submit now, not at the next driver call
    arm_ = {true, layer, f.gu_type, f.d_type, gu_act, d_act, seq_};
    return true;
}

void SecondGpu::disarm() {
    if (!arm_.on) return;
    // the last real share may still be copying the plan in: the empty plan goes in once it is done
    while (cudaEventQuery(ev_) == cudaErrorNotReady) _mm_pause();
    const PlanLayout P(cap_);
    auto* pi = (int32_t*) h_plan_;
    auto* p64 = (unsigned long long*) (pi + P.ptr);
    pi[0] = pi[1] = 0;
    pi[P.starts[0]] = 0;
    pi[P.starts[1]] = 0;
    p64[2 * cap_ + 1] = 0;   // no flag: the rows-out kernel returns at once
    p64[2 * cap_ + 2] = 0;
    _mm_sfence();
    *(volatile uint32_t*) h_go_ = arm_.seq;
    arm_.on = false;
    ++arm_missed;
    while (cudaEventQuery(arm_ev_) == cudaErrorNotReady) _mm_pause();
}

bool SecondGpu::submit(int64_t layer, const uint8_t* x, int64_t x_stride, int n_tok, int64_t k, const int32_t* slots,
                       const int32_t* starts, const int32_t* entries, int n_groups, float* out, uint32_t* flag,
                       uint32_t ring, std::string& err) {
    const int n = starts[n_groups];
    if (n_groups <= 0 || n > cap_ || n_tok > max_t_) { err = "second GPU: a layer's share is out of range"; return false; }
    const auto h0 = std::chrono::steady_clock::now();
    cudaError_t q;
    while ((q = cudaEventQuery(ev_)) == cudaErrorNotReady) _mm_pause();
    const auto h1 = std::chrono::steady_clock::now();
    if (q != cudaSuccess) {
        err = std::string("second GPU: ") + cudaGetErrorString(q);
        return false;
    }
    if (g2p.on && g2p.apend) {
        float u[3];
        if (cudaEventElapsedTime(&u[0], g2p.wa, g2p.k[0]) == cudaSuccess &&
            cudaEventElapsedTime(&u[1], g2p.k[0], g2p.k[3]) == cudaSuccess &&
            cudaEventElapsedTime(&u[2], g2p.k[3], g2p.wb) == cudaSuccess) {
            g2p.a_gap += u[0]; g2p.a_body += u[1]; g2p.a_tail += u[2]; ++g2p.na;
        }
        g2p.apend = false;
    }
    if (g2p.on && g2p.pending) {
        float t = 0.f;
        if (cudaEventElapsedTime(&t, g2p.a, g2p.b) == cudaSuccess) { g2p.gpu_ms += t; ++g2p.ng; }
        float u[3];
        if (g2p.k[0] && cudaEventElapsedTime(&u[0], g2p.k[0], g2p.k[1]) == cudaSuccess &&
            cudaEventElapsedTime(&u[1], g2p.k[1], g2p.k[2]) == cudaSuccess &&
            cudaEventElapsedTime(&u[2], g2p.k[2], g2p.k[3]) == cudaSuccess && u[0] >= 0 && u[1] >= 0 && u[2] >= 0) {
            for (int i = 0; i < 3; ++i) g2p.kms[i] += u[i];
            ++g2p.nk;
        }
        g2p.pending = false;
    }
    const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
    const int gu_act = strata::kernels::gpu_expert_act(f.gu_act), d_act = strata::kernels::gpu_expert_act(f.d_act);
    int pre_groups = 0;
    for (int g = 0; g < n_groups; ++g) pre_groups += slots[g] <= -2;
    // the armed graph runs part 0 (it copies the plan in); a prefetched part 1 follows it as usual
    const bool fits = arm_.on && arm_.layer == layer && n_groups - pre_groups <= kArmBucket &&
                      arm_.gu == f.gu_type && arm_.d == f.d_type && arm_.gu_act == gu_act && arm_.d_act == d_act;
    if (arm_.on && !fits) {
        if (arm_.layer != layer) ++miss_layer;
        else if (n_groups - pre_groups > kArmBucket) ++miss_groups;
        else ++miss_fmt;
        disarm();
    }
    int ng[2];
    write_plan(layer, n_tok, k, slots, starts, entries, n_groups, out, flag, ring, ng);
    const int64_t act_bytes = gu_act == 15 ? (n_embd_ / 256) * 292 : (n_embd_ / 32) * 36;
    for (int t = 0; t < n_tok; ++t)
        std::memcpy(h_x_ + (size_t) t * act_bytes, x + (size_t) t * x_stride, (size_t) act_bytes);
    DeviceScope scope(dev_, main_);
    if (fits) {
        _mm_sfence();
        *(volatile uint32_t*) h_go_ = arm_.seq;   // the armed graph takes the plan and the activations from here
        arm_.on = false;
        ++arm_fast;
        if (g2p.on) {
            g2p.apend = true;
            g2p.to_go += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g2p.t0).count();
            ++g2p.ngo;
        }
        cudaGraphExec_t pre = nullptr;
        if (ng[1] > 0 && (!graph_for(f.gu_type, f.d_type, gu_act, d_act, bucket(ng[1]), 1, false, pre, err) ||
                          cudaStreamWaitEvent(s_, pre_ev_, 0) != cudaSuccess || cudaGraphLaunch(pre, s_) != cudaSuccess)) {
            err = err.empty() ? std::string("second GPU: ") + cudaGetErrorString(cudaGetLastError()) : err;
            return false;
        }
        if (cudaEventRecord(ev_, s_) != cudaSuccess) {
            err = std::string("second GPU: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    } else {
        cudaGraphExec_t own = nullptr, pre = nullptr;
        if ((ng[0] > 0 && !graph_for(f.gu_type, f.d_type, gu_act, d_act, bucket(ng[0]), 0, true, own, err)) ||
            (ng[1] > 0 && !graph_for(f.gu_type, f.d_type, gu_act, d_act, bucket(ng[1]), 1, ng[0] == 0, pre, err)))
            return false;
        if (g2p.on && g2p.a == nullptr) { cudaEventCreate(&g2p.a); cudaEventCreate(&g2p.b); }
        if (g2p.on) cudaEventRecord(g2p.a, s_);
        if ((own != nullptr && cudaGraphLaunch(own, s_) != cudaSuccess) ||
            (pre != nullptr &&
             (cudaStreamWaitEvent(s_, pre_ev_, 0) != cudaSuccess || cudaGraphLaunch(pre, s_) != cudaSuccess)) ||
            cudaEventRecord(ev_, s_) != cudaSuccess) {
            err = std::string("second GPU: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (g2p.on) {
            cudaEventRecord(g2p.b, s_);
            g2p.pending = true;
        }
    }
    if (g2p.on) {
        ++g2p.n;
        const auto h2 = std::chrono::steady_clock::now();
        g2p.host_ms += std::chrono::duration<double, std::milli>(h2 - h0).count();
        g2p.wait_ms += std::chrono::duration<double, std::milli>(h1 - h0).count();
    }
    ++layers;
    experts += n_groups;
    entries_done += n;
    prefetch_used += ng[1];
    if (arm_enabled_ && layer + 1 < n_layers_ && !arm(layer + 1, err)) return false;
    return true;
}
void SecondGpu::mark_dispatch() {
    if (g2p.on) g2p.t0 = std::chrono::steady_clock::now();
}

bool SecondGpu::healthy(std::string& err) const {
    const cudaError_t q = cudaEventQuery(ev_);
    if (q == cudaSuccess || q == cudaErrorNotReady) return true;
    err = std::string("second GPU: ") + cudaGetErrorString(q);
    return false;
}

}  // namespace strata::core
