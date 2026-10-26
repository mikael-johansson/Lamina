// Kernel trace through CUPTI's activity API: every kernel's GPU time, inside CUDA graphs too.  Loaded beside the
// engine (LD_PRELOAD=bench/libktrace.so); at exit it prints, per kernel name, the launches, total / mean / max GPU
// time, launch shape and registers, and per device the time kernels ran against the time between them (the gaps a
// latency-bound sequence of small kernels leaves).  KTRACE_OUT=<file>: the table goes there (default stderr).
// KTRACE_RAW=<file>: each kernel's line as well (the last table's).
// KTRACE_SKIP_MS=n: kernels in the first n ms after the first one are left out (startup, captures).
// A server is killed rather than exiting, so with KTRACE_OUT set, creating <KTRACE_OUT>.go makes it write the table
// for the kernels since the last one (appended to KTRACE_OUT) and start over: touch it before and after a request.
// Build: see bench/ktrace.sh.
#include <cupti.h>

#include <cxxabi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct Stat {
    uint64_t n = 0, ns = 0, max_ns = 0;
    uint32_t grid = 0, block = 0, regs = 0, smem = 0;
};
struct Span { uint64_t start, end; uint32_t dev, stream; const std::string* name; };

std::mutex g_mu;
std::map<std::string, Stat> g_stats;
std::vector<Span> g_spans;
uint64_t g_first = 0, g_skip_ns = 0;

std::string demangle(const char* m) {
    int st = 0;
    char* d = abi::__cxa_demangle(m, nullptr, nullptr, &st);
    std::string s = st == 0 && d ? d : m;
    std::free(d);
    // templates' arguments make every instantiation its own row: keep them, but cut very long ones
    if (s.size() > 160) s = s.substr(0, 157) + "...";
    return s;
}

void CUPTIAPI buffer_requested(uint8_t** buf, size_t* size, size_t* max_records) {
    *size = 8u << 20;
    *buf = (uint8_t*) aligned_alloc(8, *size);
    *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* rec = nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
        if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
        const auto* k = (const CUpti_ActivityKernel9*) rec;
        if (g_first == 0) g_first = k->start;
        if (k->start < g_first + g_skip_ns) continue;
        auto it = g_stats.try_emplace("[" + std::to_string(k->gridX * k->gridY * k->gridZ) + "] " + demangle(k->name)).first;
        Stat& s = it->second;
        const uint64_t d = k->end - k->start;
        ++s.n;
        s.ns += d;
        s.max_ns = std::max<uint64_t>(s.max_ns, d);
        s.grid = k->gridX * k->gridY * k->gridZ;
        s.block = k->blockX * k->blockY * k->blockZ;
        s.regs = k->registersPerThread;
        s.smem = k->staticSharedMemory + k->dynamicSharedMemory;
        g_spans.push_back({k->start, k->end, k->deviceId, k->streamId, &it->first});
    }
    std::free(buf);
}

void report_to(const char* mode) {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> lk(g_mu);
    const char* out = std::getenv("KTRACE_OUT");
    FILE* f = out != nullptr ? std::fopen(out, mode) : stderr;
    if (f == nullptr) f = stderr;
    std::vector<std::pair<std::string, Stat>> rows(g_stats.begin(), g_stats.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    uint64_t total = 0;
    for (const auto& r : rows) total += r.second.ns;
    std::fprintf(f, "ktrace: %zu kernels, %.1f ms of kernel time\n", g_spans.size(), total / 1e6);
    std::fprintf(f, "%8s %6s %10s %9s %9s %8s %6s %5s %6s  %s\n", "ms", "%", "launches", "mean us", "max us", "grid",
                 "block", "regs", "smem", "kernel");
    for (const auto& [name, s] : rows) {
        if (s.ns < total / 2000) break;   // the long tail below 0.05%
        std::fprintf(f, "%8.1f %6.2f %10llu %9.2f %9.1f %8u %6u %5u %6u  %s\n", s.ns / 1e6, 100.0 * s.ns / total,
                     (unsigned long long) s.n, s.ns / 1e3 / s.n, s.max_ns / 1e3, s.grid, s.block, s.regs, s.smem,
                     name.c_str());
    }
    // per device: the span from its first kernel to its last, the time some kernel ran (the union), the rest idle
    std::map<uint32_t, std::vector<Span>> by;
    for (const Span& s : g_spans) by[s.dev].push_back(s);
    for (auto& [dev, v] : by) {
        std::sort(v.begin(), v.end(), [](const Span& a, const Span& b) { return a.start < b.start; });
        uint64_t busy = 0, cur_s = v[0].start, cur_e = v[0].end;
        for (const Span& s : v) {
            if (s.start > cur_e) { busy += cur_e - cur_s; cur_s = s.start; cur_e = s.end; }
            else cur_e = std::max(cur_e, s.end);
        }
        busy += cur_e - cur_s;
        const uint64_t span = v.back().end - v.front().start;
        std::fprintf(f, "device %u: %zu kernels over %.1f ms, some kernel running %.1f ms (%.1f%%), none %.1f ms\n", dev,
                     v.size(), span / 1e6, busy / 1e6, 100.0 * busy / span, (span - busy) / 1e6);
    }
    if (f != stderr) std::fclose(f);
    // KTRACE_RAW=<file>: every kernel too (device, stream, start and end ns, name), for timelines
    if (const char* raw = std::getenv("KTRACE_RAW")) {
        if (FILE* r = std::fopen(raw, "w")) {
            for (const Span& s : g_spans)
                std::fprintf(r, "%u\t%u\t%llu\t%llu\t%s\n", s.dev, s.stream, (unsigned long long) s.start,
                             (unsigned long long) s.end, s.name->c_str());
            std::fclose(r);
        }
    }
    g_stats.clear();
    g_spans.clear();
}

void report() { report_to("a"); }

struct Init {
    Init() {
        if (const char* v = std::getenv("KTRACE_SKIP_MS")) g_skip_ns = (uint64_t) (std::atof(v) * 1e6);
        cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        std::atexit(report);
        if (const char* out = std::getenv("KTRACE_OUT")) {
            std::string go = std::string(out) + ".go";
            std::thread([go] {
                for (;;) {
                    usleep(200000);
                    if (access(go.c_str(), F_OK) == 0) { unlink(go.c_str()); report(); }
                }
            }).detach();
        }
    }
} g_init;

}  // namespace
