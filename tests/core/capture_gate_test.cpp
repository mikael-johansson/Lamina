// tests/core/capture_gate_test.cpp - the capture gate's exclusion contract (issue #19).
//
// A fresh --serve server hung for good on its first concurrent request: an admission's cudaDeviceSynchronize on the
// main thread landed while a slot's thread recorded a graph on that device, which fails ("operation not permitted
// when stream is capturing") AND breaks that capture.  capture_gate.hpp keeps the two apart: a CaptureHold is
// shared, device_sync()'s wait is exclusive.  No CUDA work is launched here - the test drives the lock's handoffs
// directly, the way adaptive_policy_test drives its comparator; the host test links CUDA::cudart for cudaGetDevice,
// like the other host tests that call cudaMalloc themselves.
#include "strata/core/capture_gate.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

namespace {
int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

using namespace std::chrono_literals;
}  // namespace

int main() {
    std::printf("capture_gate_test\n");

    // 1. captures overlap: CaptureHold is shared, so two slots may record graphs at the same time.
    {
        std::atomic<int> inside{0}, peak{0};
        auto hold = [&] {
            strata::core::CaptureHold h;
            peak = std::max(peak.load(), ++inside);
            std::this_thread::sleep_for(50ms);
            --inside;
        };
        std::thread a(hold), b(hold), c(hold);
        a.join(); b.join(); c.join();
        check(peak.load() >= 2, "CaptureHolds overlap: captures are not serialized against each other");
    }

    // 2. an exclusive wait excludes every capture: while device_sync holds the gate alone, no hold enters.
    {
        std::atomic<bool> sync_region{false}, capture_in_sync{false};
        std::atomic<bool> go{false};
        auto capturer = [&] {
            while (!go.load()) std::this_thread::yield();
            strata::core::CaptureHold h;
            if (sync_region.load()) capture_in_sync = true;
            std::this_thread::sleep_for(50ms);
        };
        std::thread t1(capturer), t2(capturer);
        go = true;
        std::this_thread::sleep_for(10ms);   // both reach their (shared) holds before the wait starts
        {
            std::unique_lock<std::shared_mutex> lk(strata::core::capture_gate());
            sync_region = true;
            std::this_thread::sleep_for(80ms);   // outlasts both capturers' holds
        }
        t1.join(); t2.join();
        check(!capture_in_sync.load(), "no capture enters while the exclusive wait holds the gate");
    }

    // 3. the handoff: a wait queued behind captures gets the gate when they end, after roughly their length
    //    (real exclusion, and no deadlock).
    {
        std::thread capture([] {
            strata::core::CaptureHold h;
            std::this_thread::sleep_for(100ms);
        });
        std::this_thread::sleep_for(10ms);   // the capture is inside its hold
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::shared_mutex> lk(strata::core::capture_gate());   // queues behind the capture;
        // getting here at all is the handoff: a gate that never freed would hang the process (the ctest timeout
        // is its watchdog), and the elapsed time below shows it waited rather than ignoring the capture.
        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        lk.unlock();
        capture.join();
        check(waited >= 50ms, "the exclusive wait queued behind the capture (the gate is real, not a no-op)");
    }

    // 4. one gate per device: two calls on the same current device name the same gate, so a slot's hold and an
    //    admission's wait on that device really do meet (a second physical device needs two GPUs to exercise).
    check(&strata::core::capture_gate() == &strata::core::capture_gate(),
          "capture_gate() resolves to one gate per current device");

    if (failures)
        std::printf("capture_gate_test: %d FAILURES\n", failures);
    else
        std::printf("capture_gate_test: all ok\n");
    return failures ? 1 : 0;
}
