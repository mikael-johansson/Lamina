// include/strata/core/capture_gate.hpp - a device-wide wait never overlaps a graph capture on that device.
//
// cudaDeviceSynchronize fails while any thread records a graph on the current device ("operation not permitted when
// stream is capturing"), and it breaks that thread's capture too (its EndCapture then fails, "operation failed due to a
// previous error during capture").  A batch admission waits on the main thread beside slots' threads that capture a
// window's graph on its first use (issue #19).  Captures are rare and short, so they hold the gate shared and a wait
// holds it alone; a wait still waits for all the device's work, as it needs to before it reads a sequence's state.
#pragma once

#include <cuda_runtime.h>

#include <mutex>
#include <shared_mutex>

namespace strata::core {

inline std::shared_mutex& capture_gate() {
    static std::shared_mutex gates[64];
    int dev = 0;
    cudaGetDevice(&dev);
    return gates[dev & 63];
}

/// Held from before cudaStreamBeginCapture until after cudaStreamEndCapture, on the capture's current device.
class CaptureHold {
public:
    CaptureHold() : lk_(capture_gate()) {}
private:
    std::shared_lock<std::shared_mutex> lk_;
};

/// cudaDeviceSynchronize for the current device, beside other threads' captures.  Never call it inside a capture.
inline cudaError_t device_sync() {
    std::unique_lock<std::shared_mutex> lk(capture_gate());
    return cudaDeviceSynchronize();
}

}  // namespace strata::core
