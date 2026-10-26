// include/strata/core/decode_split.hpp - --decode-split: decode's verify windows in two stages on two GPUs.
//
// The main GPU runs layers [0, at) of every window, the second GPU layers [at, n_layers) and the head, each with the
// experts of its own layers in its VRAM tier (the profile's pairs split by layer), the CPU pool computing the rest of
// each.  The window's residual after layer at - 1 crosses once, through pinned host memory (the main verifier's
// final rows).  So no layer waits for another GPU's share of its experts: the second-GPU tier's per-layer round trip
// through host memory (~100 us) is gone, and each GPU reads only its own layers' dense weights.
//
// The second GPU keeps copies of its layers' dense weights and of the head, and its layers' sequence state for good,
// laid out as the prompt split's second stage's (prefill/split.hpp): indexed from `at`.  A prompt runs on the main
// GPU's session (or through the prompt split); `copy_in` then brings the state of the layers here across, and
// `copy_out` takes it back before the next prompt reads it.
#pragma once

#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class DecodeSplit {
public:
    DecodeSplit() = default;
    ~DecodeSplit();
    DecodeSplit(const DecodeSplit&) = delete;
    DecodeSplit& operator=(const DecodeSplit&) = delete;

    /// With the main GPU current: copies the dense weights of layers [at, n_layers) and the output's to `device`,
    /// loads the head there from `head_shards` (all `n_vocab` rows), and makes its session for those layers at
    /// `ss`'s context.  The main GPU is current again on return.
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, int64_t at, int device, int main_device,
              const std::vector<std::string>& head_shards, int64_t n_vocab, std::string& err);
    int64_t at() const { return at_; }
    int device() const { return dev_; }
    /// The second stage's window (its layers and the head); init it, and run it, with `device()` current.
    Verifier& verifier() { return ver_; }
    SessionState& session() { return ss2_; }
    const WeightTable& weights() const { return wt2_; }
    const NativeHead& head() const { return head_; }
    /// VRAM taken on the second GPU: the weights, the head and the session.
    uint64_t bytes() const { return bytes_; }

    /// The main session's state of the layers here after positions [0, cells) to this GPU: every GDN state, each QSA
    /// layer's fixed state (the indexer's tail, spare key and block position) and its cells and pooled keys.
    bool copy_in(int64_t cells, std::string& err);
    /// The other way, for the cells [c0, c1) this GPU appended since: the main session then holds every layer again.
    bool copy_out(int64_t c0, int64_t c1, std::string& err);

private:
    bool copy_state(int64_t c0, int64_t c1, bool in, std::string& err);

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;   // the main GPU's
    int64_t at_ = 0;
    int dev_ = -1, main_ = 0;
    WeightTable wt2_;
    NativeHead head_;
    SessionState ss2_;
    std::vector<QsaState> qsa2_;
    std::vector<int64_t> qsa_main_;   // their index among the main session's QSA states
    int64_t gdn_first_ = 0, gdn_count_ = 0;
    void* weights_ = nullptr;
    void* state_ = nullptr;
    uint64_t bytes_ = 0;
    Verifier ver_;
};

}  // namespace strata::core
