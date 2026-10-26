// src/core/decode_split.cpp - see include/strata/core/decode_split.hpp.
#include "strata/core/decode_split.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

namespace strata::core {
namespace {

uint64_t up(uint64_t n) { return (n + 255) & ~255ull; }

/// A device context switch for the scope.
struct OnDevice {
    int prev = 0;
    explicit OnDevice(int dev) { cudaGetDevice(&prev); cudaSetDevice(dev); }
    ~OnDevice() { cudaSetDevice(prev); }
};

}  // namespace

DecodeSplit::~DecodeSplit() {
    if (dev_ < 0) return;
    OnDevice on(dev_);
    cudaDeviceSynchronize();
    if (weights_) cudaFree(weights_);
    if (state_) cudaFree(state_);
}

bool DecodeSplit::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, int64_t at, int device,
                       int main_device, const std::vector<std::string>& head_shards, int64_t n_vocab, std::string& err) {
    if (at < 2 || at >= g.n_layers) {
        err = "--decode-split: the second GPU's first layer must be 2.." + std::to_string(g.n_layers - 1);
        return false;
    }
    g_ = &g;
    ss_ = &ss;
    at_ = at;
    dev_ = device;
    main_ = main_device;
    // ---- the dense weights of the layers from `at` on and the output's: their buffers' sizes and places
    struct Tensor {
        std::string name;
        WeightRef ref;
        uint64_t off[4] = {}, bytes[4] = {};   // data, native_data, hc_codes, hc_scales
    };
    std::vector<Tensor> tensors;
    uint64_t wbytes = 0;
    for (const auto& [name, ref] : wt.all()) {
        bool here = name.rfind("output_hc_", 0) == 0;
        if (name.rfind("blk.", 0) == 0) here = std::atoll(name.c_str() + 4) >= at;
        if (!here) continue;
        Tensor t;
        t.name = name;
        t.ref = ref;
        t.bytes[0] = ref.data != nullptr ? ref.bytes : 0;
        if (ref.native_data != nullptr) {
            int be = 0, bb = 0;
            if (!strata::block_geometry((uint32_t) ref.native_type, be, bb) || be <= 0 || (ref.ne0 * ref.ne1) % be != 0) {
                err = "--decode-split: " + name + " has a native type without a block geometry";
                return false;
            }
            t.bytes[1] = (uint64_t) (ref.ne0 * ref.ne1 / be) * (uint64_t) bb;
        }
        if (ref.hc_codes != nullptr) t.bytes[2] = (uint64_t) (ref.ne0 * ref.ne1);
        if (ref.hc_scales != nullptr) t.bytes[3] = (uint64_t) (ref.ne0 * ref.ne1 / 32) * 2;
        for (int k = 0; k < 4; ++k) {
            t.off[k] = wbytes;
            wbytes += up(t.bytes[k]);
        }
        tensors.push_back(std::move(t));
    }
    // ---- the layers' state indices: the main session's QSA states here, its GDN states [gdn_first, + gdn_count)
    for (int64_t l = 0, qi = 0, gi = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) {
            if (l >= at) qsa_main_.push_back(qi);
            ++qi;
        } else {
            if (l < at) gdn_first_ = gi + 1;
            else ++gdn_count_;
            ++gi;
        }
    }
    const int64_t cells = ss.qsa_states[0].max_cells;
    uint64_t sbytes = up((uint64_t) gdn_count_ * gdn_state_floats(g) * 4);
    for (size_t i = 0; i < qsa_main_.size(); ++i) sbytes += up(qsa_state_prompt_bytes(g, cells, 0, true, qsa_kv_format()));

    OnDevice on(device);
    if (cudaMalloc(&weights_, (size_t) wbytes) != cudaSuccess || cudaMalloc(&state_, (size_t) sbytes) != cudaSuccess) {
        err = "--decode-split: " + std::to_string((wbytes + sbytes) >> 20) + " MiB of weights and state do not fit on GPU " +
              std::to_string(device);
        return false;
    }
    // the weights, from the main GPU's copies
    uint8_t* w = (uint8_t*) weights_;
    for (const Tensor& t : tensors) {
        const void* src[4] = {t.ref.data, t.ref.native_data, t.ref.hc_codes, t.ref.hc_scales};
        for (int k = 0; k < 4; ++k)
            if (t.bytes[k] > 0 && cudaMemcpyPeer(w + t.off[k], device, src[k], main_device, (size_t) t.bytes[k]) != cudaSuccess) {
                err = "--decode-split: copying " + t.name + ": " + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        WeightRef r = t.ref;
        r.data = t.bytes[0] ? w + t.off[0] : nullptr;
        r.native_data = t.bytes[1] ? w + t.off[1] : nullptr;
        r.hc_codes = t.bytes[2] ? (const int8_t*) (w + t.off[2]) : nullptr;
        r.hc_scales = t.bytes[3] ? (const uint16_t*) (w + t.off[3]) : nullptr;
        r.native_q8_1 = nullptr;
        wt2_.put(t.name, r);
    }
    // the verifier reads the vocabulary's size from output.weight only (the head is `head_`)
    if (const WeightRef* wo = wt.find("output.weight")) {
        WeightRef r = *wo;
        r.data = nullptr;
        r.native_data = nullptr;
        r.native_q8_1 = nullptr;
        wt2_.put("output.weight", r);
    }
    // the session of the layers here, zeroed
    uint8_t* p = (uint8_t*) state_;
    ss2_.gdn_state = (float*) p;
    cudaMemset(p, 0, (size_t) gdn_count_ * gdn_state_floats(g) * 4);
    p += up((uint64_t) gdn_count_ * gdn_state_floats(g) * 4);
    qsa2_.assign(qsa_main_.size(), QsaState{});
    for (QsaState& q : qsa2_) {
        const uint64_t n = qsa_state_prompt_init(g, cells, 0, true, qsa_kv_format(), p, q, nullptr);
        if (n == 0) { err = "--decode-split: the K/V state"; return false; }
        p += up(n);
    }
    ss2_.qsa_states = qsa2_.data();
    ss2_.k = ss.k;
    ss2_.max_cells = ss.max_cells;
    // the head, all of it
    if (!head_.load(head_shards, g.n_embd, n_vocab, err)) {
        err = "--decode-split: the head: " + err;
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("--decode-split: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    bytes_ = wbytes + sbytes + head_.weight_bytes();
    return true;
}

bool DecodeSplit::copy_state(int64_t c0, int64_t c1, bool in, std::string& err) {
    const ModelGeometry& g = *g_;
    const strata::kernels::QsaShapes s = qsa_shapes(g);
    // (dst, its device, src, its device, bytes)
    auto copy = [&](void* here, const void* main, uint64_t n) {
        if (n == 0) return true;
        const cudaError_t e = in ? cudaMemcpyPeer(here, dev_, main, main_, (size_t) n)
                                 : cudaMemcpyPeer((void*) main, main_, here, dev_, (size_t) n);
        if (e == cudaSuccess) return true;
        err = std::string("--decode-split: the state's copy: ") + cudaGetErrorString(e);
        return false;
    };
    const uint64_t gdn = (uint64_t) gdn_count_ * gdn_state_floats(g) * 4;
    if (!copy(ss2_.gdn_state, ss_->gdn_state + (uint64_t) gdn_first_ * gdn_state_floats(g), gdn)) return false;
    const uint64_t row = (uint64_t) s.idx_dim * 4;
    for (size_t i = 0; i < qsa2_.size(); ++i) {
        const QsaState &q = qsa2_[i], &m = ss_->qsa_states[qsa_main_[i]];
        if (!copy(q.idx_tail, m.idx_tail, (uint64_t) (s.idx_block - 1) * row) || !copy(q.idx_dead, m.idx_dead, row) ||
            !copy(q.idx_block_pos, m.idx_block_pos, 4))
            return false;
        if (c1 <= c0) continue;
        // whole pages of the cells: both states lay the pools out [page][kv_head][page_size][row] from page 0
        const int64_t p0 = c0 / s.page_size, p1 = (c1 + s.page_size - 1) / s.page_size;
        const auto aq = qsa_kv_arrays(q, g), am = qsa_kv_arrays(m, g);
        for (size_t k = 0; k < aq.size(); ++k) {
            const uint64_t page = (uint64_t) s.n_head_kv * (uint64_t) s.page_size * aq[k].row_bytes;
            if (!copy((uint8_t*) aq[k].data + (uint64_t) p0 * page, (const uint8_t*) am[k].data + (uint64_t) p0 * page,
                      (uint64_t) (p1 - p0) * page))
                return false;
        }
        const int64_t r0 = c0 / s.idx_block, r1 = c1 / s.idx_block + 1;   // the open block's row too
        if (!copy((uint8_t*) q.idx_pooled + (uint64_t) r0 * row, (const uint8_t*) m.idx_pooled + (uint64_t) r0 * row,
                  (uint64_t) (r1 - r0) * row))
            return false;
    }
    return true;
}

bool DecodeSplit::copy_in(int64_t cells, std::string& err) { return copy_state(0, cells, true, err); }

bool DecodeSplit::copy_out(int64_t c0, int64_t c1, std::string& err) { return copy_state(c0, c1, false, err); }

}  // namespace strata::core
