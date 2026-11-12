// src/core/prompt_cache.cpp - the --serve prompt cache (see include/strata/core/prompt_cache.hpp).
#include "strata/core/prompt_cache.hpp"
#include "strata/core/capture_gate.hpp"
#include "strata/core/on_device.hpp"

#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>

namespace strata::core {

namespace {

/// A replaced sequence is stashed when the switch would drop at least this many of its cells.
constexpr int64_t kStashMin = 1024;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

}  // namespace

PromptCache::HostPages::~HostPages() {
    for (const auto& [p, pinned] : slabs_) {
        if (pinned) cudaFreeHost(p);
        else std::free(p);
    }
}

void PromptCache::HostPages::init(uint64_t budget, bool pin) {
    for (uint64_t left = budget / kPage * kPage; left >= kPage;) {
        const uint64_t bytes = std::min(left, kSlab);
        void* p = nullptr;
        bool pinned = false;
        if (pin) {
            pinned = cudaHostAlloc(&p, (size_t) bytes, cudaHostAllocPortable) == cudaSuccess;
            if (!pinned) {   // the rest pageable: slower copies, same result
                (void) cudaGetLastError();
                pin = false;
                p = nullptr;
            }
        }
        if (!pinned) p = std::malloc((size_t) bytes);
        if (p == nullptr) break;
        const uint32_t first = (uint32_t) slabs_.size() * kPerSlab, n = (uint32_t) (bytes / kPage);
        slabs_.emplace_back((uint8_t*) p, pinned);
        ref_.resize((size_t) slabs_.size() * kPerSlab, 0);
        for (uint32_t i = 0; i < n; ++i) free_.push_back(first + i);
        total_ += n;
        if (pinned) pinned_ += bytes;
        left -= bytes;
    }
    std::reverse(free_.begin(), free_.end());   // taken from the back: ascending, so a sequence's pages run together
}

bool PromptCache::HostPages::take(size_t n, std::vector<uint32_t>& out) {
    if (free_.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        out.push_back(free_.back());
        ref_[free_.back()] = 1;
        free_.pop_back();
    }
    return true;
}

void PromptCache::HostPages::give(uint32_t page) {
    if (ref_[page] > 0 && --ref_[page] == 0) free_.push_back(page);
}

void PromptCache::HostPages::give(std::vector<uint32_t>& pages) {
    for (size_t i = pages.size(); i-- > 0;) give(pages[i]);
    pages.clear();
}

std::vector<uint32_t> PromptCache::HostPages::share(const std::vector<uint32_t>& pages) {
    for (uint32_t pg : pages) ++ref_[pg];
    return pages;
}

PromptCache::PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream,
                         int max_ckpts, uint64_t stash_bytes)
    : g_(g), ss_(ss), kv_(std::move(kv)), stream_(stream), max_ckpts_(max_ckpts), stash_budget_(stash_bytes),
      ckpt_bytes_(session_ckpt_bytes(g)) {
    if (max_ckpts_ <= 0) return;
    const Clock::time_point t0 = Clock::now();
    // the checkpoint pool page-locked at once (a buffer pinned when the first request needs it took ~60 ms)
    while (allocated_ < max_ckpts_) {
        void* h = nullptr;
        if (cudaMallocHost(&h, (size_t) ckpt_bytes_) != cudaSuccess) {
            (void) cudaGetLastError();
            break;   // the rest as take_buffer finds it
        }
        owned_.emplace_back(h, true);
        free_.push_back(h);
        ++allocated_;
    }
    if (stash_budget_ == 0) return;
    pages_.init(stash_budget_, std::getenv("STRATA_STASH_PAGEABLE") == nullptr);
    std::fprintf(stderr, "strata serve: prompt cache: %d checkpoint buffers and %zu MiB of stash pages, %llu MiB of "
                         "them page-locked (%.1f s)\n",
                 allocated_, pages_.pages() * (size_t) (HostPages::kPage >> 20),
                 (unsigned long long) (pages_.pinned_bytes() >> 20), ms_since(t0) / 1000.0);
}

PromptCache::~PromptCache() {
    for (const auto& [dev, st] : streams_) {
        OnDevice on(dev);
        cudaStreamDestroy((cudaStream_t) st);
    }
    for (const auto& [p, pinned] : owned_) {
        if (pinned) cudaFreeHost(p);
        else std::free(p);
    }
}

int64_t PromptCache::reusable(const Seq& seq, const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs,
                              bool live) const {
    const int64_t m = std::min<int64_t>((int64_t) seq.tokens.size(), (int64_t) ids.size() - 1);
    int64_t lcp = 0;
    while (lcp < m && seq.tokens[(size_t) lcp] == ids[(size_t) lcp]) ++lcp;

    // The token prefix is necessary but not sufficient for multimodal input:
    // every image that falls inside the reusable prefix must have the same
    // embedding/grid identity at the same position.  If an image was changed,
    // removed, added, or reordered, fail closed before its first pad token.
    auto matching_image = [](const std::vector<ImgKey>& all, const ImgKey& wanted) {
        for (const ImgKey& image : all)
            if (image.start == wanted.start) return image == wanted;
        return false;
    };
    for (const ImgKey& image : seq.imgs)
        if (image.start < lcp && !matching_image(imgs, image)) lcp = image.start;
    for (const ImgKey& image : imgs)
        if (image.start < lcp && !matching_image(seq.imgs, image)) lcp = image.start;

    if (live && lcp > 0 && lcp == (int64_t) seq.tokens.size()) return lcp;
    int64_t best = -1;
    for (const Ckpt& c : seq.ckpts)
        if (c.pos <= lcp && c.pos > best) best = c.pos;
    return best;
}

PromptCache::Ckpt* PromptCache::find(Seq& seq, int64_t pos) {
    for (Ckpt& c : seq.ckpts)
        if (c.pos == pos) return &c;
    return nullptr;
}

void* PromptCache::take_buffer(int64_t pos) {
    if (!free_.empty()) {
        void* h = free_.back();
        free_.pop_back();
        return h;
    }
    if (allocated_ < max_ckpts_) {
        void* h = nullptr;
        const bool pinned = cudaMallocHost(&h, (size_t) ckpt_bytes_) == cudaSuccess;
        if (!pinned) {
            (void) cudaGetLastError();
            h = std::malloc((size_t) ckpt_bytes_);   // pageable: slower copies, same result
        }
        if (h != nullptr) {
            owned_.emplace_back(h, pinned);
            ++allocated_;
        }
        return h;
    }
    // the pool is full: each sequence offers the checkpoint it can spare best (the live one's chain with the new
    // checkpoint at its end), and the least recently used offer goes
    Seq* owner = nullptr;
    size_t at = 0;
    uint64_t oldest = UINT64_MAX;
    std::vector<size_t> order;
    std::vector<int64_t> chain;
    auto offer = [&](Seq& s, bool live) {
        order.resize(s.ckpts.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return s.ckpts[a].pos < s.ckpts[b].pos; });
        chain.clear();
        size_t keep = SIZE_MAX;
        for (size_t k = 0; k < order.size(); ++k) {
            chain.push_back(s.ckpts[order[k]].pos);
            if (chain.back() == s.pin) keep = k;
        }
        if (live) chain.push_back(pos);
        const size_t k = spread_victim(chain.data(), chain.size(), keep);
        size_t pick = k < order.size() ? order[k] : SIZE_MAX;
        if (pick == SIZE_MAX)   // too few to spread: the least recently used
            for (size_t i = 0; i < s.ckpts.size(); ++i)
                if (s.ckpts[i].pos != s.pin && (pick == SIZE_MAX || s.ckpts[i].stamp < s.ckpts[pick].stamp))
                    pick = i;
        if (pick != SIZE_MAX && s.ckpts[pick].stamp < oldest) {
            oldest = s.ckpts[pick].stamp;
            owner = &s;
            at = pick;
        }
    };
    offer(live_, true);
    for (Seq& s : stash_) offer(s, false);
    if (owner == nullptr) return nullptr;
    void* h = owner->ckpts[at].host;
    owner->ckpts.erase(owner->ckpts.begin() + (ptrdiff_t) at);
    if (owner != &live_ && owner->ckpts.empty()) {   // a stashed sequence without a checkpoint cannot be resumed
        drop_kv(*owner);
        stash_.erase(stash_.begin() + (owner - stash_.data()));
    }
    return h;
}

void PromptCache::release(Seq& seq, int64_t keep_upto) {
    for (size_t i = 0; i < seq.ckpts.size();) {
        if (seq.ckpts[i].pos > keep_upto) {
            free_.push_back(seq.ckpts[i].host);
            seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
}

std::vector<PromptCache::Part> PromptCache::kv_parts(int64_t cells) const {
    // [page][kv_head][page_size][head_dim] with an identity page table: the first cells are one prefix of each
    // pool.  The indexer's pooled rows are one per idx_block cells.
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const uint64_t row = (uint64_t) g_.idx_key_dim * sizeof(float);
    std::vector<Part> parts;
    for (size_t i = 0; i < kv_.size(); ++i) {
        const QsaState* st = kv_[i];
        const int dev = i < kv_device.size() ? kv_device[i] : -1;
        const uint64_t pages = (uint64_t) std::min<int64_t>(st->n_pages, (cells + sh.page_size - 1) / sh.page_size);
        const uint64_t c = pages * (uint64_t) g_.n_head_kv * (uint64_t) sh.page_size;
        for (const KvArray& a : qsa_kv_arrays(*st, g_)) parts.push_back({a.data, c * a.row_bytes, dev});
        const int64_t rows = std::min<int64_t>(cells / sh.idx_block + 2, st->max_cells / sh.idx_block + 2);
        parts.push_back({st->idx_pooled, (uint64_t) rows * row, dev, row});
    }
    return parts;
}

int64_t PromptCache::copy_floor(int64_t valid) const {
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const int64_t blk = std::max<int64_t>({sh.page_size, sh.idx_block, 1});
    const int64_t v = valid - 256;   // the draft layer writes a cell once the token after it is known
    return v > 0 ? v / blk * blk : 0;
}

void PromptCache::drop_kv(Seq& seq) {
    for (std::vector<uint32_t>& p : seq.kv) pages_.give(p);
    seq.kv.clear();
    seq.kv_cells = seq.kv_valid = 0;
    seq.kv_bytes = 0;
}

uint64_t PromptCache::part_from(const std::vector<Part>& lo, const Part& p, size_t i, int64_t c0) const {
    if (c0 <= 0) return 0;
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    return std::min(p.bytes, p.row ? (uint64_t) (c0 / sh.idx_block) * p.row : lo[i].bytes);
}

bool PromptCache::ensure_pages(Seq& seq, int64_t cells, int64_t c0) {
    const std::vector<Part> parts = kv_parts(cells), lo = c0 > 0 ? kv_parts(c0) : std::vector<Part>{};
    seq.kv.resize(parts.size());
    cow_.assign(parts.size(), 0);
    std::vector<size_t> want(parts.size());
    size_t need = 0, all = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        want[i] = (size_t) ((parts[i].bytes + HostPages::kPage - 1) / HostPages::kPage);
        std::vector<uint32_t>& pg = seq.kv[i];
        while (pg.size() > want[i]) {   // a longer sequence's pages past these cells
            pages_.give(pg.back());
            pg.pop_back();
        }
        all += want[i];
        need += want[i] - pg.size();
        for (size_t k = (size_t) (part_from(lo, parts[i], i, c0) / HostPages::kPage); k < pg.size(); ++k)
            need += pages_.shared(pg[k]) ? 1 : 0;
    }
    if (all > pages_.pages()) return false;
    while (pages_.free_pages() < need) {   // the least recently used sequences go, but not one lending its prefix
        auto lru = stash_.end();
        for (auto it = stash_.begin(); it != stash_.end(); ++it)
            if (it->pin < 0 && (lru == stash_.end() || it->stamp < lru->stamp)) lru = it;
        if (lru == stash_.end()) return false;
        release(*lru, 0);
        drop_kv(*lru);
        stash_.erase(lru);
    }
    size_t held = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        std::vector<uint32_t>& pg = seq.kv[i];
        const size_t k0 = (size_t) (part_from(lo, parts[i], i, c0) / HostPages::kPage);
        for (size_t k = k0; k < pg.size(); ++k) {
            if (!pages_.shared(pg[k])) continue;
            std::vector<uint32_t> fresh;
            if (!pages_.take(1, fresh)) return false;
            pages_.give(pg[k]);
            pg[k] = fresh[0];
            if (k == k0) cow_[i] = 1;
        }
        if (pg.size() < want[i] && !pages_.take(want[i] - pg.size(), pg)) return false;
        held += pg.size();
    }
    seq.kv_bytes = (uint64_t) held * HostPages::kPage;
    return true;
}

bool PromptCache::kv_copy(Seq& seq, int64_t c0, int64_t c1, bool to_host) {
    if (c1 <= c0) return true;
    const std::vector<Part> hi = kv_parts(c1), lo = c0 > 0 ? kv_parts(c0) : std::vector<Part>{};
    if (seq.kv.size() != hi.size()) return false;
    // each GPU's parts on its own stream (the main GPU's: the session's), all queued before the first wait
    std::vector<std::pair<int, cudaStream_t>> used;
    auto stream_for = [&](int dev) -> cudaStream_t {
        for (const auto& [d, st] : used)
            if (d == dev) return st;
        cudaStream_t st = (cudaStream_t) stream_;
        if (dev >= 0) {
            st = nullptr;
            for (const auto& [d, s2] : streams_)
                if (d == dev) st = (cudaStream_t) s2;
            if (st == nullptr) {
                OnDevice on(dev);
                if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
                streams_.emplace_back(dev, (void*) st);
            }
        }
        used.emplace_back(dev, st);
        return st;
    };
    bool ok = true;
    for (size_t i = 0; ok && i < hi.size(); ++i) {
        const Part& p = hi[i];
        uint64_t off = part_from(lo, p, i, c0);
        if (to_host && i < cow_.size() && cow_[i]) off = off / HostPages::kPage * HostPages::kPage;   // a fresh page
        const std::vector<uint32_t>& pg = seq.kv[i];
        if ((uint64_t) pg.size() * HostPages::kPage < p.bytes) { ok = false; break; }   // not held: no partial copy
        cudaStream_t cs = stream_for(p.device);
        if (cs == nullptr && p.device >= 0) { ok = false; break; }
        OnDevice on(p.device);
        uint8_t* dev = (uint8_t*) p.dev;
        while (ok && off < p.bytes) {   // runs of adjacent pages as one copy
            const size_t k = (size_t) (off / HostPages::kPage);
            size_t k2 = k;
            while (k2 + 1 < pg.size() && (uint64_t) (k2 + 1) * HostPages::kPage < p.bytes &&
                   HostPages::adjacent(pg[k2], pg[k2 + 1]))
                ++k2;
            const uint64_t end = std::min<uint64_t>(p.bytes, (uint64_t) (k2 + 1) * HostPages::kPage);
            uint8_t* h = pages_.at(pg[k]) + off % HostPages::kPage;
            const size_t n = (size_t) (end - off);
            ok = (to_host ? cudaMemcpyAsync(h, dev + off, n, cudaMemcpyDeviceToHost, cs)
                          : cudaMemcpyAsync(dev + off, h, n, cudaMemcpyHostToDevice, cs)) == cudaSuccess;
            moved_ += n;
            off = end;
        }
    }
    for (const auto& [d, st] : used) {
        OnDevice on(d);
        if (cudaStreamSynchronize(st) != cudaSuccess) ok = false;
    }
    cow_.clear();
    return ok || session_copy_failed(stream_);
}

void PromptCache::stash_live(int64_t keep_upto) {
    const int64_t L = (int64_t) live_.tokens.size();
    // a short sequence is not worth its copies, unless a client keeps a rolling prefix in it (a voice conversation of
    // a few hundred tokens that a Hermes turn displaces between two decisions)
    const bool rolling = std::any_of(live_.ckpts.begin(), live_.ckpts.end(), [](const Ckpt& c) { return c.roll; });
    // the live sequence's pages (a restored copy) go with it, its cells up to kv_valid unchanged since
    Seq s;
    int64_t from = 0;
    if (!live_.kv.empty() && live_.kv_valid > 0) from = copy_floor(std::min(live_.kv_valid, L));
    s.kv = std::move(live_.kv);
    live_.kv.clear();
    drop_kv(live_);
    if (stash_budget_ == 0 || (L < kStashMin && !rolling) || (ensure_kv && !ensure_kv(L))) {
        drop_kv(s);
        return;
    }
    const Clock::time_point t0 = Clock::now();
    moved_ = 0;
    if (!ensure_pages(s, L, from) || !kv_copy(s, from, L, true)) {
        drop_kv(s);
        return;
    }
    cost_.out_ms += ms_since(t0);
    cost_.out_bytes += moved_;
    cost_.out_cells = L;
    cost_.out_new = L - from;
    // STRATA_STASH_VERIFY=1 (a test): the cells kept from the restored copy against the device's, byte for byte
    static const bool verify = std::getenv("STRATA_STASH_VERIFY") != nullptr;
    if (verify && from > 0) {
        const std::vector<Part> lo = kv_parts(from);
        uint64_t checked = 0, bad = 0;
        std::vector<uint8_t> dev;
        for (size_t i = 0; i < lo.size(); ++i) {
            const uint64_t n = part_from(lo, kv_parts(L)[i], i, from);
            dev.resize((size_t) n);
            OnDevice on(lo[i].device);
            if (n == 0 || cudaMemcpy(dev.data(), lo[i].dev, (size_t) n, cudaMemcpyDeviceToHost) != cudaSuccess) continue;
            for (uint64_t off = 0; off < n;) {
                const uint64_t len = std::min<uint64_t>(n - off, HostPages::kPage - off % HostPages::kPage);
                const uint8_t* h = pages_.at(s.kv[i][(size_t) (off / HostPages::kPage)]) + off % HostPages::kPage;
                for (uint64_t b = 0; b < len; ++b) bad += h[b] != dev[(size_t) (off + b)] ? 1 : 0;
                off += len;
            }
            checked += n;
        }
        std::fprintf(stderr, "strata serve: prompt cache verify: %llu kept bytes below cell %lld, %llu differ%s\n",
                     (unsigned long long) checked, (long long) from, (unsigned long long) bad, bad ? " (A BUG)" : "");
    }
    const Clock::time_point t1 = Clock::now();
    checkpoint(L);   // the end state: the session is still at L
    cost_.ckpt_out_ms += ms_since(t1);
    for (size_t i = 0; i < live_.ckpts.size();) {
        if (live_.ckpts[i].pos > keep_upto) {
            s.ckpts.push_back(live_.ckpts[i]);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
    if (s.ckpts.empty()) {
        drop_kv(s);
        return;
    }
    s.tokens = live_.tokens;
    for (const ImgKey& image : live_.imgs)
        if (image.start < L) s.imgs.push_back(image);
    s.kv_cells = L;
    s.stamp = ++clock_;
    s.used = live_.used;
    stash_.push_back(std::move(s));
}

bool PromptCache::restore(Seq& seq, int64_t pos, bool prefix) {
    Ckpt* ck = find(seq, pos);
    if (ck == nullptr) return false;
    const int64_t cells = prefix ? pos : seq.kv_cells;
    // (the live sequence's own pages: the device holds those cells already)
    if (&seq != &live_ && !seq.kv.empty()) {
        if (ensure_kv && !ensure_kv(cells)) return false;
        const Clock::time_point t0 = Clock::now();
        moved_ = 0;
        if (!kv_copy(seq, 0, cells, false)) return false;
        cost_.in_ms += ms_since(t0);
        cost_.in_bytes += moved_;
        cost_.in_cells = cells;
    }
    ck->stamp = ++clock_;
    const Clock::time_point t1 = Clock::now();
    // the layers this session holds, as checkpoint() saved them: with a second session (the decode split) only the
    // ones before it, the main GPU's copies of the others having been freed or lent (batch slots)
    const int64_t here = second_from > 0 ? second_from : g_.n_layers;
    const bool ok = (here < g_.n_layers ? session_ckpt_load_layers(ss_, g_, pos, ck->host, stream_, 0, here, 0)
                                        : session_ckpt_load(ss_, g_, pos, ck->host, stream_)) &&
                    (second_from <= 0 || !second_load || second_load(pos, ck->host));
    cost_.ckpt_in_ms += ms_since(t1);
    return ok;
}

int64_t PromptCache::begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs) {
    cost_ = SwitchCost{};
    const Clock::time_point t0 = Clock::now();
    const int64_t r = begin_impl(ids, imgs);
    live_.used = Clock::now();
    if (cost_.out_bytes > 0 || cost_.in_bytes > 0 || cost_.ckpt_in_ms > 0) {
        char idle[32] = "-";
        if (cost_.idle_s >= 0) std::snprintf(idle, sizeof idle, "%.1f s", cost_.idle_s);
        std::fprintf(stderr,
                     "strata serve: prompt cache switch (%s): out %.0f MiB (%lld cells, %lld copied) in %.1f ms + "
                     "checkpoint %.1f ms; in %.0f MiB (%lld cells) in %.1f ms + checkpoint %.1f ms; idle %s; "
                     "%.1f ms in all\n",
                     cost_.from ? cost_.from : "live", (double) cost_.out_bytes / 1048576.0, (long long) cost_.out_cells,
                     (long long) cost_.out_new, cost_.out_ms, cost_.ckpt_out_ms, (double) cost_.in_bytes / 1048576.0,
                     (long long) cost_.in_cells, cost_.in_ms, cost_.ckpt_in_ms, idle, ms_since(t0));
    }
    return r;
}

int64_t PromptCache::begin_impl(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs) {
    const int64_t L = (int64_t) live_.tokens.size();
    auto zero = [&]() -> int64_t {
        release(live_, 0);
        live_.tokens.clear();
        live_.imgs.clear();
        session_zero(ss_, g_, nullptr, stream_);
        cudaStreamSynchronize((cudaStream_t) stream_);
        if (second_from > 0 && second_zero) second_zero();
        return 0;
    };
    if (!enabled()) return zero();
    int64_t best = reusable(live_, ids, imgs, true);
    int from = -1;   // -1: the live sequence, else a stashed one
    for (size_t i = 0; i < stash_.size(); ++i) {
        const int64_t r = reusable(stash_[i], ids, imgs, false);
        if (r > best) {
            best = r;
            from = (int) i;
        }
    }
    if (best <= 0) {
        cost_.from = "none";
        stash_live(0);
        const int64_t result = zero();
        live_.imgs = imgs;
        return result;
    }
    if (from < 0) {
        // The live sequence continues.  When the prompt drops most of it (another conversation that shares a
        // prefix), the rest goes to the stash; an edit near the end just overwrites it.
        const int64_t dropped = L - best;
        const bool rolling = std::any_of(live_.ckpts.begin(), live_.ckpts.end(),
                                         [best](const Ckpt& c) { return c.roll && c.pos > best; });
        cost_.from = "live";
        if ((dropped >= kStashMin || rolling) && dropped > best) {
            live_.pin = best;
            stash_live(best);
            live_.pin = -1;
        }
        if (best < L && !restore(live_, best)) return zero();
        release(live_, best);
        live_.tokens.resize((size_t) best);
        live_.kv_valid = std::min(live_.kv_valid, best);
        live_.imgs = imgs;
        return best;
    }
    int64_t newest = 0;
    for (const Ckpt& c : stash_[(size_t) from].ckpts) newest = std::max(newest, c.pos);
    if (best < newest) {
        // Another conversation that shares the stashed one's start: it borrows the cells before `best`, and the
        // stashed sequence stays whole (pinned while the live one is stashed beside it).
        stash_[(size_t) from].pin = best;
        cost_.from = "shared prefix";
        cost_.idle_s = std::chrono::duration<double>(Clock::now() - stash_[(size_t) from].used).count();
        stash_live(0);
        release(live_, 0);
        live_.tokens.clear();
        auto donor = std::find_if(stash_.begin(), stash_.end(), [](const Seq& s) { return s.pin >= 0; });
        if (donor == stash_.end()) return zero();
        const bool ok = restore(*donor, best, true);
        donor->pin = -1;
        if (!ok) return zero();
        donor->stamp = ++clock_;
        // its pages are read by the live sequence too: the cells before `best` match them, so the next stash of this
        // one copies only the cells after those (a page it writes that the donor holds is replaced first)
        drop_kv(live_);
        for (const std::vector<uint32_t>& pg : donor->kv) live_.kv.push_back(pages_.share(pg));
        live_.kv_cells = donor->kv_cells;
        live_.kv_bytes = donor->kv_bytes;
        live_.kv_valid = best;
        live_.tokens.assign(donor->tokens.begin(), donor->tokens.begin() + best);
        live_.imgs = imgs;
        return best;
    }
    // That sequence continues: the live one is stashed and the chosen one becomes live.
    Seq chosen = std::move(stash_[(size_t) from]);
    stash_.erase(stash_.begin() + from);
    cost_.from = "stash";
    cost_.idle_s = std::chrono::duration<double>(Clock::now() - chosen.used).count();
    stash_live(0);
    release(live_, 0);
    live_.tokens.clear();
    if (!restore(chosen, best)) {
        release(chosen, 0);
        drop_kv(chosen);
        return zero();
    }
    live_.tokens.assign(chosen.tokens.begin(), chosen.tokens.begin() + best);
    live_.ckpts = std::move(chosen.ckpts);
    // its pages stay with it: the cells before `best` match the device until the sequence rewrites them
    drop_kv(live_);
    live_.kv = std::move(chosen.kv);
    chosen.kv.clear();
    live_.kv_cells = chosen.kv_cells;
    live_.kv_bytes = chosen.kv_bytes;
    live_.kv_valid = best;
    release(live_, best);
    live_.imgs = imgs;
    return best;
}

void PromptCache::set(int64_t pos, const int64_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.kv_valid = std::min(live_.kv_valid, pos);
    live_.used = Clock::now();
    live_.tokens.resize((size_t) pos);
    live_.tokens.insert(live_.tokens.end(), tokens, tokens + n);
}

void PromptCache::set(int64_t pos, const int32_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.kv_valid = std::min(live_.kv_valid, pos);
    live_.used = Clock::now();
    live_.tokens.resize((size_t) pos);
    for (int64_t i = 0; i < n; ++i) live_.tokens.push_back(tokens[i]);
}

void PromptCache::checkpoint(int64_t pos, bool roll) {
    if (!enabled() || pos <= 0 || pos != (int64_t) live_.tokens.size()) return;
    if (Ckpt* c = find(live_, pos)) {
        c->roll = c->roll && roll;   // a conversation's own checkpoint there stays one
        return;
    }
    void* h = take_buffer(pos);
    if (h == nullptr) return;
    strata::core::device_sync();
    const bool split = split_from > 0 && split_save;
    const int64_t here = split ? split_from : second_from > 0 ? second_from : g_.n_layers;   // this session's layers
    if (!(here < g_.n_layers ? session_ckpt_save_layers(ss_, g_, pos, h, stream_, 0, here, 0)
                             : session_ckpt_save(ss_, g_, pos, h, stream_)) ||
        (!split && second_from > 0 && second_save && !second_save(pos, h))) {
        free_.push_back(h);
        return;
    }
    if (split) split_save(pos, h);
    live_.ckpts.push_back({pos, h, ++clock_, roll});
    if (!roll) return;
    // the older rolling checkpoints go but the newest of them (a revised last word resumes there), and never the one
    // this request resumed from
    int64_t keep = -1;
    for (const Ckpt& c : live_.ckpts)
        if (c.roll && c.pos < pos && c.pos > keep) keep = c.pos;
    for (size_t i = 0; i < live_.ckpts.size();) {
        const Ckpt& c = live_.ckpts[i];
        if (c.roll && c.pos < keep && c.pos != live_.pin) {
            free_.push_back(c.host);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
}

void PromptCache::forget() {
    drop_kv(live_);
    release(live_, 0);
    live_.tokens.clear();
    live_.imgs.clear();
}

void PromptCache::clear() {
    drop_kv(live_);
    release(live_, -1);
    live_.tokens.clear();
    live_.imgs.clear();
    for (Seq& s : stash_) {
        release(s, -1);
        drop_kv(s);
    }
    stash_.clear();
}

std::string PromptCache::summary() const {
    size_t n = live_.ckpts.size();
    uint64_t kv = 0;
    for (const Seq& s : stash_) {
        n += s.ckpts.size();
        kv += s.kv_bytes;
    }
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "live %lld cells; %zu checkpoints (%.0f MiB); %zu stashed sequences (%.0f MiB); live's copy %.0f MiB; "
                  "%zu of %zu stash pages free",
                  (long long) live_.tokens.size(), n, (double) (n * ckpt_bytes_) / 1048576.0, stash_.size(),
                  (double) kv / 1048576.0, (double) live_.kv_bytes / 1048576.0, pages_.free_pages(), pages_.pages());
    return buf;
}

}  // namespace strata::core
