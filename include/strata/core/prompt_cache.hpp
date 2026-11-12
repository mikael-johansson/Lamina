// include/strata/core/prompt_cache.hpp - the --serve prompt cache.
//
// A request continues from the longest prefix of its prompt that the engine still holds, instead of processing
// the whole conversation again.  Three things hold prefixes:
//
//   * the LIVE sequence: the tokens whose cells the session state holds right now;
//   * CHECKPOINTS of the live sequence (session_ckpt_*): the fixed-size state at one position.  The K/V cells
//     are indexed by position, so restoring a checkpoint at `pos` and overwriting the cells from `pos` on
//     continues the sequence from there;
//   * STASHED sequences: a sequence the live one replaced, kept in host memory with its K/V cells and its
//     checkpoints, so a client that alternates between conversations (a chat and its title request, an agent
//     and its subagents) does not start over on every switch.  A prompt that continues a stashed sequence from its
//     newest checkpoint takes it out of the stash; one that matches it only up to an earlier checkpoint (another
//     conversation with the same start: sibling subagents, sessions with one system prompt) borrows the cells up to
//     there, and the stashed sequence stays whole.
//
// Checkpoint buffers are a shared pool of `max_ckpts`.  When it is full, each sequence offers the checkpoint it can
// spare best (spread_victim over its own; with fewer than three, its least recently used), and the least recently
// used offer goes: a conversation's checkpoints thin out geometrically back from its newest instead of all ending up
// in its last few turns.  The stash's K/V copies share a byte budget, the least recently used sequence going first.
//
// The K/V copies live in one host arena of 2 MiB pages (`--cache-ram`), page-locked at startup so the copies run at
// the links' speed, the two GPUs' at once (STRATA_STASH_PAGEABLE=1: ordinary memory, the old speed).  A sequence
// restored from the stash keeps its pages while it is live: the cells it has not overwritten still match them, so
// its next stash copies only the cells after those.
#pragma once

#include "strata/core/session.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// The checkpoint of a sequence's chain that leaves when one must: `pos` ascending, its first (the chain's root, the
/// deepest point later prompts share), its last and `keep` (another index; none: any value >= n) kept.  Of the
/// others, the one whose removal leaves the smallest gap for its distance from the last (upstream's
/// --prompt-cache-spread rule), so the kept points thin out geometrically back from the newest: a prompt that leaves
/// the chain d cells before its end reads about d to 2d of them again.  n: none (fewer than three points).
inline size_t spread_victim(const int64_t* pos, size_t n, size_t keep) {
    size_t v = n;
    double best = 0;
    for (size_t i = 1; i + 1 < n; ++i) {
        if (i == keep) continue;
        const double gap = double(pos[i + 1] - pos[i - 1]) / double(pos[n - 1] - pos[i - 1]);
        if (v == n || gap < best) {
            v = i;
            best = gap;
        }
    }
    return v;
}

/// Identity of an image embedded in a GENI request.  Token IDs alone only
/// describe the image-pad run; the embedding bytes and grid describe what the
/// model actually saw.  `start` is the first image-pad token position.
struct ImgKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& other) const { return start == other.start && hash == other.hash; }
};

class PromptCache {
public:
    /// `kv`: every state whose cells a stashed sequence keeps (the QSA layers and the draft layer).
    PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream, int max_ckpts,
                uint64_t stash_bytes);
    ~PromptCache();
    PromptCache(const PromptCache&) = delete;
    PromptCache& operator=(const PromptCache&) = delete;

    bool enabled() const { return max_ckpts_ > 0; }
    /// The elastic K/V (--kv-grow): called before any copy of `cells` K/V cells to or from the device (a stashed
    /// sequence's restore, the live one's stash), so the pools can grow to hold them first.  false: no room.
    std::function<bool(int64_t cells)> ensure_kv;
    /// Item bf, while a split prompt path runs (prefill/split.hpp): a checkpoint saves the layers below `split_from`
    /// itself and hands its buffer to `split_save`, whose stage writes the rest once that chunk is through it (the
    /// prompt's end waits for that).  0: not split.
    int64_t split_from = 0;
    std::function<void(int64_t pos, void* host)> split_save;
    /// --decode-split (core/decode_split.hpp): the layers from `second_from` on live in another session, on another
    /// GPU: a checkpoint saves them through `second_save` (unless a split prompt path's stage does: split_save), a
    /// restore loads them through `second_load`, a fresh sequence zeroes them through `second_zero`.  0: one session.
    int64_t second_from = 0;
    std::function<bool(int64_t pos, void* host)> second_save, second_load;
    std::function<void()> second_zero;
    /// The device each of `kv`'s states is on, for those not on the stream's (-1, or past its end: the stream's).
    std::vector<int> kv_device;
    /// Cells the live sequence holds (a later stash reads all of them).
    int64_t live_cells() const { return (int64_t) live_.tokens.size(); }
    uint64_t ckpt_bytes() const { return ckpt_bytes_; }

    /// Sets the session up for a prompt of `ids`: continues from the longest prefix (at most ids.size() - 1
    /// cells, the last token always goes through a verify window) whose token
    /// and image prefix matches the live sequence, a checkpoint or a stashed
    /// sequence, or zeroes the session.  Image identity is checked before the
    /// matching position is returned; a changed image clamps reuse to before
    /// that image.  Text-only requests pass an empty image-key vector.
    int64_t begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs);
    /// The cells [pos, pos + n) now hold `tokens`; pos <= live length, and anything after pos is gone.
    void set(int64_t pos, const int64_t* tokens, int64_t n);
    void set(int64_t pos, const int32_t* tokens, int64_t n);
    /// Checkpoints the session, which is at `pos` == the live length.  `roll`: a client's rolling prefix (roll=N, a
    /// voice decision's context before its throw-away probe): it replaces the live sequence's older rolling checkpoints
    /// but the newest one before it, so a conversation that grows a word at a time holds two, not one a request.
    void checkpoint(int64_t pos, bool roll = false);
    /// Forgets the live sequence after a failed/cancelled request.
    void forget();
    /// Forgets the live sequence, every checkpoint and every stashed sequence (after non-finite logits: whatever the
    /// state holds may be poisoned).
    void clear();

    /// One line for the log: checkpoints and stashed sequences.
    std::string summary() const;

private:
    struct Ckpt { int64_t pos; void* host; uint64_t stamp; bool roll = false; };
    struct Seq {
        std::vector<int64_t> tokens;
        std::vector<ImgKey> imgs;
        std::vector<Ckpt> ckpts;
        /// Each K/V part's host pages (kv_parts order): cells [0, kv_cells) of every K/V state.  A stashed sequence's
        /// copy; the live one's while it continues a restored copy, whose cells [0, kv_valid) still match the device.
        std::vector<std::vector<uint32_t>> kv;
        int64_t kv_cells = 0;
        int64_t kv_valid = 0;
        uint64_t kv_bytes = 0;   // the pages' bytes
        uint64_t stamp = 0;
        std::chrono::steady_clock::time_point used{};   // last read or written (the idle time of a switch)
        /// begin(): the checkpoint a request resumes from, kept through the pool's evictions while the live sequence
        /// is stashed (a stashed sequence lending its prefix is not evicted either); -1: none.
        int64_t pin = -1;
    };

    /// One device region of the K/V cells, stored back to back in a stashed sequence's buffer.
    struct Part { void* dev; uint64_t bytes; int device = -1; uint64_t row = 0; };   // row: the indexer's pooled rows

    /// The stash's host memory: `budget` bytes in pages of kPage, page-locked unless that fails or is switched off.
    class HostPages {
    public:
        static constexpr uint64_t kPage = 2ull << 20;
        static constexpr uint64_t kSlab = 256ull << 20;   // one allocation
        ~HostPages();
        void init(uint64_t budget, bool pin);
        uint8_t* at(uint32_t page) const { return slabs_[page / kPerSlab].first + (uint64_t) (page % kPerSlab) * kPage; }
        /// Whether `b` follows `a` in the same allocation (one copy can span both).
        static bool adjacent(uint32_t a, uint32_t b) { return b == a + 1 && b % kPerSlab != 0; }
        bool take(size_t n, std::vector<uint32_t>& out);   // appends n pages; false (none taken) when fewer are free
        void give(std::vector<uint32_t>& pages);           // all of them back (a shared page: one holder less), emptied
        void give(uint32_t page);
        /// Another holder of each of `pages` (a sequence that borrows a stashed one's prefix reads them too).
        std::vector<uint32_t> share(const std::vector<uint32_t>& pages);
        bool shared(uint32_t page) const { return ref_[page] > 1; }
        size_t free_pages() const { return free_.size(); }
        size_t pages() const { return total_; }
        uint64_t pinned_bytes() const { return pinned_; }
    private:
        static constexpr uint32_t kPerSlab = (uint32_t) (kSlab / kPage);
        std::vector<std::pair<uint8_t*, bool>> slabs_;   // base, page-locked
        std::vector<uint32_t> free_;
        std::vector<uint16_t> ref_;   // holders of each page
        size_t total_ = 0;
        uint64_t pinned_ = 0;
    };

    /// What one begin() moved, for its log line.
    struct SwitchCost {
        const char* from = nullptr;   // null: no copy (the live sequence went on)
        uint64_t out_bytes = 0, in_bytes = 0;
        int64_t out_cells = 0, out_new = 0, in_cells = 0;
        double out_ms = 0, ckpt_out_ms = 0, in_ms = 0, ckpt_in_ms = 0, idle_s = -1;
    };

    /// Cells of `seq` a prompt can continue from: the whole sequence when it is a prefix of the prompt (live
    /// only: the state is there), else its best checkpoint within the common prefix (its position; -1: none).
    int64_t reusable(const Seq& seq, const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs,
                     bool live) const;
    static Ckpt* find(Seq& seq, int64_t pos);
    /// A buffer for the live sequence's checkpoint at `pos`; a full pool gives up one (the header's policy).
    void* take_buffer(int64_t pos);
    void release(Seq& seq, int64_t keep_upto);   // frees the checkpoints after `keep_upto`
    /// Moves a copy of the live sequence to the stash: its K/V cells, a checkpoint of its end, and its
    /// checkpoints after `keep_upto` (the ones before stay with the live sequence, which continues from there).
    void stash_live(int64_t keep_upto);
    std::vector<Part> kv_parts(int64_t cells) const;
    /// `seq`'s pages enough for `cells` cells of every part, the least recently used stashed sequences going first
    /// when the arena is short (not one lending its prefix); false: no room.
    /// With `c0` > 0 (a live sequence's copy from c0 on), a page the copy would write that another sequence holds
    /// too is replaced first (cow_ marks the parts whose copy then starts at that page's start).
    bool ensure_pages(Seq& seq, int64_t cells, int64_t c0 = 0);
    /// The byte of part `i` (of kv_parts) from which a copy of the cells from `c0` on starts.
    uint64_t part_from(const std::vector<Part>& lo, const Part& p, size_t i, int64_t c0) const;
    void drop_kv(Seq& seq);   // its pages back to the arena
    /// The K/V cells [c0, c1) between the device and `seq`'s pages (c0 a multiple of the pages' and the indexer's
    /// blocks): each GPU's parts on a stream of its own, all of them in flight before the wait.
    bool kv_copy(Seq& seq, int64_t c0, int64_t c1, bool to_host);
    /// A cell below `valid` from which a copy of a live sequence's cells starts: what its next windows may still
    /// rewrite (the draft layer's last cells) lies after it.
    int64_t copy_floor(int64_t valid) const;
    /// The checkpoint at `pos` and, of a stashed sequence, its K/V cells: all of them, or with `prefix` only those
    /// before `pos` (a borrowed prefix).
    bool restore(Seq& seq, int64_t pos, bool prefix = false);
    int64_t begin_impl(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs);

    const ModelGeometry& g_;
    SessionState& ss_;
    std::vector<QsaState*> kv_;
    void* stream_;
    int max_ckpts_;
    uint64_t stash_budget_;
    uint64_t ckpt_bytes_;
    int allocated_ = 0;
    uint64_t clock_ = 0;
    Seq live_;
    std::vector<Seq> stash_;
    std::vector<void*> free_;
    std::vector<std::pair<void*, bool>> owned_;   // every buffer, and whether it is pinned
    HostPages pages_;
    std::vector<std::pair<int, void*>> streams_;   // the other GPUs' copy streams (device, stream)
    SwitchCost cost_;
    uint64_t moved_ = 0;   // bytes the last kv_copy calls queued
    std::vector<uint8_t> cow_;   // ensure_pages -> kv_copy: parts whose first page written was replaced
};

}  // namespace strata::core
