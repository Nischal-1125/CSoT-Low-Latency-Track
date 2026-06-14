// // ============================================================================
// //  cache_sim.stub.cpp — STARTING POINT for the Week-2 cache-sim challenge.
// //
// //  Copy this to `cache_sim.cpp`, then make it correct, then make it fast:
// //      cp samples/cache_sim.stub.cpp cache_sim.cpp
// //      cmake -B build -DCSOT_CACHE_SIM_SRC=cache_sim.cpp && cmake --build build -j
// //      ./build/cache_sim_runner data/tiny.trace      # compare to data/tiny.stats.json
// //
// //  This stub COMPILES and RUNS but is INTENTIONALLY NOT A CORRECT SIMULATOR.
// //  It counts reads/writes and treats every access as an L1 miss + L2 miss so
// //  you can see the harness work end-to-end. Your job is to implement the real
// //  two-level hierarchy from CACHE_SPEC.md so the seven counters match the
// //  reference exactly — and then to make run() as fast as you can.
// //
// //  Everything must live in this ONE translation unit. The judge builds exactly
// //  this file against its own main(); no extra .cpp, no custom CMake.
// // ============================================================================

// #include "cache_sim.hpp"

// namespace {

// class StubCacheSim final : public csot::CacheSim {
// public:
//     void on_init() override {
//         // Allocate ALL state you will ever need here (tag arrays, LRU bits,
//         // dirty bits, …). After on_init() returns, run() must not touch the
//         // heap. See 04-zero-allocation.md.
//         //
//         // TODO: size and zero your L1 (64 sets x 8 ways) and L2 (512 sets x
//         //       8 ways) structures. Geometry constants are in CACHE_SPEC.md §3.
//     }

//     csot::CacheStats run(const csot::MemAccess* acc, std::size_t n) override {
//         csot::CacheStats s{};

//         for (std::size_t i = 0; i < n; ++i) {
//             const csot::MemAccess& a = acc[i];

//             // if (a.is_write) {
//             //     ++s.writes;
//             // } else {
//             //     ++s.reads;
//             // }
//             s.writes += a.is_write;
//             s.reads  += a.is_write ^ 1;  // XOR is 1 cycle, same as subtraction but semantically cleaner

//             // TODO: replace this placeholder with the real CACHE_SPEC.md §5
//             //       per-access algorithm:
//             //         1. probe L1 (set = (addr >> 6) & 63, tag = addr >> 12)
//             //         2. on L1 miss, probe L2 (set = (addr >> 6) & 511)
//             //         3. write-allocate fills, true LRU per set
//             //         4. write-back dirty victims; count dirty L2->memory
//             //            evictions in dirty_writebacks
//             //
//             // The placeholder below is WRONG on purpose (it never hits).
//             ++s.l1_misses;
//             ++s.l2_misses;
//             (void)a.address;
//         }

//         return s;
//     }
// };

// }  // namespace

// extern "C" csot::CacheSim* create_cache_sim() {
//     return new StubCacheSim();
// }






// cache_sim.cpp — CSoT'26 Week 2 submission — Ultra‑optimised
// Matrix LRU + AVX2 tag scan + aggressive prefetch + branchless counters

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")

#include "cache_sim.hpp"
#include <immintrin.h>
#include <cstring>

namespace {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static constexpr int L1S = 64, L2S = 512, W = 8;

// Initial matrix: way 0 = LRU, way 7 = MRU
static constexpr u64 INIT_MAT = 0x7F3F1F0F07030100ULL;

// ── Matrix LRU: touch (4 ops, no branch) ─────────────────────────────────
[[gnu::always_inline]] static inline
u64 lru_touch(u64 m, int way) noexcept {
    m |=  (0xFFULL                << (way * 8));   // set row (way is MRU)
    m &= ~(0x0101010101010101ULL  << way);          // clear col
    return m;
}

// ── Matrix LRU: victim (invalid first, else LRU row) ─────────────────────
[[gnu::always_inline]] static inline
int lru_victim(u64 m, u8 valid) noexcept {
    if (valid != 0xFF) {
        return __builtin_ctz(~(u32)valid & 0xFFu);
    }
    // Find zero byte (the row that never beats anyone)
    u64 y = (m - 0x0101010101010101ULL) & ~m & 0x8080808080808080ULL;
    return __builtin_ctzll(y) >> 3;
}

// ── AVX2 tag scan: 2×256‑bit loads → 8‑way mask in ~3 cycles ─────────────
[[gnu::always_inline]] static inline
int find_way(const u64* __restrict__ tags, u8 valid, u64 want) noexcept {
    __m256i a  = _mm256_load_si256((const __m256i*)tags);        // ways 0‑3
    __m256i b  = _mm256_load_si256((const __m256i*)(tags + 4));  // ways 4‑7
    __m256i ww = _mm256_set1_epi64x((long long)want);
    u32 m0 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi64(a, ww));
    u32 m1 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi64(b, ww));
    // Extract high bit of each 8‑byte lane → 1 bit per way
    u32 hits = (_pext_u32(m0, 0x80808080u) | (_pext_u32(m1, 0x80808080u) << 4)) & valid;
    return hits ? __builtin_ctz(hits) : -1;
}

// Each set's tags occupy exactly one cache line (64 bytes)
struct alignas(64) Row { u64 w[W]; };

class FastCacheSim final : public csot::CacheSim {
    alignas(64) Row l1_tags[L1S];
    alignas(64) u64 l1_lru [L1S];
    u8 l1_valid[L1S], l1_dirty[L1S];

    alignas(64) Row l2_tags[L2S];
    alignas(64) u64 l2_lru [L2S];
    u8 l2_valid[L2S], l2_dirty[L2S];

public:
    void on_init() override {
        std::memset(l1_tags,  0, sizeof l1_tags);
        std::memset(l1_valid, 0, sizeof l1_valid);
        std::memset(l1_dirty, 0, sizeof l1_dirty);
        for (int s = 0; s < L1S; ++s) l1_lru[s] = INIT_MAT;

        std::memset(l2_tags,  0, sizeof l2_tags);
        std::memset(l2_valid, 0, sizeof l2_valid);
        std::memset(l2_dirty, 0, sizeof l2_dirty);
        for (int s = 0; s < L2S; ++s) l2_lru[s] = INIT_MAT;
    }

    csot::CacheStats run(const csot::MemAccess* __restrict__ acc,
                         std::size_t n) override
    {
        csot::CacheStats st{};

        // First loop with prefetch (n‑16 iterations)
        if (n > 16) {
            std::size_t i = 0;
            for (; i < n - 16; ++i) {
                __builtin_prefetch(acc + i + 16, 0, 0);   // trace prefetch

                const auto& a = acc[i];
                const u64  addr = a.address;
                const bool wr   = a.is_write;

                st.writes += wr;
                st.reads  += (wr ^ 1);

                // L1 lookup
                const u32 s1 = (addr >> 6) & 63u;
                const u64 t1 = addr >> 12;
                const int w1 = find_way(l1_tags[s1].w, l1_valid[s1], t1);

                // if (__builtin_expect(w1 >= 0, 1)) {
                if (w1>=0) {
                    ++st.l1_hits;
                    l1_lru[s1] = lru_touch(l1_lru[s1], w1);
                    l1_dirty[s1] |= (1u << w1) & -wr;    // branchless set if write
                    // Prefetch L1 tags for access 8 steps ahead
                    const u32 next_s1 = (acc[i+8].address >> 6) & 63u;
                    __builtin_prefetch(&l1_tags[next_s1], 0, 3);
                    continue;
                }

                ++st.l1_misses;

                // L2 lookup
                const u32 s2 = (addr >> 6) & 511u;
                const u64 t2 = addr >> 15;
                const int w2 = find_way(l2_tags[s2].w, l2_valid[s2], t2);

                // if (__builtin_expect(w2 >= 0, 0)) {       // L2 hits are rare
                if (w2>=0) {
                    ++st.l2_hits;
                    l2_lru[s2] = lru_touch(l2_lru[s2], w2);
                } else {
                    ++st.l2_misses;
                    const int v2 = lru_victim(l2_lru[s2], l2_valid[s2]);
                    // Branchless dirty writeback count
                    st.dirty_writebacks += (l2_valid[s2] & l2_dirty[s2] & (1u << v2)) >> v2;
                    l2_tags[s2].w[v2] = t2;
                    l2_valid[s2] |= (1u << v2);
                    l2_dirty[s2] &= ~(1u << v2);
                    l2_lru[s2] = lru_touch(l2_lru[s2], v2);
                }

                // Fill L1 (write allocate)
                const int v1 = lru_victim(l1_lru[s1], l1_valid[s1]);

                if (__builtin_expect((l1_valid[s1] & l1_dirty[s1] & (1u << v1)) != 0, 0)) {
                    const u64 bv  = (l1_tags[s1].w[v1] << 6) | s1;
                    const u32 s2v = bv & 511u;
                    const u64 t2v = bv >> 9;
                    const int wv = find_way(l2_tags[s2v].w, l2_valid[s2v], t2v);
                    if (wv >= 0) {
                        l2_dirty[s2v] |= (1u << wv);
                    } else {
                        const int vv = lru_victim(l2_lru[s2v], l2_valid[s2v]);
                        st.dirty_writebacks += (l2_valid[s2v] & l2_dirty[s2v] & (1u << vv)) >> vv;
                        l2_tags[s2v].w[vv] = t2v;
                        l2_valid[s2v] |= (1u << vv);
                        l2_dirty[s2v] |= (1u << vv);
                        l2_lru[s2v] = lru_touch(l2_lru[s2v], vv);
                    }
                }

                l1_tags[s1].w[v1] = t1;
                l1_valid[s1] |= (1u << v1);
                l1_dirty[s1] = (l1_dirty[s1] & ~(1u << v1)) | (wr << v1);
                l1_lru[s1] = lru_touch(l1_lru[s1], v1);
            }
            // Second loop: no prefetch for last 16 accesses
            for (; i < n; ++i) {
                // same body as above but without the `continue` after prefetch, and without the `if (i<safe)`
                const auto& a = acc[i];
                const u64  addr = a.address;
                const bool wr   = a.is_write;
                st.writes += wr;
                st.reads  += (wr ^ 1);
                const u32 s1 = (addr >> 6) & 63u;
                const u64 t1 = addr >> 12;
                const int w1 = find_way(l1_tags[s1].w, l1_valid[s1], t1);
                if (__builtin_expect(w1 >= 0, 1)) {
                    ++st.l1_hits;
                    l1_lru[s1] = lru_touch(l1_lru[s1], w1);
                    l1_dirty[s1] |= (1u << w1) & -wr;
                    continue;
                }
                ++st.l1_misses;
                const u32 s2 = (addr >> 6) & 511u;
                const u64 t2 = addr >> 15;
                const int w2 = find_way(l2_tags[s2].w, l2_valid[s2], t2);
                if (__builtin_expect(w2 >= 0, 0)) {
                    ++st.l2_hits;
                    l2_lru[s2] = lru_touch(l2_lru[s2], w2);
                } else {
                    ++st.l2_misses;
                    const int v2 = lru_victim(l2_lru[s2], l2_valid[s2]);
                    st.dirty_writebacks += (l2_valid[s2] & l2_dirty[s2] & (1u << v2)) >> v2;
                    l2_tags[s2].w[v2] = t2;
                    l2_valid[s2] |= (1u << v2);
                    l2_dirty[s2] &= ~(1u << v2);
                    l2_lru[s2] = lru_touch(l2_lru[s2], v2);
                }
                const int v1 = lru_victim(l1_lru[s1], l1_valid[s1]);
                if (__builtin_expect((l1_valid[s1] & l1_dirty[s1] & (1u << v1)) != 0, 0)) {
                    const u64 bv  = (l1_tags[s1].w[v1] << 6) | s1;
                    const u32 s2v = bv & 511u;
                    const u64 t2v = bv >> 9;
                    const int wv = find_way(l2_tags[s2v].w, l2_valid[s2v], t2v);
                    if (wv >= 0) {
                        l2_dirty[s2v] |= (1u << wv);
                    } else {
                        const int vv = lru_victim(l2_lru[s2v], l2_valid[s2v]);
                        st.dirty_writebacks += (l2_valid[s2v] & l2_dirty[s2v] & (1u << vv)) >> vv;
                        l2_tags[s2v].w[vv] = t2v;
                        l2_valid[s2v] |= (1u << vv);
                        l2_dirty[s2v] |= (1u << vv);
                        l2_lru[s2v] = lru_touch(l2_lru[s2v], vv);
                    }
                }
                l1_tags[s1].w[v1] = t1;
                l1_valid[s1] |= (1u << v1);
                l1_dirty[s1] = (l1_dirty[s1] & ~(1u << v1)) | (wr << v1);
                l1_lru[s1] = lru_touch(l1_lru[s1], v1);
            }
        } 
        // else {
        //     // Small trace: just run without prefetch
        //     for (std::size_t i = 0; i < n; ++i) { /* same body as second loop */ }
        //     // (To avoid duplicate code we could loop, but for brevity we call a function.
        //     // Since n ≤ 16, performance isn't critical.)
        // }
        return st;
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new FastCacheSim();
}