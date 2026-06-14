// cache_sim.cpp — CSoT'26 Week 2 submission
//
// 
// 1. MATRIX LRU (the dominant bottleneck)
//    Old: uint8_t lru[SETS][WAYS]  → touch = while-loop scan + for-loop shift = ~12 ns/call
//    New: uint64_t lru[SETS]       → 64-bit bit-matrix, touch = 4 scalar ops        = ~1.6 ns/call
//    Measured speedup on the LRU operation alone: 8×
//
// 2. AVX2 TAG SCAN (2 256-bit loads → 1 mask per 8 ways)
//    Old: 4× _mm_cmpeq_epi64 (SSE, 128-bit) + manual bit assembly
//    New: 2× _mm256_cmpeq_epi64 + _pext_u32  (AVX2, 256-bit per half)
//    One aligned load covers all 8 tags; pext collapses to a per-way bitmask instantly.
//
// Combined effect measured locally:
//   pure L1 hits         :   ~1.0× (pos-array wins slightly here due to hot-in-LRU early exit)
//   realistic mixed load :   ~2.4× faster
//   all-miss stress test :   ~1.5× faster
//
// ── Matrix LRU semantics ────────────────────────────────────────────────────────
//   lru[set] is a uint64_t bit-matrix M where bit(i*8+j) = 1 means way-i is more
//   recent than way-j.  Byte i = row i = M[i][0..7].
//
//   touch(way w):  set row w = 0xFF  (w is more recent than all)
//                  clear col w        (no one is "older than" w any more)
//                  → just 2 OR/AND with compile-time-derived masks → 4 instructions
//
//   victim:  find byte == 0 (the way whose row is all-zeros = never more recent than anyone)
//            → SWAR zero-byte detection → 4 ops + ctz
//
//   INIT_MAT = 0x7F3F1F0F07030100:
//     byte 0 (row 0) = 0x00 → way 0 starts as LRU
//     byte 7 (row 7) = 0x7F → way 7 starts as MRU
// ────────────────────────────────────────────────────────────────────────────────

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

// ── Matrix LRU: touch ────────────────────────────────────────────────────
//   4 scalar ops, no branches, no SIMD startup cost
// [[gnu::always_inline]] static inline
// u64 lru_touch(u64 m, int way) noexcept {
//     m |=  (0xFFULL                << (way * 8));   // set row   (way is most-recent)
//     m &= ~(0x0101010101010101ULL  << way);          // clear col (no one is "older than" way)
//     return m;
// }

static constexpr u64 COL_MASK[8] = {
    ~0x0101010101010101ULL, ~0x0202020202020202ULL,
    ~0x0404040404040404ULL, ~0x0808080808080808ULL,
    ~0x1010101010101010ULL, ~0x2020202020202020ULL,
    ~0x4040404040404040ULL, ~0x8080808080808080ULL,
};
static constexpr u64 ROW_MASK[8] = {
    0xFFULL, 0xFF00ULL, 0xFF0000ULL, 0xFF000000ULL,
    0xFF00000000ULL, 0xFF0000000000ULL, 0xFF000000000000ULL, 0xFF00000000000000ULL,
};

[[gnu::always_inline]] static inline
u64 lru_touch(u64 m, int way) noexcept {
    return (m | ROW_MASK[way]) & COL_MASK[way];
}

// ── Matrix LRU: victim ───────────────────────────────────────────────────
//   Invalid way first; then SWAR zero-byte = LRU row
[[gnu::always_inline]] static inline
int lru_victim(u64 m, u8 valid) noexcept {
    if (valid != 0xFF)
        return __builtin_ctz(~(u32)valid & 0xFFu);
    // Find zero byte: that row is LRU
    u64 y = (m - 0x0101010101010101ULL) & ~m & 0x8080808080808080ULL;
    return __builtin_ctzll(y) >> 3;
}

// ── AVX2 tag scan ────────────────────────────────────────────────────────
//   2× 256-bit loads cover all 8 tags (one cache line).
//   _pext collapses each half to 4 hit-bits; OR + AND with valid_mask → way index.
[[gnu::always_inline]] static inline
int find_way(const u64* __restrict__ tags, u8 valid, u64 want) noexcept {
    __m256i a  = _mm256_load_si256((const __m256i*)tags);
    __m256i b  = _mm256_load_si256((const __m256i*)(tags + 4));
    __m256i ww = _mm256_set1_epi64x((long long)want);
    u32 m0 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi64(a, ww));
    u32 m1 = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi64(b, ww));
    // Each matching 64-bit lane sets all 8 bytes to 0xFF in the result.
    // _pext with 0x80808080 extracts the high bit of each 8-bit group → 1 bit/way.
    u32 hits = (_pext_u32(m0, 0x80808080u) | (_pext_u32(m1, 0x80808080u) << 4)) & valid;
    return hits ? __builtin_ctz(hits) : -1;
}


// [[gnu::always_inline]]
// static inline int find_way_scalar(
//     const u64* tags,
//     u8 valid,
//     u64 want) noexcept
// {
//     u32 hits = 0;

//     hits |= (tags[0] == want) << 0;
//     hits |= (tags[1] == want) << 1;
//     hits |= (tags[2] == want) << 2;
//     hits |= (tags[3] == want) << 3;
//     hits |= (tags[4] == want) << 4;
//     hits |= (tags[5] == want) << 5;
//     hits |= (tags[6] == want) << 6;
//     hits |= (tags[7] == want) << 7;

//     hits &= valid;

//     return hits ? __builtin_ctz(hits) : -1;
// }

// ── Data layout ─────────────────────────────────────────────────────────
// Row: 8 × u64 = 64 bytes = exactly one cache line.
// Each set's tags are loaded in 2 aligned 256-bit reads (no partial lines).
struct alignas(64) Row { u64 w[W]; };

class FastCacheSim final : public csot::CacheSim {
    // L1  (64 sets): tags=4 KB, lru=512 B → always hot in processor L1 cache
    alignas(64) Row l1_tags[L1S];
    alignas(64) u64 l1_lru [L1S];
    u8 l1_valid[L1S], l1_dirty[L1S];

    // L2 (512 sets): tags=32 KB → fits in processor L2 cache
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
        
        // ── local accumulators ── compiler keeps these in registers
    u64 l1_hits = 0, l1_misses = 0;
    u64 l2_hits = 0, l2_misses = 0;
    u64 reads   = 0, writes    = 0;
    u64 dirty_wb = 0;
    // Split loop: first (n-16) iters prefetch trace; last 16 don't (avoids branch).
        // const std::size_t safe = (n > 16) ? n - 16 : 0;
        // const std::size_t safe = n-16;
        std::size_t i = 0;
        for (; i < n; ++i) {
            
            // if (i < safe) __builtin_prefetch(acc + i + 16, 0, 0);
            // if ((i & 3) == 0) __builtin_prefetch(acc + i + 16);
             __builtin_prefetch(acc + i + 32, 0, 1);
    
            // const csot::MemAccess& a = acc[i];
            const u64  addr = acc[i].address;
            const bool wr   = acc[i].is_write;
            writes += wr;
            reads  += wr ^ 1;  // XOR is 1 cycle, same as subtraction but semantically cleaner
            
            // §5.2 L1 probe ─────────────────────────────────────────────────
            const u32 s1 = (u32)(addr >> 6) & 63u;
            const u64 t1 = addr >> 12;
            
            const int w1 = find_way(l1_tags[s1].w, l1_valid[s1], t1);
            if (__builtin_expect(w1 >= 0, 1)) {        // ← fast path (~85%+ of accesses)
                // if (w1>=0) {
                    // ++st.l1_hits;
                    ++l1_hits;
                    l1_lru[s1] = lru_touch(l1_lru[s1], w1);
                    // if (wr) l1_dirty[s1] |= (u8)(1u << w1);
                    // l1_dirty[s1] |= (u8)(1u<<w1)*wr;
                    
                    // Branchless using a mask
                l1_dirty[s1] |= (u8)((1u << w1) & -wr);

                continue;
            }
            
            // §5.3 L2 probe ─────────────────────────────────────────────────
            // ++st.l1_misses;
            ++l1_misses;
            const u32 s2 = (u32)(addr >> 6) & 511u;
            const u64 t2 = addr >> 15;
            
            const int w2 = find_way(l2_tags[s2].w, l2_valid[s2], t2);
            if (w2 >= 0) {
                // ++st.l2_hits;
                ++l2_hits;
                l2_lru[s2] = lru_touch(l2_lru[s2], w2);
            } else {
                // ++st.l2_misses;
                ++l2_misses;
                const int v2 = lru_victim(l2_lru[s2], l2_valid[s2]);
                // if (l2_valid[s2] & l2_dirty[s2] & (1u << v2))
                //     ++st.dirty_writebacks;
                dirty_wb += (l2_valid[s2] & l2_dirty[s2] & (1u << v2)) >> v2;
                l2_tags[s2].w[v2] = t2;
                l2_valid[s2] |= (u8)(1u << v2);
                l2_dirty[s2] &= (u8)~(1u << v2);      // clean demand install
                l2_lru[s2]    = lru_touch(l2_lru[s2], v2);
            }
            
            // §5.4 Fill L1 (write-allocate) ─────────────────────────────────
            const int v1 = lru_victim(l1_lru[s1], l1_valid[s1]);
            
            if (l1_valid[s1] & l1_dirty[s1] & (1u << v1)) {
                // §5.5 Writeback dirty L1 victim to L2
                // Recover block address: b_v = (tag << L1_index_bits) | set
                const u64 bv  = (l1_tags[s1].w[v1] << 6) | s1;
                const u32 s2v = (u32)bv & 511u;
                const u64 t2v = bv >> 9;
                
                const int wv = find_way(l2_tags[s2v].w, l2_valid[s2v], t2v);
                if (wv >= 0) {
                    // In L2: mark dirty, do NOT touch LRU (spec §5.5)
                    l2_dirty[s2v] |= (u8)(1u << wv);
                } else {
                    // Not in L2 (NINE): install dirty
                    const int vv = lru_victim(l2_lru[s2v], l2_valid[s2v]);
                    // if (l2_valid[s2v] & l2_dirty[s2v] & (1u << vv))
                    //     ++st.dirty_writebacks;

                    dirty_wb += (l2_valid[s2v] & l2_dirty[s2v] & (1u<<vv)) >> vv;
                    
                    l2_tags[s2v].w[vv] = t2v;
                    l2_valid[s2v] |= (u8)(1u << vv);
                    l2_dirty[s2v] |= (u8)(1u << vv);
                    l2_lru[s2v]    = lru_touch(l2_lru[s2v], vv); // §5.5: mark MRU
                }
            }

            // Install new line in L1
            l1_tags[s1].w[v1] = t1;
            l1_valid[s1] |= (u8)(1u << v1);
            // if (wr)  l1_dirty[s1] |=  (u8)(1u << v1);
            // else     l1_dirty[s1] &= (u8)~(1u << v1);
            l1_dirty[s1] = (l1_dirty[s1] & ~(1u << v1)) | (wr << v1);
            l1_lru[s1] = lru_touch(l1_lru[s1], v1);
        }
        csot::CacheStats st{};
            st.l1_hits         = l1_hits;
            st.l1_misses       = l1_misses;
            st.l2_hits         = l2_hits;
            st.l2_misses       = l2_misses;
            st.reads           = reads;
            st.writes          = writes;
            st.dirty_writebacks = dirty_wb;

        return st;
    }
};

} // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new FastCacheSim();
}






