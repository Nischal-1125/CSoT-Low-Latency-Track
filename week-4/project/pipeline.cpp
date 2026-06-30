



#include "pipeline.hpp"
#include <thread>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sched.h>
#include <immintrin.h>

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;

constexpr u32 WINDOW = 64;
constexpr double ENTRY_Z = 2.0;
constexpr double EXIT_Z = 0.5;
constexpr double EPSILON = 1e-9;
constexpr u32 PRICE_SCALE = 10000;
constexpr double WINDOW_INV = (1/64);

static inline std::string make_name(u32 id) {
    return "SYM" + std::to_string(id);
}

struct SymbolState {
    double window[WINDOW];
    u32    head;
    u32    count;
    int    position;
};

struct DecodedTick {
    csot::Tick tick;
    u32        symbol_id;
    u64        tick_index;
};

// ---- SPSC ring buffer (simple, non‑batched) --------------------------------
template <typename T, u32 Capacity>
class SpscRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two");
    static constexpr u32 MASK = Capacity - 1;

    // struct alignas(64) AtomicIndex {
    //     std::atomic<u64> v{0};
    // };

    T slots_[Capacity];

    // AtomicIndex tail_;
    // AtomicIndex head_;

    alignas(64) std::atomic<std::size_t> tail_{0};   // producer writes
    alignas(64) std::atomic<std::size_t> head_{0};   // consumer writes

public:
    void reset() {
        tail_.store(0, std::memory_order_relaxed);
        head_.store(0, std::memory_order_relaxed);
    }


    // PRODUCER ONLY. Returns false if the queue is full (back-pressure).
    bool try_push(const T& item) {
        const u64 t = tail_.load(std::memory_order_relaxed);   // own index: relaxed
        if (t - head_.load(std::memory_order_acquire) == Capacity)     // other's: acquire
            return false;                                              // full
        slots_[t & MASK] = item;                                      // write payload
        tail_.store(t + 1, std::memory_order_release);                 // publish (release)
        return true;
    }

    // CONSUMER ONLY. Returns false if the queue is empty.
    bool try_pop(T& out) {
        const u64 h = head_.load(std::memory_order_relaxed);   // own index: relaxed
        if (h == tail_.load(std::memory_order_acquire))                // other's: acquire
            return false;                                              // empty
        out = slots_[h & MASK];                                       // read payload
        head_.store(h + 1, std::memory_order_release);                 // publish (release)
        return true;
    }
};

static inline void pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

// ---- The pipeline ----------------------------------------------------------
class FastPipeline final : public csot::Pipeline {
    static constexpr u32 RING_CAPACITY = 8192;   // power‑of‑two

    u32 num_symbols_ = 0;
    std::vector<std::string> names_;
    std::vector<SymbolState> state_;
    SpscRing<DecodedTick, RING_CAPACITY> ring_;

    void process_tick(const DecodedTick& dt,
                      csot::OrderRecord* out,
                      std::size_t& num_orders) {
        const u32 sym = dt.symbol_id;
        const double bid = dt.tick.bid_px;
        const double ask = dt.tick.ask_px;
        const double mid = (bid + ask) * 0.5;

        SymbolState& st = state_[sym];

        // ---- Update rolling window (simple array) -------------------------
        st.window[st.head] = mid;
        st.head = (st.head + 1) & (WINDOW - 1);
        if (st.count < WINDOW) {
            ++st.count;
            if (st.count < WINDOW) return;   // warm‑up
        }

        // ---- Two‑pass mean and variance (reference arithmetic) -----------
        double sum = 0.0;
        for (double x : st.window) sum += x;
        const double mean = sum / static_cast<double>(WINDOW);
        // const double mean = sum * WINDOW_INV;

        double sq = 0.0;
        for (double x : st.window) {
            const double d = x - mean;
            sq += d * d;
        }
        const double variance = sq / static_cast<double>(WINDOW);
        // const double variance = sq * WINDOW_INV;
        const double stddev = std::sqrt(variance);
        if (stddev < EPSILON) return;

        const double z = (mid - mean) / stddev;
        const double abs_z = std::fabs(z);

        csot::Order order{};
        bool emit = false;

        if (st.position == 0) {
            if (z >= ENTRY_Z) {
                order = {csot::Order::Side::SELL, names_[sym], bid, 1};
                st.position = -1;
                emit = true;
            } else if (z <= -ENTRY_Z) {
                order = {csot::Order::Side::BUY, names_[sym], ask, 1};
                st.position = 1;
                emit = true;
            }
        } else if (st.position > 0 && abs_z <= EXIT_Z) {
            order = {csot::Order::Side::SELL, names_[sym], bid,
                     static_cast<u32>(st.position)};
            st.position = 0;
            emit = true;
        } else if (st.position < 0 && abs_z <= EXIT_Z) {
            order = {csot::Order::Side::BUY, names_[sym], ask,
                     static_cast<u32>(-st.position)};
            st.position = 0;
            emit = true;
        }

        if (emit) {
            out[num_orders].tick_index = dt.tick_index;
            out[num_orders].order = order;
            ++num_orders;
        }
    }

public:
    void on_init(u32 num_symbols) override {
        num_symbols_ = num_symbols;
        names_.resize(num_symbols);
        for (u32 i = 0; i < num_symbols; ++i)
            names_[i] = make_name(i);
        state_.assign(num_symbols, SymbolState{});
    }

    std::size_t run(const csot::WireTick* in,
                    std::size_t n,
                    csot::OrderRecord* out) override {

        // Pin consumer to core 3, producer to core 2 (adjust if needed)
        pin_to_core(3);

        std::atomic<bool> producer_done{false};

        // ---- Producer thread ---------------------------------------------
        std::thread producer([this, in, n, &producer_done] {
            pin_to_core(2);
            for (std::size_t i = 0; i < n; ++i) {
                __builtin_prefetch(in + i + 64);
                const auto& w = in[i];
                DecodedTick dt;
                dt.tick.timestamp_ns = w.timestamp_ns;
                dt.tick.symbol = names_[w.symbol_id];
                dt.tick.bid_px = static_cast<double>(w.bid_px_fp) / PRICE_SCALE;
                dt.tick.ask_px = static_cast<double>(w.ask_px_fp) / PRICE_SCALE;
                dt.tick.bid_qty = w.bid_qty;
                dt.tick.ask_qty = w.ask_qty;
                dt.symbol_id = w.symbol_id;
                dt.tick_index = i;

                while (!ring_.try_push(dt)) {
                    // __builtin_ia32_pause();
                }
            }
            // std::atomic_thread_fence(std::memory_order_release);
            producer_done.store(true, std::memory_order_release);
        });

        // ---- Consumer loop -----------------------------------------------
        std::size_t num_orders = 0;
        DecodedTick dt;

        while (true) {
            if (ring_.try_pop(dt)) {
                process_tick(dt, out, num_orders);
            } else {
                if (producer_done.load(std::memory_order_acquire)) {
                    if (!ring_.try_pop(dt)) break;
                    process_tick(dt, out, num_orders);
                } else {
                    // __builtin_ia32_pause();
                }
            }
        }

        producer.join();
        return num_orders;
    }
};

} // namespace

extern "C" csot::Pipeline* create_pipeline() {
    return new FastPipeline();
}