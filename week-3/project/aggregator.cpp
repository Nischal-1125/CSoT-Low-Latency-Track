#pragma GCC optimize("O3")

#include "aggregate.hpp"

#include <thread>
#include <limits>
#include <vector>
#include <cstdint>
#include <sched.h>

namespace {

constexpr int THREADS = 4;

using i64 = std::int64_t;

static constexpr i64 INF_POS = INT64_MAX;
static constexpr i64 INF_NEG = INT64_MIN;

struct alignas(64) PartialTable {
    csot::SymbolAgg agg[1024];
};

class FastAggregator final : public csot::Aggregator {
private:
    std::uint32_t num_symbols_ = 0;

    PartialTable partials_[THREADS];

static inline void pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

    static void process_chunk( const csot::AggTick* begin, const csot::AggTick* end, PartialTable& local) {
        
        // pin_to_core(static_cast<int>(std::this_thread::get_id())); // FIXME: map id to core
        // Better: pass core id via lambda
        for (const csot::AggTick* p = begin; p != end; ++p) {
            // Prefetch a few ticks ahead (adjust distance as needed)
            // __builtin_prefetch(p + 16);
            // __builtin_prefetch(p + 32);
            __builtin_prefetch(p+128);                                       // ---> best
            // __builtin_prefetch(p+256); 

            const auto& tick = *p;

            auto& r = local.agg[tick.symbol_id];

            ++r.count;
            r.sum_price += tick.price;
            r.sum_qty += tick.qty;

            // if (tick.price < r.min_price)
            //     r.min_price = tick.price;

            // if (tick.price > r.max_price)
            //     r.max_price = tick.price;
            r.min_price = (tick.price < r.min_price) ? tick.price : r.min_price;
            r.max_price = (tick.price > r.max_price) ? tick.price : r.max_price;
        }
    }



public:
    void on_init(std::uint32_t num_symbols) override {
        num_symbols_ = num_symbols;

        for (int t = 0; t < THREADS; ++t) {
            for (std::uint32_t s = 0; s < num_symbols_; ++s) {

                auto& r = partials_[t].agg[s];

                r.count = 0;
                r.sum_price = 0;
                r.sum_qty = 0;

                r.min_price = INF_POS;
                    // std::numeric_limits<std::int64_t>::max();

                r.max_price = 0;
                // INF_NEG;
                    // std::numeric_limits<std::int64_t>::min();
            }
        }
    }

    void run( const csot::AggTick* ticks, std::size_t n, csot::SymbolAgg* out) override {
        // Reset partial tables
        for (int t = 0; t < THREADS; ++t) {
            for (std::uint32_t s = 0; s < num_symbols_; ++s) {

                auto& r = partials_[t].agg[s];

                r.count = 0;
                r.sum_price = 0;
                r.sum_qty = 0;

                r.min_price = INF_POS;
                    // std::numeric_limits<std::int64_t>::max();

                r.max_price = 0;
                    // std::numeric_limits<std::int64_t>::min();
            }
        }
    {
        std::vector<std::jthread> workers;
        workers.reserve(THREADS);

        // const std::size_t chunk = n / THREADS;

        // for (unsigned t = 0; t < THREADS; ++t) {
        //     const std::size_t start = t * chunk;
        //     const std::size_t end = (t == THREADS - 1) ? n : start + chunk;
        //     workers.emplace_back([this, t, start, end, ticks] {
        //         // Pin this thread to core t (if t < number of cores)
        //         pin_to_core(static_cast<int>(t));
        //         process_chunk(ticks + start, ticks + end, partials_[t]);
        //     });
        // }
        for (unsigned t = 0; t < THREADS; ++t) {
        
            const std::size_t start =
                n * t / THREADS;
        
            const std::size_t end =
                n * (t + 1) / THREADS;
        
            workers.emplace_back([this, t, start, end, ticks] {
            
                pin_to_core(t);
            
                process_chunk(
                    ticks + start,
                    ticks + end,
                    partials_[t]
                );
            });
        }
    }
        // jthread destructors join automatically

        // Merge partials into final output
        for (std::uint32_t s = 0; s < num_symbols_; ++s) {
            std::uint64_t count = 0;
            std::int64_t  sum_price = 0;
            std::uint64_t sum_qty = 0;
            std::int64_t  mn = INF_POS;
            std::int64_t  mx = 0;

            for (unsigned t = 0; t < THREADS; ++t) {
                const auto& r = partials_[t].agg[s];
                if (r.count == 0) continue;
                count += r.count;
                sum_price += r.sum_price;
                sum_qty += r.sum_qty;
                if (r.min_price < mn) mn = r.min_price;
                if (r.max_price > mx) mx = r.max_price;
            }

            if (count == 0) {
                // out[s] = csot::SymbolAgg{0, 0, 0, 0, 0};
                out[s].count = 0;
                out[s].sum_price = 0;
                out[s].min_price = 0;
                out[s].max_price = 0;
                out[s].sum_qty = 0;
            } else {
                out[s].count = count;
                out[s].sum_price = sum_price;
                out[s].min_price = mn;
                out[s].max_price = mx;
                out[s].sum_qty = sum_qty;
            }
        }
    }
};

} // namespace

extern "C" csot::Aggregator* create_aggregator() {
    return new FastAggregator();
}