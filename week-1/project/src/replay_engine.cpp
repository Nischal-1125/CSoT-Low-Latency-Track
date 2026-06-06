#include "replay_engine.hpp"

#include <chrono>

namespace csot {

void ReplayEngine::run(
    Strategy& strategy,
    const std::vector<Tick>& ticks)
{
    for (const auto& tick : ticks)
    {
        auto start =
            std::chrono::high_resolution_clock::now();

        auto orders =
            strategy.on_tick(tick);

        auto end =
            std::chrono::high_resolution_clock::now();

        auto latency =
            std::chrono::duration_cast<
                std::chrono::nanoseconds
            >(end - start).count();

        histogram_.record(
            static_cast<uint64_t>(latency)
        );

        for (const auto& order : orders)
        {
            ++orders_generated_;
            
            strategy.on_fill(
                order,
                order.price,
                order.qty
            );

            ++fills_processed_;
        }
    }
}

}