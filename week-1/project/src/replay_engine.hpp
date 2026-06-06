#pragma once

#include "../include/histogram.hpp"
#include "../include/strategy.hpp"

#include <vector>

namespace csot {

class ReplayEngine {
public:
    void run(
        Strategy& strategy,
        const std::vector<Tick>& ticks
    );

    const LatencyHistogram& histogram() const
    {
        return histogram_;
    }

    uint64_t orders_generated() const
    {
        return orders_generated_;
    }

    uint64_t fills_processed() const
    {
        return fills_processed_;
    }

private:
    LatencyHistogram histogram_;

    uint64_t orders_generated_ = 0;
    uint64_t fills_processed_  = 0;
};

}