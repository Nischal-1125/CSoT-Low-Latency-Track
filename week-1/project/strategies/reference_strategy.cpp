#include "reference_strategy.hpp"

#include <cmath>
#include <stdexcept>

namespace {
    constexpr double INV64 = 1.0 / 64.0;
}


namespace csot {

void ReferenceStrategy::on_init() {
    // order_buffer_.reserve(1);
}



std::vector<Order> ReferenceStrategy::on_tick(const Tick& t) {

    size_t idx = get_symbol_index(t.symbol);

    // auto& st = states_[idx];

    const double mid = (t.bid_px + t.ask_px) * 0.5;

    // const double old_mid =
    //     st.mids[st.head];
const uint32_t head = head_[idx];

const double old_mid =
    mids_[idx][head];

    // st.mids[st.head] = mid;
mids_[idx][head] = mid;

    // st.rolling_sum += mid - old_mid;
rolling_sum_[idx] += mid - old_mid;

    // st.rolling_sq_sum +=
rolling_sq_sum_[idx] +=
        mid * mid
        - old_mid * old_mid;

    // st.head = (st.head + 1) & 63;
    head_[idx] = (head_[idx] + 1)&63;

    // if (st.count < 64)
    // {
    //     ++st.count;
    // }
if (__builtin_expect(count_[idx] < 64, 0))
{
    ++count_[idx];

    if (count_[idx] < 64)
    {
        return {};
    }
}
    // if (st.count < 64)
    // {
    //     return {};
    // }

    const double mean =
        // st.rolling_sum * INV64;
        rolling_sum_[idx]*INV64;

    const double variance =
        rolling_sq_sum_[idx] * INV64
        - mean * mean;

    if (variance <= 1e-18)
    {
        return {};
    }

    const double diff =
        mid - mean;

    const double diff_sq =
        diff * diff;

    // if (st.position == 0)
    if (__builtin_expect(position_[idx] == 0, 1))
    {
        if (diff > 0.0 &&
            diff_sq >= 4.0 * variance)
        {
            return {
                Order{
                    Order::Side::SELL,
                    t.symbol,
                    t.bid_px,
                    1
                }
            };
            // order_buffer_.push_back(
            //     Order{
            //         Order::Side::SELL,
            //         t.symbol,
            //         t.bid_px,
            //         1
            //     }
            // );

            // return order_buffer_;
        }

        if (diff < 0.0 &&
            diff_sq >= 4.0 * variance)
        {
            return {
                Order{
                    Order::Side::BUY,
                    t.symbol,
                    t.ask_px,
                    1
                }
            };
            // order_buffer_.push_back(
            //     Order{
            //         Order::Side::BUY,
            //         t.symbol,
            //         t.ask_px,
            //         1
            //     }
            // );

            // return order_buffer_;
        }

        return {};
    }

    // if (st.position > 0 &&
    //     diff_sq <= 0.25 * variance)
    // {
    if (__builtin_expect(position_[idx] > 0, 0) &&
        diff_sq <= 0.25 * variance)
    {
        return {
            Order{
                Order::Side::SELL,
                t.symbol,
                t.bid_px,
                static_cast<uint32_t>(position_[idx])
            }
        };
        // order_buffer_.push_back(
        //      Order{
        //          Order::Side::SELL,
        //          t.symbol,
        //          t.bid_px,
        //          static_cast<uint32_t>(st.position)
        //      }
        //  );

        //  return order_buffer_;
    }

    // if (st.position < 0 &&
    //     diff_sq <= 0.25 * variance)
    // {
    if (__builtin_expect(position_[idx] < 0, 0) &&
        diff_sq <= 0.25 * variance)
    {
        return {
            Order{
                Order::Side::BUY,
                t.symbol,
                t.ask_px,
                static_cast<uint32_t>(-position_[idx])
            }
        };
        // order_buffer_.push_back(
        //     Order{
        //         Order::Side::BUY,
        //         t.symbol,
        //         t.ask_px,
        //         static_cast<uint32_t>(-st.position)
        //     }
        //  );
        //  return order_buffer_;
    }

    return {};
}


void ReferenceStrategy::on_fill(
    const Order& o,
    double fill_price,
    uint32_t fill_qty)
{
    (void)fill_price;

    size_t idx = get_symbol_index(o.symbol);

    if (o.side == Order::Side::BUY)
    {
        position_[idx] += fill_qty;
    }
    else
    {
        position_[idx] -= fill_qty;
    }
}

} // namespace csot

extern "C" csot::Strategy* create_strategy()
{
    return new csot::ReferenceStrategy();
}