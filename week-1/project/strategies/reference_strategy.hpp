#pragma once

#include "../include/strategy.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>


namespace csot {

class ReferenceStrategy : public Strategy {
public:
    void on_init() override;

    std::vector<Order> on_tick(const Tick& t) override;

    void on_fill(const Order& o,
                 double fill_price,
                 uint32_t fill_qty) override;

private:
    // -----------------------------------------------------------------------
    // Per-symbol state.
    // Lay out hot scalars FIRST so they land in the first cache-line (64 B).
    // The mids ring-buffer follows; all 4 symbols' data fits in ~2.3 KB —
    // well within a 32 KB L1.
    // -----------------------------------------------------------------------

    // struct alignas(64) SymbolState {
    //     // --- cache-line 0 (64 bytes) — touched every tick ---
    //     double        rolling_sum    = 0.0;
    //     double        rolling_sq_sum = 0.0;
    //     int32_t       position       = 0;
    //     uint32_t      head           = 0;
    //     uint32_t      count          = 0;
    //     // uint32_t      _pad           = 0;
    //     // const char*   symbol_ptr     = nullptr;   // intern ptr for O(1) compare
    //     // 8+8+4+4+4+4+8 = 40 B; pad to 64 B
    //     // char          _pad2[24]      = {};
 
    //     // --- cache-lines 1-8 (512 bytes) — ring buffer ---
    //     double        mids[64]       = {};

    // };

    struct SymbolEntry {
        const char* symbol_ptr = nullptr;
        std::string_view symbol{};
        bool used = false;
    };

    // struct PointerMapEntry {
    //     const char* ptr = nullptr;
    //     uint8_t idx = 0;
    // };


    // alignas(64) std::array<SymbolState,64> states_;

alignas(64) std::array<double,64> rolling_sum_{};
alignas(64) std::array<double,64> rolling_sq_sum_{};

alignas(64) std::array<uint32_t,64> head_{};
alignas(64) std::array<uint32_t,64> count_{};

alignas(64) std::array<int32_t,64> position_{};

alignas(64) std::array<std::array<double,64>,64> mids_{};


alignas(64)    std::array<SymbolEntry,64> symbols_;
    // std::array<PointerMapEntry,64> ptr_map_{};
    // uint8_t ptr_map_size_ = 0;

    // mutable std::vector<Order> order_buffer_;

    // size_t get_symbol_index(std::string_view symbol);


    __attribute__((always_inline))
    inline size_t get_symbol_index(std::string_view symbol)
    {
        const char* ptr = symbol.data();

        for (size_t i = 0; i < symbols_.size(); ++i)
        {
            if (symbols_[i].used &&
                symbols_[i].symbol_ptr == ptr)
            {
                return i;
            }
        }

        for (size_t i = 0; i < symbols_.size(); ++i)
        {
            if (!symbols_[i].used)
            {
                symbols_[i].used = true;
                symbols_[i].symbol = symbol;
                symbols_[i].symbol_ptr = ptr;
                return i;
            }
        }

        throw std::runtime_error("Maximum symbol count exceeded");
    }

// __attribute__((always_inline))
// inline size_t get_symbol_index(std::string_view symbol)
// {
//     size_t idx = 0;

//     for (size_t i = 3; i < symbol.size(); ++i)
//     {
//         idx = idx * 10 + (symbol[i] - '0');
//     }

//     return idx;
// }
// __attribute__((always_inline))
// inline size_t get_symbol_index(std::string_view symbol)
// {
//     const char* ptr = symbol.data();

//     for (uint8_t i = 0; i < ptr_map_size_; ++i)
//     {
//         if (ptr_map_[i].ptr == ptr)
//         {
//             return ptr_map_[i].idx;
//         }
//     }

//     if (ptr_map_size_ >= 64)
//     {
//         throw std::runtime_error(
//             "Maximum symbol count exceeded"
//         );
//     }

//     const uint8_t idx = ptr_map_size_;

//     ptr_map_[ptr_map_size_].ptr = ptr;
//     ptr_map_[ptr_map_size_].idx = idx;

//     ++ptr_map_size_;

//     return idx;
// }
};

} // namespace csot