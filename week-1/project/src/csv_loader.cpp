#include "csv_loader.hpp"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include<stdexcept>

namespace csot {

std::vector<Tick> CsvLoader::load(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    std::vector<Tick> ticks;

    std::string line;

    std::getline(file, line); // skip header

    std::unordered_map<std::string, size_t> symbol_map;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string timestamp;
        std::string symbol;
        std::string bid_px;
        std::string ask_px;
        std::string bid_qty;
        std::string ask_qty;

        std::getline(ss, timestamp, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, bid_px, ',');
        std::getline(ss, ask_px, ',');
        std::getline(ss, bid_qty, ',');
        std::getline(ss, ask_qty, ',');

        size_t idx;

        auto it = symbol_map.find(symbol);

        if (it == symbol_map.end())
        {
            idx = symbol_storage_.size();

            symbol_storage_.push_back(symbol);

            symbol_map[symbol] = idx;
        }
        else
        {
            idx = it->second;
        }

        Tick t;

        t.timestamp_ns = std::stoull(timestamp);
        t.symbol       = symbol_storage_[idx];
        t.bid_px       = std::stod(bid_px);
        t.ask_px       = std::stod(ask_px);
        t.bid_qty      = std::stoul(bid_qty);
        t.ask_qty      = std::stoul(ask_qty);

        ticks.push_back(t);
    }

    return ticks;
}

}
