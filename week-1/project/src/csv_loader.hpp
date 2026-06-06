#pragma once

#include <vector>
#include <string>
#include <string_view>

#include "../include/strategy.hpp"



namespace csot {

class CsvLoader {
public:
    std::vector<Tick> load(const std::string& filename);

private:
    std::vector<std::string> symbol_storage_;
};

} // namespace csot