#include "../strategies/reference_strategy.hpp"
#include "csv_loader.hpp"
#include "replay_engine.hpp"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr
            << "Usage: ./quant_runner <csv_file>\n";

        return 1;
    }

    csot::CsvLoader loader;

    auto ticks =    loader.load(argv[1]);

    csot::ReferenceStrategy strategy;

    strategy.on_init();

    csot::ReplayEngine engine;

    engine.run(
        strategy,
        ticks
    );

    std::cout
        << "Ticks processed: "
        << ticks.size()
        << "\n\n";

    engine.histogram().print(std::cout);
    
    std::cout
        << "\nOrders generated: "
        << engine.orders_generated()
        << '\n';

    std::cout
        << "Fills processed: "
        << engine.fills_processed()
        << '\n';

    
    return 0;
}