#pragma once

#include <argparse/argparse.hpp>

#include "hdc.hpp"

namespace common_args {
    void add_args(argparse::ArgumentParser& program) {
        // Optional arguments
        program.add_argument("-d", "--dim")
            .help("Number of dimensions.")
            .scan<'d', hdc::dim_t>()
            .default_value<size_t>(1000);
        program.add_argument("-r", "--retrain")
            .help("Number of retrains.")
            .scan<'d', size_t>()
            .default_value<size_t>(0);

        program.add_argument("--hdc").
            help("Choose the HDC type used between supported options. Values "
                 "accepted: {bin, int, float}.")
            .default_value("bin");
    }
}
