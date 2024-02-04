#include <cassert>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>

#include "AssociativeMemory.hpp"
#include "ItemMemory.hpp"
#include "common_args.hpp"
#include "hdc.hpp"

using lang_t = std::vector<std::string>;
using dataset_t = std::vector<lang_t>;

std::vector<std::string> languages = {
    "bul",
    "ces",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "fin",
    "fra",
    "hun",
    "ita",
    "lav",
    "lit",
    "nld",
    "pol",
    "por",
    "ron",
    "slk",
    "slv",
    "spa",
    "swe"
};

// Dataset parsers
lang_t read_lang_file(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }
    lang_t lines;

    std::string line;
    while (std::getline(f, line)) {
        lines.emplace_back(line);
    }

    return lines;
}

dataset_t read_dataset(const std::string &dataset_path) {
    dataset_t dataset;

    //for (const auto &lang_path : language_files) {
    for (const auto &lang : languages) {
        dataset.emplace_back(read_lang_file(dataset_path+"/"+lang+".txt"));
    }

    return dataset;
}

template<typename VectorType>
VectorType encode_3gram(const hdc::ItemMemory<VectorType> &im, const char *str) {
    // ASCII table offset to the first lower char, i.e., the letter "a". The IM
    // is 27-entry memory packed as 26 letters and the space (' ') entry. The
    // input text in the *str variable must contain only lower-case letters
    // without punctuation.
    const int FIRST_lOWER_CHAR = 97;
    std::vector<VectorType> items;
    for (int i = 0; i < 3; i++, str++) {
        if (*str != ' ') {
            // Assign the lower-case letter to its IM correspondent
            items.emplace_back(im.at(*str-FIRST_lOWER_CHAR));
        }
        else {
            // Assign the space char as the last item in the IM
            items.emplace_back(im.back());
        }
    }
    // Permute the items
    items[0] = hdc::p(items[0], 2);
    items[1] = hdc::p(items[1], 1);

    // Return the trigram HV
    return hdc::mul(items);
}

template<typename VectorType>
VectorType encode_query(const hdc::ItemMemory<VectorType> &im, const char *str) {
    std::vector<VectorType> n_grams;

    // Encode n-grams. Since we create 3-grams, loop until the pointer of str+3
    // is different from the null character.
    while (*(str+3)) {
        n_grams.emplace_back(encode_3gram(im, str));
        str++;
    }

    return hdc::add(n_grams);
}

template<typename VectorType>
VectorType train_language(
        const hdc::ItemMemory<VectorType> &im,
        const lang_t &lang
        ) {
    std::vector<VectorType> query_vectors;

    for (auto &line : lang) {
        const char *line_ptr = line.c_str();
        query_vectors.emplace_back(encode_query(im, line_ptr));
    }

    return hdc::add(query_vectors);
}

template<typename VectorType>
std::size_t test_language(
        const hdc::ItemMemory<VectorType> &im,
        const hdc::AssociativeMemory<VectorType> &am,
        const lang_t &lang,
        const std::size_t right_answer
        ) {
    std::size_t correct = 0;

    for (auto &sentence : lang) {
        const char *str = sentence.c_str();
        auto query = encode_query(im, str);
        std::size_t prediction = am.search(query);
        correct += (prediction == right_answer) ? 1 : 0;
    }

    return correct;
}

template<typename VectorType>
int language(const argparse::ArgumentParser& args) {
    std::size_t retrain = args.get<size_t>("--retrain");
    hdc::dim_t dim = args.get<hdc::dim_t>("--dim");

    //std::cout << "language: retrain: " << retrain <<
    //    " D: " << dim << std::endl;
    std::cout << " D: " << dim << std::endl;

    const auto &dataset = read_dataset(args.get("train_dir"));
    const auto &testset = read_dataset(args.get("test_dir"));

    auto im = hdc::ItemMemory<VectorType>(27, dim);

    std::vector<VectorType> trained_languages;
    for (auto &lang : dataset) {
        trained_languages.emplace_back(train_language(im, lang));
    }

    hdc::AssociativeMemory<VectorType> am(trained_languages);
    std::vector<std::size_t> correct;
    for (std::size_t i = 0; i < testset.size(); i++) {
        const auto &lang = testset[i];
        correct.emplace_back(test_language(im, am, lang, i));
    }

    // Print results summary
    for (std::size_t i = 0; i < languages.size(); i++) {
        std::cout << languages[i] << ": " << correct[i] << "\t"
            << (float)correct[i]/1000*100 << "%" << std::endl;
    }

    return 0;
}

auto add_args(argparse::ArgumentParser& program) {
    program.add_argument("train_dir")
        .help("Path to the train data directory.");
    program.add_argument("test_dir")
        .help("Path to the test data directory.");

    // Optional arguments
    common_args::add_args(program);
    //program.add_argument("--load-model")
    //    .help("Load model from path and only execute the test stage. The path "
    //          "given must be of a directory containing the data for the IM, "
    //          "CIM, and AM.");
    //program.add_argument("--save-model")
    //    .help("Write all used memories to the given path. The files are "
    //          "created according to the name of the memory. ItemMemory is "
    //          "saved as im.txt, ContinuousItemMemory as cim.txt, and "
    //          "AssociativeMemory as am.txt");

    return program;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser args("Language");

    try {
        add_args(args);
        args.parse_args(argc, argv);
    } catch (const std::runtime_error& e) {
        std::cout << args << std::endl;
        std::cerr << "Failed to parse arguments! " << e.what() << std::endl;
        return -1;
    }

    auto hdc = args.get("hdc");

    if (hdc == "bin") {
        std::cout << "language binary" << std::endl;
        return language<hdc::bin_t>(args);
    } else if (hdc == "int") {
        std::cout << "language int" << std::endl;
        return language<hdc::int32_t>(args);
    } else if (hdc == "float") {
        std::cout << "language float" << std::endl;
        return language<hdc::float_t>(args);
    }
}

