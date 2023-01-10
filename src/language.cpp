#include <array>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hdc.hpp"

typedef std::vector<std::string> lang_t;
typedef std::vector<lang_t> dataset_t;

// Constants and helper variables
static const std::string train_dir = "../dataset/language/train/";
static const std::string test_dir = "../dataset/language/test/";

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

class ExMajReduction : public std::exception {
    private:
        std::string _msg;
        std::size_t _size;
    public:
    ExMajReduction(std::size_t size) : _size(size) {
        this->_msg = "Failed to bundle reduce a collection of hyper vectors. "
            "Vector size: " + std::to_string(_size);
    }

    virtual const char* what() const throw() {
        return this->_msg.c_str();
    }
};

static hdc::HDV _maj_reduction(const std::vector<hdc::HDV> &v) {
    if (v.size() < 3) { // 3 is the minimal number to compute the maj()
        throw ExMajReduction(v.size());
    }
    return hdc::maj(v);
}

static std::size_t predict(const hdc::HDV &query,
        const std::vector<hdc::HDV> &am) {
    std::size_t answer = 0;
    hdc::dim_t dist = query.dim;
    hdc::dim_t min_dist = query.dim;

    for (std::size_t i = 0; i < am.size(); i++) {
        const auto &v = am[i];
        dist = hdc::dist(query, v);
        if (dist < min_dist) {
            answer = i;
            min_dist = dist;
        }
    }

    return answer;
}

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
        dataset.emplace_back(read_lang_file(dataset_path+lang+".txt"));
    }

    return dataset;
}

hdc::HDV encode_3gram(const std::vector<hdc::HDV> &im, const char *str) {
    // ASCII table offset to the first lower char, i.e., the letter "a". The IM
    // is 27-entry memory packed as 26 letters and the space (' ') entry. The
    // input text in the *str variable must contain only lower-case letters
    // without punctuation.
    const int FIRST_lOWER_CHAR = 97;
    std::vector<hdc::HDV> items;
    for (int i = 0; i < 3; i++, str++) {
        if (*str != ' ') {
            // Assign the lower-case letter to its IM correspondent
            items.emplace_back(im[*str-FIRST_lOWER_CHAR]);
        }
        else {
            // Assign the space char as the last item in the IM
            items.emplace_back(im.back());
        }
    }
    // Permute the items
    items[0].p(); items[0].p();
    items[1].p();

    // Return the trigram HV
    return items[0]*items[1]*items[2];
}

hdc::HDV encode_query(const std::vector<hdc::HDV> &im, const char *str) {
    std::vector<hdc::HDV> n_grams;

    // Encode n-grams. Since we create 3-grams, loop until the pointer of str+3
    // is different from the null character.
    while (*(str+3)) {
        n_grams.emplace_back(encode_3gram(im, str));
        str++;
    }

    return _maj_reduction(n_grams);
}

hdc::HDV train_language(const std::vector<hdc::HDV> &im, const lang_t &lang) {
    std::vector<hdc::HDV> query_vectors;

    for (auto &line : lang) {
        //std::cout << "Encoding line: " << line << std::endl;
        const char *line_ptr = line.c_str();
        try {
            query_vectors.emplace_back(encode_query(im, line_ptr));
        } catch (ExMajReduction &e) {
            std::cout << e.what() << std::endl;
            std::cout << "Line:" << line << std::endl;
            std::cout << "lang[0]:" << lang[0] << std::endl;
            throw e;
        }
    }

    return _maj_reduction(query_vectors);
}

std::size_t test_language(const std::vector<hdc::HDV> &im,
        const std::vector<hdc::HDV> &am,
        const lang_t &lang,
        const std::size_t right_answer) {
    std::size_t correct = 0;

    for (auto &sentence : lang) {
        const char *str = sentence.c_str();
        const hdc::HDV &query = encode_query(im, str);
        std::size_t prediction = predict(query, am);
        correct += prediction == right_answer ? 1 : 0;
    }

    return correct;
}

int language(int argc, char *argv[]) {
    const hdc::dim_t dim = 1000;
    std::vector<hdc::HDV> im;
    std::vector<hdc::HDV> am;

    const auto &dataset = read_dataset(train_dir);
    const auto &testset = read_dataset(test_dir);

    /* Train */
    // Initialize Item Memory with 27 vector, one for each alphabet letter +
    // space
    for (int i = 0; i < 27; i++) {
        im.emplace_back(hdc::HDV(dim));
    }

    for (auto &lang : dataset) {
        am.emplace_back(train_language(im, lang));
    }

    std::vector<std::size_t> correct;
    for (std::size_t i = 0; i < testset.size(); i++) {
        const auto &lang = testset[i];
        correct.emplace_back(test_language(im, am, lang, i));
    }

    for (std::size_t i = 0; i < languages.size(); i++) {
        std::cout << languages[i] << ": " << correct[i] << "\t"
            << (float)correct[i]/1000*100 << "%" << std::endl;
    }

    return 0;
}

