#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "voicehd.hpp"
#include "HDV.hpp"

typedef std::vector<float> data_t;
typedef std::vector<data_t> dataset_t;
typedef std::vector<int> label_t;

dataset_t read_dataset(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }

    dataset_t dataset;
    std::string line;
    while (std::getline(f, line)) {
        float i;
        data_t data;
        std::stringstream ss(line);
        while (ss >> i) {
            data.emplace_back(i);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        dataset.emplace_back(data);
    }

    return dataset;
}

label_t read_labels(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }

    label_t labels;
    std::string line;
    while (std::getline(f, line)) {
        labels.emplace_back(std::stoi(line));
    }

    return labels;
}

int get_amplitude_bin(float amp, int levels) {
    // Values defined by the dataset
    const float min = -1.0;
    const float max =  1.0;

    assert(amp >= min);
    assert(amp <= max);

    float range = 2.0;
    float step = range / levels;

    for (int i = 0; i <= levels; i++) {
        float top_threshold = min+(step*(i+1));
        if (amp <= top_threshold) {
            return i;
        }
    }

    assert(false);
}

hdv::HDV encode_query(
        const int levels,
        const data_t &amplitudes,
        const std::vector<hdv::HDV> &idm,
        const std::vector<hdv::HDV> &cim
        ) {
    std::vector<hdv::HDV> vec;

    for (std::size_t i = 0; i < amplitudes.size(); i++) {
        float amp = amplitudes[i];
        int amp_bin = get_amplitude_bin(amp, levels);
        vec.emplace_back(idm.at(i)*cim.at(amp_bin));
    }

    return hdv::maj(vec);
}


std::vector<hdv::HDV> train_am(
        int levels,
        const dataset_t &dataset,
        const label_t &labels,
        const std::vector<hdv::HDV> &idm,
        const std::vector<hdv::HDV> &cim
        ) {
    assert(labels.size() == dataset.size());

    std::vector<hdv::HDV> am;
    int max = *std::max_element(labels.begin(), labels.end())+1;
    std::vector<std::vector<hdv::HDV>> encoded(max, std::vector<hdv::HDV>());

    for (std::size_t i = 0; i < labels.size(); i++) {
        hdv::HDV v = encode_query(levels, dataset[i], idm, cim);
        encoded.at(labels[i]).emplace_back(v);
    }

    for (auto &i : encoded) {
        hdv::HDV acc = hdv::maj(i);
        am.emplace_back(acc);
    }

    return am;
}

float predict(
        int levels,
        const dataset_t &test_data,
        const label_t &labels,
        const std::vector<hdv::HDV> &idm,
        const std::vector<hdv::HDV> &cim,
        const std::vector<hdv::HDV> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        int pred_label = hdv::am_search(
                encode_query(levels, test_data[i], idm, cim),
                am);
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

int voicehd(int argc, char *argv[]) {
    int levels = 10;
    hdv::dim_t dim = 10000;

    std::cout << "levels: " << levels << " D: " << dim << std::endl;

    std::vector<hdv::HDV> idm; // ID memory
    std::vector<hdv::HDV> cim; // Continuous item memory
    std::vector<hdv::HDV> am;  // Associative memory

    dataset_t train_dataset = read_dataset("../dataset/voicehd/train_data.txt");
    label_t train_labels = read_labels("../dataset/voicehd/train_labels.txt");
    dataset_t test_data = read_dataset("../dataset/voicehd/test_data.txt");
    label_t test_labels = read_labels("../dataset/voicehd/test_labels.txt");

    // Initialize the ID memory with 617 entries, one for each frequency bin in
    // the Isolet dataset
    idm = hdv::init_im(617, dim);
    // Initialize the continuous item memory
    cim = hdv::init_cim(levels, idm[0].dim);

    am = train_am(levels, train_dataset, train_labels, idm, cim);

    float accuracy = predict(levels, test_data, test_labels, idm, cim, am);
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
