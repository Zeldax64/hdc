#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "voicehd.hpp"
#include "hdc.hpp"

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

hdc::HDV encode_query(
        const int levels,
        const data_t &amplitudes,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim
        ) {
    std::vector<hdc::HDV> vec;

    for (std::size_t i = 0; i < amplitudes.size(); i++) {
        float amp = amplitudes[i];
        int amp_bin = get_amplitude_bin(amp, levels);
        vec.emplace_back(idm.at(i)*cim.at(amp_bin));
    }

    return hdc::maj(vec);
}

float predict(
        int levels,
        const dataset_t &test_data,
        const label_t &labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim,
        const std::vector<hdc::HDV> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        int pred_label = hdc::am_search(
                encode_query(levels, test_data[i], idm, cim),
                am);
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

std::vector<hdc::HDV> train_am(
        int retrain,
        int levels,
        const dataset_t &train_dataset,
        const label_t &train_labels,
        const dataset_t &test_dataset,
        const label_t &test_labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim
        ) {
    assert(train_labels.size() == train_dataset.size());

    std::vector<hdc::HDV> am;
    std::vector<hdc::HDV> queries;
    int max = *std::max_element(train_labels.begin(), train_labels.end())+1;
    std::vector<std::vector<hdc::HDV>> encoded(max, std::vector<hdc::HDV>());

    // Train
    for (std::size_t i = 0; i < train_labels.size(); i++) {
        queries.emplace_back(encode_query(levels, train_dataset[i], idm, cim));
        encoded.at(train_labels[i]).emplace_back(queries[i]);
    }

    for (auto &i : encoded) {
        hdc::HDV acc = hdc::maj(i);
        am.emplace_back(acc);
    }

    // Retraining
    for (int times = 0; times < retrain; times++) {
        float train_acc = -1.0;
        std::size_t correct = 0;

        for (std::size_t i = 0; i < train_dataset.size(); i++) {
            const hdc::HDV &query = queries[i];
            int pred_label = hdc::am_search(query, am);
            if (pred_label != train_labels[i]) {
                encoded[train_labels[i]].emplace_back(query);
                encoded[pred_label].emplace_back(hdc::invert(query));
            }
            else {
                correct++;
            }
        }

        train_acc = (float)correct/(float)train_dataset.size() * 100.;

        am.clear();
        for (auto &i : encoded) {
            hdc::HDV acc = hdc::maj(i);
            am.emplace_back(acc);
        }

        float test_acc = predict(levels,
                test_dataset,
                test_labels,
                idm,
                cim,
                am);

        std::cout << "Iteration: " << times <<
            " Accuracy on train dataset: " << train_acc <<
            " Accuracy on test dataset: " << test_acc << std::endl;

    }

    return am;
}

int voicehd(int argc, char *argv[]) {
    int retrain = 20;
    int levels = 10;
    hdc::dim_t dim = 10000;

    std::cout << "retrain: " << retrain <<
        " levels: " << levels <<
        " D: " << dim << std::endl;

    std::vector<hdc::HDV> idm; // ID memory
    std::vector<hdc::HDV> cim; // Continuous item memory
    std::vector<hdc::HDV> am;  // Associative memory

    dataset_t train_dataset = read_dataset("../dataset/voicehd/train_data.txt");
    label_t train_labels = read_labels("../dataset/voicehd/train_labels.txt");
    dataset_t test_dataset = read_dataset("../dataset/voicehd/test_data.txt");
    label_t test_labels = read_labels("../dataset/voicehd/test_labels.txt");

    // Initialize the ID memory with 617 entries, one for each frequency bin in
    // the Isolet dataset
    idm = hdc::init_im(617, dim);
    // Initialize the continuous item memory
    cim = hdc::init_cim(levels, idm[0].dim);

    am = train_am(retrain,
            levels,
            train_dataset,
            train_labels,
            test_dataset,
            test_labels,
            idm,
            cim);

    float accuracy = predict(levels, test_dataset, test_labels, idm, cim, am);
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
