#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mnist.hpp"
#include "hdc.hpp"

// Each image contains 28x28 (784) pixels
const std::size_t _SIZE_IMG = 784;

typedef std::vector<std::uint8_t> data_t;
typedef std::vector<data_t> dataset_t;
typedef std::vector<std::uint8_t> label_t;

dataset_t read_dataset(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }

    dataset_t dataset;

    std::size_t f_size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer;
    buffer.resize(f_size);
    f.read(buffer.data(), f_size);

    if (!f) {
        std::cerr << "Failed to read file" << std::endl;
        assert(false);
    }

    // Make sure the buffer size is multiple of the image size. Buffer size
    // must be multiple of the number of pixels in the image. Each pixel has 1
    // byte, and each image is 28x28 (784) pixels.
    assert((buffer.size() % _SIZE_IMG) == 0);

    // Convert the pixel values from 0-255 to 0-1
    const std::uint8_t threshold = 255/2;
    for (std::size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = (std::uint8_t)buffer[i] > threshold ? 1 : 0;
    }

    for (std::size_t i = 0; i < buffer.size(); i += _SIZE_IMG) {
        data_t img;
        img.resize(_SIZE_IMG);
        std::memcpy(img.data(), buffer.data()+i, _SIZE_IMG);
        dataset.emplace_back(img);
    }

    return dataset;
}

label_t read_labels(const char* path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }

    std::size_t f_size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer;
    buffer.resize(f_size);
    f.read(buffer.data(), f_size);

    if (!f) {
        std::cerr << "Failed to read file" << std::endl;
        assert(false);
    }

    label_t labels;
    labels.resize(buffer.size());

    std::memcpy(labels.data(), buffer.data(), buffer.size());

    return labels;
}

hdc::HDV encode_query(
        const data_t &pixels,
        const std::vector<hdc::HDV> &idm
        ) {
    std::vector<hdc::HDV> vec;

    for (std::size_t i = 0; i < pixels.size(); i++) {
        hdc::HDV p_vec = idm.at(i);
        // Bitshift black pixels
        if (!pixels[i]) {
            p_vec.p();
        }
        vec.emplace_back(p_vec);
    }

    return hdc::maj(vec);
}

float predict(
        const dataset_t &test_data,
        const label_t &labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        int pred_label = hdc::am_search(
                encode_query(test_data[i], idm),
                am);
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

std::vector<hdc::HDV> train_am(
        int retrain,
        const dataset_t &train_dataset,
        const label_t &train_labels,
        const dataset_t &test_dataset,
        const label_t &test_labels,
        const std::vector<hdc::HDV> &idm
        ) {
    assert(train_labels.size() == train_dataset.size());

    std::vector<hdc::HDV> am;
    std::vector<hdc::HDV> queries;
    int max = *std::max_element(train_labels.begin(), train_labels.end())+1;
    std::vector<std::vector<hdc::HDV>> encoded(max, std::vector<hdc::HDV>());

    // Train
    for (std::size_t i = 0; i < train_labels.size(); i++) {
        queries.emplace_back(encode_query(train_dataset[i], idm));
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

        float test_acc = predict(
                test_dataset,
                test_labels,
                idm,
                am);

        std::cout << "Iteration: " << times <<
            " Accuracy on train dataset: " << train_acc <<
            " Accuracy on test dataset: " << test_acc << std::endl;
    }

    return am;
}

int mnist(int argc, char *argv[]) {
    int retrain = 20;
    hdc::dim_t dim = 10000;

    std::cout << "retrain: " << retrain <<
        " D: " << dim << std::endl;

    dataset_t train_dataset = read_dataset("../dataset/mnist/train_data.bin");
    label_t train_labels = read_labels("../dataset/mnist/train_labels.bin");
    dataset_t test_dataset = read_dataset("../dataset/mnist/test_data.bin");
    label_t test_labels = read_labels("../dataset/mnist/test_labels.bin");

    std::vector<hdc::HDV> idm; // ID memory
    std::vector<hdc::HDV> am; // Associative memory

    // Initialize the ID memory with one ID vector to each pixel in the image
    idm = hdc::init_im(_SIZE_IMG, dim);

    am = train_am(retrain,
            train_dataset,
            train_labels,
            test_dataset,
            test_labels,
            idm);

    float accuracy = predict(test_dataset, test_labels, idm, am);
    std::cout << "Final accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
