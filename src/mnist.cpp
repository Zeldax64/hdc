#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <argparse/argparse.hpp>

#include "AssociativeMemory.hpp"
#include "ItemMemory.hpp"
#include "hdc.hpp"
#include "common_args.hpp"

// Each image contains 28x28 (784) pixels
const std::size_t _SIZE_IMG = 784;

typedef std::vector<std::uint8_t> data_t;
typedef std::vector<data_t> dataset_t;
typedef std::vector<std::uint8_t> label_t;

dataset_t read_dataset(const std::string& path) {
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

label_t read_labels(const std::string& path) {
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

template<typename VectorType>
VectorType encode_query(
        const data_t &pixels,
        const hdc::ItemMemory<VectorType> &idm
        ) {
    std::vector<VectorType> vec;

    for (std::size_t i = 0; i < pixels.size(); i++) {
        auto p_vec = idm.at(i);
        // Bitshift black pixels
        if (!pixels[i]) {
            p_vec = hdc::p(p_vec);
        }
        vec.emplace_back(p_vec);
    }

    return hdc::add(vec);
}

template<typename VectorType>
float predict(
        const dataset_t &test_data,
        const label_t &labels,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::AssociativeMemory<VectorType> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        int pred_label = am.search(
                encode_query(test_data[i], idm)
                );
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

template<typename VectorType>
hdc::AssociativeMemory<VectorType> train_am(
        int retrain,
        const dataset_t &train_dataset,
        const label_t &train_labels,
        const dataset_t &test_dataset,
        const label_t &test_labels,
        const hdc::ItemMemory<VectorType> &idm
        ) {
    if (train_labels.size() != train_dataset.size()) {
        throw std::runtime_error("Attempt to train AM using incompatible train and label datasets.");
    }

    std::vector<VectorType> encoded_train; // Container of encoded vectors from the train dataset
    // Container of vectors belonging to a label (or class)
    int max = *std::max_element(train_labels.begin(), train_labels.end())+1;
    std::vector<std::vector<VectorType>> class_vectors(max, std::vector<VectorType>());

    // Encode the train dataset and store each encoded vector in the
    // class_vectors list
    for (std::size_t i = 0; i < train_labels.size(); i++) {
        encoded_train.emplace_back(encode_query(train_dataset[i], idm)); // Codifica o train dataset
        class_vectors.at(train_labels[i]).emplace_back(encoded_train[i]); // Cria os class vectors
    }

    // Create AM
    auto am = hdc::AssociativeMemory<VectorType>();
    for (auto &i : class_vectors) {
        VectorType acc = hdc::add(i);
        am.emplace_back(acc);
    }

    // Retraining
    for (int times = 0; times < retrain; times++) {
        // Recreate AM only if it is not the first training time
        if (times > 0) {
            am.clear();
            for (auto &i : class_vectors) {
                VectorType acc = hdc::add(i);
                am.emplace_back(acc);
            }
        }

        // Do we have another retraining round? If so, then lets predict with
        // the train dataset and retrain the class vectors
        if (times < retrain) {
            float train_acc = -1.0;
            std::size_t correct = 0;

            // Retrain the class vectors while predicting on the train dataset
            for (std::size_t i = 0; i < train_dataset.size(); i++) {
                const VectorType &query = encoded_train[i];
                int pred_label = am.search(query);
                if (pred_label != train_labels[i]) {
                    class_vectors[train_labels[i]].emplace_back(query);
                    auto inverted_query = query;
                    inverted_query.invert();
                    class_vectors[pred_label].emplace_back(inverted_query);
                }
                else {
                    correct++;
                }
            }

            train_acc = (float)correct/(float)train_dataset.size() * 100.;

            // Test accuracy on the test dataset
            float test_acc = predict(
                    test_dataset,
                    test_labels,
                    idm,
                    am);

            std::cout << "Iteration: " << times <<
                " Accuracy on train dataset: " << train_acc <<
                " Accuracy on test dataset: " << test_acc << std::endl;
        }

    }

    return am;
}

template<typename VectorType>
int mnist(const argparse::ArgumentParser& args) {
    int retrain = args.get<size_t>("--retrain");
    hdc::dim_t dim = args.get<size_t>("--dim");

    std::cout << "retrain: " << retrain <<
        " D: " << dim << std::endl;

    auto train_dataset = read_dataset(args.get("train_data"));
    auto train_labels = read_labels(args.get("train_labels"));
    auto test_dataset = read_dataset(args.get("test_data"));
    auto test_labels = read_labels(args.get("test_labels"));

    hdc::ItemMemory<VectorType> idm(_SIZE_IMG, dim); // ID memory
    //std::vector<hdc::HDV> am; // Associative memory

    auto am = train_am(
            retrain,
            train_dataset,
            train_labels,
            test_dataset,
            test_labels,
            idm);

    float accuracy = predict(test_dataset, test_labels, idm, am);
    std::cout << "Final accuracy: " << accuracy << "%" << std::endl;

    return 0;
}

auto add_args(argparse::ArgumentParser& program) {
    program.add_argument("train_data")
        .help("Path to the train data.");
    program.add_argument("train_labels")
        .help("Path to the train labels.");
    program.add_argument("test_data")
        .help("Path to the test data.");
    program.add_argument("test_labels")
        .help("Path to the test labels.");

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
    argparse::ArgumentParser args("MNIST");

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
        std::cout << "mnist binary" << std::endl;
        return mnist<hdc::bin_t>(args);
    } else if (hdc == "int") {
        std::cout << "mnist int" << std::endl;
        return mnist<hdc::int32_t>(args);
    } else if (hdc == "float") {
        std::cout << "mnist float" << std::endl;
        return mnist<hdc::float_t>(args);
    }
}

