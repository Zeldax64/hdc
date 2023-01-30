#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "AssociativeMemory.hpp"
#include "ContinuousItemMemory.hpp"
#include "ItemMemory.hpp"
#include "types.hpp"
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

int get_amplitude_bin(float amp, std::size_t levels) {
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

template<typename VectorType>
VectorType encode_query(
        const int levels,
        const data_t &amplitudes,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::ContinuousItemMemory<VectorType> &cim
        ) {
    std::vector<VectorType> vec;

    for (std::size_t i = 0; i < amplitudes.size(); i++) {
        float amp = amplitudes[i];
        int amp_bin = get_amplitude_bin(amp, levels);
        auto temp = hdc::mul(idm.at(i), cim.at(amp_bin));
        vec.emplace_back(temp);
    }

    return hdc::add(vec);
}

template<typename VectorType>
float predict(
        int levels,
        const dataset_t &test_data,
        const label_t &labels,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::ContinuousItemMemory<VectorType> &cim,
        const hdc::AssociativeMemory<VectorType> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        auto query = encode_query(levels, test_data[i], idm, cim);
        int pred_label = am.search(query);
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

template<typename VectorType>
hdc::AssociativeMemory<VectorType> train_am(
        std::size_t retrain,
        std::size_t levels,
        const dataset_t &train_dataset,
        const label_t &train_labels,
        const dataset_t &test_dataset,
        const label_t &test_labels,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::ContinuousItemMemory<VectorType> &cim
        ) {
    assert(train_labels.size() == train_dataset.size());

    std::vector<VectorType> encoded_train; // Container of encoded vectors from the train dataset
    // Container of vectors belonging to a label (or class)
    int max = *std::max_element(train_labels.begin(), train_labels.end())+1;
    std::vector<std::vector<VectorType>> class_vectors(max, std::vector<VectorType>());

    // Encode the train dataset and store each encoded vector in the
    // class_vectors list
    for (std::size_t i = 0; i < train_labels.size(); i++) {
        encoded_train.emplace_back(encode_query(levels, train_dataset[i], idm, cim)); // Codifica o train dataset
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

    }

    return am;
}

static bool _has_option(const std::vector<std::string>& args, const std::string& opt) {
    return std::find(args.begin(), args.end(), opt) != args.end();
}

static bool _parse_uint(const std::vector<std::string>& args, const std::string& opt, std::size_t& ret) {
    auto it = std::find(args.begin(), args.end(), opt);
    it++;
    if (it != args.end()) {
        ret = std::stoul(*it, 0, 10);
        return true;
    }

    return false;
}

static void _parse_dim(const std::vector<std::string>& args, hdc::dim_t& ret) {
    if (_has_option(args, "-d")) {
        _parse_uint(args, "-d", ret);
    }
}

static void _parse_retrain(const std::vector<std::string>& args, std::size_t& ret) {
    if (_has_option(args, "-r")) {
        _parse_uint(args, "-r", ret);
    }
}
static void _parse_levels(const std::vector<std::string>& args, std::size_t& ret) {
    if (_has_option(args, "-L")) {
        _parse_uint(args, "-L", ret);
    }
}

static void _parse_hdc_args(
        const std::vector<std::string>& args,
        hdc::dim_t& dim,
        std::size_t& retrain
        ) {
    _parse_dim(args, dim);
    _parse_retrain(args, retrain);
}

template<typename VectorType>
int voicehd(int argc, char *argv[]) {
    std::size_t retrain = 0;
    std::size_t levels = 10;
    hdc::dim_t dim = 10000;

    std::vector<std::string> args;
    if (argc > 1) {
        args.assign(argv+1, argv+argc);
        _parse_hdc_args(args, dim, retrain);
        _parse_levels(args, levels);
    }

    std::cout << "voicehd: retrain: " << retrain <<
        " levels: " << levels <<
        " D: " << dim << std::endl;

    dataset_t train_dataset = read_dataset("../dataset/voicehd/train_data.txt");
    label_t train_labels = read_labels("../dataset/voicehd/train_labels.txt");
    dataset_t test_dataset = read_dataset("../dataset/voicehd/test_data.txt");
    label_t test_labels = read_labels("../dataset/voicehd/test_labels.txt");

    auto idm = hdc::ItemMemory<VectorType>(617, dim);
    auto cim = hdc::ContinuousItemMemory<VectorType>(levels, dim);

    auto am = train_am(retrain,
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

int main(int argc, char *argv[]) {
    std::cout << "voicehd binary" << std::endl;
    return voicehd<hdc::bin_t>(argc, argv);
}
