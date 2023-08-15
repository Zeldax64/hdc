#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <argparse/argparse.hpp>

#include "AssociativeMemory.hpp"
#include "ContinuousItemMemory.hpp"
#include "ItemMemory.hpp"
#include "types.hpp"
#include "hdc.hpp"

typedef std::vector<float> data_t;
typedef std::vector<data_t> dataset_t;
typedef std::vector<int> label_t;

dataset_t read_dataset(const std::string& path) {
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

label_t read_labels(const std::string& path) {
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
        const data_t &amplitudes,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::ContinuousItemMemory<VectorType> &cim
        ) {
    int levels = cim.size();
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
        const dataset_t &test_data,
        const label_t &labels,
        const hdc::ItemMemory<VectorType> &idm,
        const hdc::ContinuousItemMemory<VectorType> &cim,
        const hdc::AssociativeMemory<VectorType> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i < test_data.size(); i++) {
        auto query = encode_query(test_data[i], idm, cim);
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
        encoded_train.emplace_back(encode_query(train_dataset[i], idm, cim)); // Codifica o train dataset
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
                    cim,
                    am);

            std::cout << "Iteration: " << times <<
                " Accuracy on train dataset: " << train_acc <<
                " Accuracy on test dataset: " << test_acc << std::endl;
        }

    }

    return am;
}

template <typename VectorType>
void save_model(const argparse::ArgumentParser &args,
                const hdc::ItemMemory<VectorType> &idm,
                const hdc::ContinuousItemMemory<VectorType> &cim,
                const hdc::AssociativeMemory<VectorType> &am) {
    auto path = args.get("--save-model");
    idm.save(path+"/./im.txt");
    cim.save(path+"/./cim.txt");
    am.save(path+"/./am.txt");
}

template <typename VectorType>
std::tuple<
        hdc::ItemMemory<VectorType>,
        hdc::ContinuousItemMemory<VectorType>,
        hdc::AssociativeMemory<VectorType>
    >
load_model(const argparse::ArgumentParser &args) {
    auto path = args.get("--load-model");
    hdc::ItemMemory<VectorType> idm(path+"/./im.txt");
    hdc::ContinuousItemMemory<VectorType> cim(path+"/./cim.txt");
    hdc::AssociativeMemory<VectorType> am(path+"/./am.txt");

    return {idm, cim, am};
}

template<typename VectorType>
int voicehd(const argparse::ArgumentParser& args) {
    std::size_t retrain = args.get<size_t>("--retrain");
    std::size_t levels = args.get<size_t>("--levels");
    hdc::dim_t dim = args.get<hdc::dim_t>("--dim");

    std::cout << "voicehd: retrain: " << retrain <<
        " levels: " << levels <<
        " D: " << dim << std::endl;

    dataset_t train_dataset;
    label_t train_labels;
    if (!args.is_used("--load-model")) {
        train_dataset = read_dataset(args.get("train_data"));
        train_labels = read_labels(args.get("train_labels"));
    }
    auto test_dataset = read_dataset(args.get("test_data"));
    auto test_labels = read_labels(args.get("test_labels"));

    auto idm = hdc::ItemMemory<VectorType>(617, dim);
    auto cim = hdc::ContinuousItemMemory<VectorType>(levels, dim);

    hdc::AssociativeMemory<VectorType> am;
    if (!args.is_used("--load-model")) {
        am = train_am(retrain,
                train_dataset,
                train_labels,
                test_dataset,
                test_labels,
                idm,
                cim);
        if (args.is_used("--save-model")) {
            save_model(args, idm, cim, am);
        }
    }
    else {
        auto ret = load_model<VectorType>(args);
        idm = std::get<0>(ret);
        cim = std::get<1>(ret);
        am  = std::get<2>(ret);
    }

    float accuracy = predict(test_dataset, test_labels, idm, cim, am);
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

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
    program.add_argument("-d", "--dim")
        .help("Number of dimensions.")
        .scan<'d', hdc::dim_t>()
        .default_value<size_t>(1000);
    program.add_argument("-l", "--levels")
        .help("Number of levels.")
        .scan<'d', size_t>()
        .default_value<size_t>(1000);
    program.add_argument("-r", "--retrain")
        .help("Number of retrains.")
        .scan<'d', size_t>()
        .default_value<size_t>(0);

    program.add_argument("--load-model")
        .help("Load model from path and only execute the test stage. The path "
              "given must be of a directory containing the data for the IM, "
              "CIM, and AM.");
    program.add_argument("--save-model")
        .help("Write all used memories to the given path. The files are "
              "created according to the name of the memory. ItemMemory is "
              "saved as im.txt, ContinuousItemMemory as cim.txt, and "
              "AssociativeMemory as am.txt");

    return program;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser args("VoiceHD");

    try {
        add_args(args);
        args.parse_args(argc, argv);
    } catch (const std::runtime_error& e) {
        std::cout << args << std::endl;
        std::cerr << "Failed to parse arguments! " << e.what() << std::endl;
        return -1;
    }

    std::cout << "voicehd binary" << std::endl;
    return voicehd<hdc::bin_t>(args);
}
