#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "emg.hpp"
#include "common.hpp"
#include "hdc.hpp"

typedef double data_entry_t;
typedef std::array<data_entry_t, 4> data_t;
typedef std::vector<data_t> dataset_t;
typedef std::uint8_t label_entry_t;
typedef std::vector<label_entry_t> label_t;

// Number of subjects in the dataset
const int _SUBJECTS = 5;

enum encode_t {SPATIAL, TEMPORAL};
static encode_t g_encode;

dataset_t read_dataset(const char* path) {
    bin_buffer_t buffer = read_bin_file(path);
    dataset_t dataset;

    // Each entry in the dataset comprise 4 doubles since the EMG
    // contains 4 channels
    const std::size_t ENTRY_SIZE = sizeof(double) * 4;

    // Make sure the buffer size is multiple of the entry size
    assert((buffer.size() % ENTRY_SIZE) == 0);

    for (std::size_t i = 0; i < buffer.size(); i += ENTRY_SIZE) {
        data_t entry;
        std::memcpy(entry.data(), buffer.data()+i, ENTRY_SIZE);
        dataset.emplace_back(entry);
    }

    return dataset;
}

label_t read_labels(const char* path) {
    bin_buffer_t buffer = read_bin_file(path);
    label_t labels;

    labels.resize(buffer.size());
    std::memcpy(labels.data(), buffer.data(), buffer.size());

    return labels;
}

void downsample(
        std::uint32_t downsamp_rate,
        const dataset_t &d_i,
        const label_t &l_i,
        dataset_t &d_o,
        label_t &l_o) {
    // Check if the input dataset and label are from the same subject
    assert(d_i.size() == l_i.size());

    d_o.clear();
    l_o.clear();

    for (std::size_t i = 0; i < d_i.size(); i += downsamp_rate) {
        d_o.emplace_back(d_i[i]);
        l_o.emplace_back(l_i[i]);
    }
}

void gen_train_data(
        float training_frac,
        dataset_t d_i,
        label_t l_i,
        dataset_t &d_o,
        label_t &l_o) {
    // Get the N first indexes labels from 1 to 7
    auto find_indexes = [](
            std::uint8_t val,
            float training_frac,
            label_t &labels) -> std::vector<std::size_t> {
        std::vector<std::size_t> ret;

        int count = std::count(labels.begin(), labels.end(), val);
        std::size_t train_size = (std::size_t)((float)count * training_frac);

        // Count all entries in vec_labels that are equal to val
        for (std::size_t i = 0; i < labels.size() && ret.size() < train_size; i++) {
            if (labels[i] == val) {
                ret.emplace_back(i);
            }
        }

        return ret;
    };

    auto L1 = find_indexes(1, training_frac, l_i);
    auto L2 = find_indexes(2, training_frac, l_i);
    auto L3 = find_indexes(3, training_frac, l_i);
    auto L4 = find_indexes(4, training_frac, l_i);
    auto L5 = find_indexes(5, training_frac, l_i);
    auto L6 = find_indexes(6, training_frac, l_i);
    auto L7 = find_indexes(7, training_frac, l_i);

    // Create data vector
    d_o.clear();

    auto append_data = [](
            const std::vector<std::size_t> &ind_vec,
            const dataset_t &d_i,
            dataset_t &d_o) {
        for (auto i : ind_vec) {
            d_o.emplace_back(d_i[i]);
        }
    };

    append_data(L1, d_i, d_o);
    append_data(L2, d_i, d_o);
    append_data(L3, d_i, d_o);
    append_data(L4, d_i, d_o);
    append_data(L5, d_i, d_o);
    append_data(L6, d_i, d_o);
    append_data(L7, d_i, d_o);

    // Create labels vector
    l_o.clear();
    auto append_label = [](const std::vector<std::size_t> &ind_vec,
            const label_t &l_i,
            label_t &l_o) {
        for (auto i : ind_vec) {
            l_o.emplace_back(l_i[i]);
        }
    };

    append_label(L1, l_i, l_o);
    append_label(L2, l_i, l_o);
    append_label(L3, l_i, l_o);
    append_label(L4, l_i, l_o);
    append_label(L5, l_i, l_o);
    append_label(L6, l_i, l_o);
    append_label(L7, l_i, l_o);
}

int get_amplitude_bin(float amp, int levels) {
    // Dataset values vary between 0.0 and 20.0
    const float min =  0.0;
    const float max = 20.0;
    const float epsilon = 0.2;

    // Some entries in the dataset are slightly higher than the 20.0 value
    // specified in the paper. Adjust their values to 20.0
    amp = amp > max ? max : amp;

    float range = max-min;
    float step = range / levels;

    for (int i = 0; i < levels; i++) {
        float top_threshold = min+(step*(i+1));
        if (amp <= top_threshold) {
            return i;
        }
    }

    std::cerr << "Unreachable condition in get_amplitude_bin(). Value: " <<
        amp << std::endl;
    assert(false);
    return -1;
}

hdc::HDV encode_query(
        int levels,
        int N_grams,
        std::size_t entry,
        const dataset_t &dataset,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim
        ) {
    std::vector<hdc::HDV> spatial;
    std::vector<hdc::HDV> temporal;

    for (int i = 0; i < N_grams; i++) {
        const data_t &channels = dataset.at(entry+i);

        for (std::size_t c = 0; c < channels.size(); c++) {
            auto &amp = channels[c];
            int amp_bin = get_amplitude_bin(amp, levels);
            spatial.emplace_back(idm.at(c)*cim.at(amp_bin));
        }

        if (g_encode == TEMPORAL) {
            hdc::HDV t = hdc::maj(spatial);
            t.p(i);
            temporal.emplace_back(t);
        }
    }

    if (g_encode == SPATIAL) {
        return hdc::maj(spatial);
    }
    else {
        return hdc::mul(temporal);
    }
}

float predict(
        int levels,
        int N_grams,
        const dataset_t &test_data,
        const label_t &labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim,
        const std::vector<hdc::HDV> &am) {
    assert(labels.size() == test_data.size());

    std::size_t correct = 0;

    for (std::size_t i = 0; i+N_grams-1 < test_data.size(); i++) {
        int pred_label = hdc::am_search(
                encode_query(levels, N_grams, i, test_data, idm, cim),
                am);
        // Adjust the predicted label value since the labels dataset use values
        // between 1 <-> 5
        pred_label++;
        if (pred_label == labels[i]) {
            correct++;
        }
    }

    return (float)correct/(float)test_data.size()*100.;
}

std::vector<hdc::HDV> train_am(
        int levels,
        int N,
        const dataset_t &train_dataset,
        const label_t &train_labels,
        const dataset_t &test_dataset,
        const label_t &test_labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim
        ) {
    std::vector<hdc::HDV> am;
    std::vector<hdc::HDV> encoded;

    int min = *std::min_element(train_labels.begin(), train_labels.end());
    label_entry_t label = min;

    for (std::size_t i = 0; i+N-1 < train_labels.size(); i++) {
        if (label != train_labels[i]) {
            am.emplace_back(hdc::maj(encoded));
            encoded.clear();
            label = train_labels[i];
        }

        if (train_labels[i] == train_labels[i+N-1]) {
            hdc::HDV enc = encode_query(levels, N, i, train_dataset, idm, cim);
            encoded.emplace_back(enc);
        }
    }

    // Append last value
    am.emplace_back(hdc::maj(encoded));

    return am;
}

int predict_window_max(
        int levels,
        int N_grams,
        std::size_t start,
        std::size_t stop,
        const dataset_t &dataset,
        const label_t &labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim,
        const std::vector<hdc::HDV> &am) {
    std::vector<hdc::HDV> encoded;

    // TODO: dado um começo e um final, predizer quem está correto dentro da
    // window
    for (int i = start; i < stop; i++) {
        hdc::HDV enc = encode_query(levels, N_grams, i, dataset, idm, cim);
        encoded.emplace_back(enc);
    }

    // Search for the vector with highest similarity
    int index = 0;
    hdc::dim_t min_dist = encoded.at(0).dim;
    for (const auto &v : encoded) {
        for (int i = 0; i < am.size(); i++) {
            hdc::dim_t dist = hdc::dist(v, am[i]);

            if (dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }
    }

    return index;
}

float test_slicing(
        int levels,
        int N_grams,
        const dataset_t &dataset,
        const label_t &labels,
        const std::vector<hdc::HDV> &idm,
        const std::vector<hdc::HDV> &cim,
        const std::vector<hdc::HDV> &am) {
    // This function is a simplified version of the same function
    // available in Imani's matlab script since it does not consider
    // overlapping windows

    // Find the min label value used
    int min = *std::min_element(labels.begin(), labels.end());
    label_entry_t label = min;

    int predictions = 0;
    int correct = 0;
    int start = labels.size();
    int stop;
    int window;
    for (std::size_t i = 0; i+N_grams-1 < labels.size(); i++) {
        if ((labels[i] == labels[i+1]) && (start > i)) {
            start = i;
        }
        else if ((labels[i] != labels[i+1]) && (start <= i)) {
            stop = i;
            window = stop - start;
            window = std::max(window, N_grams);

            int pred_label = predict_window_max(
                    levels,
                    N_grams,
                    start,
                    start+window,
                    dataset,
                    labels,
                    idm,
                    cim,
                    am);

            // Adjust the 0-indexed pred_label to compare it with the
            // label value
            pred_label += min;

            predictions++;
            if (pred_label == labels[start]) {
                correct++;
            }
            start = labels.size();
        }
        else {
            // Unreachable condition
            assert(false);
        }
    }

    return (float)correct/(float)predictions * 100.;
}

// Main //
int emg(int argc, char *argv[]) {
    const int CHANNELS = 4;

    hdc::dim_t dim = 10000;
    int levels = 21;
    g_encode = SPATIAL;
    int N_grams = 4;
    float training_frac = 0.25;
    std::uint32_t downsample_rate = 1;
    std::string enc_str;

    //if (g_encode == SPATIAL) {
    //    enc_str = "SPATIAL";
    //    N_grams = 1;
    //}
    //else {
    //    enc_str = "TEMPORAL";
    //}

    //std::cerr << "D: " << dim <<
    //    " Levels: " << levels <<
    //    " Encode type: " << enc_str <<
    //    " N-grams: " << N_grams <<
    //    " Training Fraction: " << training_frac * 100.0 << "%" <<
    //    " Downsample: " << downsample_rate << std::endl;

    std::vector<hdc::HDV> idm; // ID memory
    std::vector<hdc::HDV> cim; // Continuous item memory
    std::vector<hdc::HDV> am;  // Associative memory

    // Initialize the ID memory with 4 entries, one for each channel used in
    // the dataset
    idm = hdc::init_im(CHANNELS, dim);
    // Initialize the continuous item memory
    cim = hdc::init_cim(levels, dim);

    // Dataset variables
    std::vector<dataset_t> complete;
    std::vector<label_t> labels;
    std::array<dataset_t, _SUBJECTS> ts_complete;
    std::array<label_t, _SUBJECTS> ts_labels;
    std::array<dataset_t, _SUBJECTS> train_complete;
    std::array<label_t, _SUBJECTS> train_labels;

    //-- Experiments --//

    // Spatial encoding experiment
    // Train and predict spatial encoding
    std::cout << "Spatial encoding" << std::endl;
    if (g_encode == SPATIAL) {
        enc_str = "SPATIAL";
        N_grams = 1;
    }
    else {
        enc_str = "TEMPORAL";
    }

    std::cout << "D: " << dim <<
        " Levels: " << levels <<
        " Encode type: " << enc_str <<
        " N-grams: " << N_grams <<
        " Training Fraction: " << training_frac * 100.0 << "%" <<
        " Downsample: " << downsample_rate << std::endl;

    for (int i = 1; i <= _SUBJECTS; i++) {
        // Read datasets
        std::string num_str = std::to_string(i);
        std::string d_path = "../dataset/emg/complete"+num_str+".bin";
        std::string l_path = "../dataset/emg/labels"+num_str+".bin";
        complete.emplace_back(read_dataset(d_path.c_str()));
        labels.emplace_back(read_labels(l_path.c_str()));
    }

    // Generate testsets. A testset is a downsampled version of the dataset.
    for (int i = 0; i < complete.size(); i++) {
        downsample(
                downsample_rate,
                complete[i],
                labels[i],
                ts_complete[i],
                ts_labels[i]);

        // Generate train data. The train data is only a fraction
        // (training_frac) of the test set.
        gen_train_data(
                training_frac,
                ts_complete[i],
                ts_labels[i],
                train_complete[i],
                train_labels[i]);

        float accuracy;

        am = train_am(
                levels,
                N_grams,
                train_complete[i],
                train_labels[i],
                ts_complete[i],
                ts_labels[i],
                idm,
                cim);

        accuracy = predict(
                levels,
                N_grams,
                ts_complete[i],
                ts_labels[i],
                idm,
                cim,
                am);

        std::cout << "Accuracy[" << std::to_string(i) << "]: "
            << accuracy << "%" << std::endl;
    }

    // Temporal encoding experiment
    downsample_rate = 250;
    N_grams = 4;
    g_encode = TEMPORAL;

    std::cout << "Temporal encoding" << std::endl;
    if (g_encode == SPATIAL) {
        enc_str = "SPATIAL";
        N_grams = 1;
    }
    else {
        enc_str = "TEMPORAL";
    }
    std::cout << "D: " << dim <<
        " Levels: " << levels <<
        " Encode type: " << enc_str <<
        " N-grams: " << N_grams <<
        " Training Fraction: " << training_frac * 100.0 << "%" <<
        " Downsample: " << downsample_rate << std::endl;
    for (int i = 0; i < _SUBJECTS; i++) {
        if (i == _SUBJECTS-1) {
            downsample_rate = 50;
        }
        // Generate new test and train sets
        downsample(
                downsample_rate,
                complete[i],
                labels[i],
                ts_complete[i],
                ts_labels[i]);

        gen_train_data(
                training_frac,
                ts_complete[i],
                ts_labels[i],
                train_complete[i],
                train_labels[i]);

        float accuracy;

        am = train_am(
                levels,
                N_grams,
                train_complete[i],
                train_labels[i],
                ts_complete[i],
                ts_labels[i],
                idm,
                cim);

        accuracy = test_slicing(
                levels,
                N_grams,
                ts_complete[i],
                ts_labels[i],
                idm,
                cim,
                am);

        std::cout << "Accuracy[" << std::to_string(i) << "]: "
            << accuracy << "%" << std::endl;
    }

    return 0;
}
