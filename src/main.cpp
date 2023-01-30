#include "emg.hpp"
#include "language.hpp"
#include "mnist.hpp"

int main(int argc, char *argv[]) {
#ifdef EMG
    emg(argc, argv);
#endif
#ifdef LANGUAGE
    language(argc, argv);
#endif
#ifdef MNIST
    mnist(argc, argv);
#endif
#ifdef VOICEHD
    voicehd(argc, argv);
#endif
}
