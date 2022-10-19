#include "language.hpp"
#include "mnist.hpp"
#include "voicehd.hpp"

int main(int argc, char *argv[]) {
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
