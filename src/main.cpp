#include "language.hpp"
#include "voicehd.hpp"

int main(int argc, char *argv[]) {
#ifdef LANGUAGE
    language(argc, argv);
#endif
#ifdef VOICEHD
    voicehd(argc, argv);
#endif
}
