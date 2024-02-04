# Hyperdimensional Computing

This repository implements Hyperdimensional Computing (HDC) models in C++. The models provide solutions to common AI problems but use the HDC framework instead of other usual models such as NNs. The implementations available in this repository are based on papers available in HDC literature.

Models implemented:

- Language recognition (language): Identify the language an unknown string.
- VoiceHD (voicehd): Speech recognition with ISOLET dataset.
- Digit recognition (mnist): Detect a handwritten digit using the MNIST dataset.
- EMG (emg): Hand gesture recognition.

## Running

```
cmake -B build -S .
cmake --build build # -j<nprocs> to build in parallel
```

You can create optimized builds by executing:
```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
```

This generates a binary for each implemented HDC model in the path `build/hdc`. Execute the binary from the repository's root directory using:

```
./build/language dataset/language/train dataset/language/test
./build/voicehd dataset/voicehd/train* dataset/voicehd/test*
./build/mnist dataset/mnist dataset/mnist/train* dataset/mnist/test*
./build/emg dataset/emg
```

