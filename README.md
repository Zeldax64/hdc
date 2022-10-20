# Hyperdimensional Computing

This repository implements Hyperdimensional Computing (HDC) models in C++. The models provide solutions to common AI problems but use the HDC framework instead of other usual models.

Models implemented:

- Language recognition (language): Detect the text of an unknown string. 
- VoiceHD (voicehd): Speech recognition.
- Digit recognition (mnist): Detect a handwritten digit using the MNIST dataset.

## Running

```
mkdir build
cd build
cmake ..
make
```

This generates a binary for each implemented HDC model. Execute the binary from the build directory using:

```
./language
```

