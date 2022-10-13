# VoiceHD Dataset

This dataset is a modified version of the dataset used in Imani et al. \[1\] and is available on [Github](https://github.com/moimani/HD-IDHV). The modification consists of unserializing the python pickled data and saving the dataset in plain text files. Each line in `*_data.txt` contains 617 float numbers to represent the channels, whose values range from -1.0 to 1.0. The `_labels.txt` files contain unsigned numbers (0 to 25) representing the letters in the English alphabet.

\[1\] M. Imani, D. Kong, A. Rahimi, T. Rosing, “VoiceHD: Hyperdimensional Computing for Efficient Speech Recognition", IEEE International Conference on Rebooting Computing (ICRC), 2017.
