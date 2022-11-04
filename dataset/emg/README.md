# EMG Dataset

This dataset is the same version of the EMG dataset used in Rahimi et al. \[1\] and is available on [Github](https://github.com/abbas-rahimi/HDC-EMG). The modification consists of unserializing the Matlab data and saving the dataset as plain binary files. Each subject is a `complete*.bin` file. There are 5 subjects available. Each entry in the dataset contains 4 channels, and each channel value is a double. Thus, a channel is 8 bytes, and an entry is 32 bytes. Thus, the file is a multiple of the entry size. The `labels*.bin` files contain labels from 1 to 5. Each value is stored as 1 byte.

\[1\] A. Rahimi, S. Benatti, P. Kanerva, L. Benini and J. M. Rabaey, "Hyperdimensional biosignal processing: A case study for EMG-based hand gesture recognition," 2016 IEEE International Conference on Rebooting Computing (ICRC), 2016, pp. 1-8, doi: 10.1109/ICRC.2016.7738683.
