# MNIST Dataset

This dataset is the same version of the MNIST dataset used in Hassan et al. \[1\] and is available on [Github](https://github.com/emfhasan/HDCvsCNN_StudyCase). The modification consists of unserializing the Matlab data and saving the dataset as plain binary files. Each image in `*_data.bin` contains 784 bytes, i.e., 1 byte for each pixel and each image is 28x28. Thus, the file is a multiple of the image size. The `*_labels.bin` files contain labels from 0 to 9. Each value is stored as 1 byte.

\[1\] E. Hassan, Y. Halawani, B. Mohammad and H. Saleh, "Hyper-Dimensional Computing Challenges and Opportunities for AI Applications," in IEEE Access, vol. 10, pp. 97651-97664, 2022, doi: 10.1109/ACCESS.2021.3059762.
