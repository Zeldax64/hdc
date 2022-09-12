# Language Recognition Dataset

The datasets used to train and test language recognition are the same as those used in Rahimi et al. \[1\]. The dataset comprises 21 European languages, which are available as train and test data. The train data is taken from the Wortschatz Corpora, and the test data is taken from the Europarl Parallel Corpus. However, this repository's dataset contains some quality of life changes compared to the original Rahimi's repository available on Github.

1. Unused language data in the train folder were removed.
2. Removed pattern such as empty lines ("\n\n") and lines that contain one or more space characters (e.g., "\n \n").
3. Modified test folder file structure to be similar to the train folder. This eases the implementation of I/O routines to load the datasets.

\[1\] A. Rahimi, P. Kanerva, and J. M. Rabaey "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing," In ACM/IEEE International Symposium on Low-Power Electronics and Design (ISLPED), 2016.

