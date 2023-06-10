# Simple Characters classification

This project demonstrates simple solution to digits and characters classification. More about training process in files train.py or train.ipynb

## Data

The data used in this project is balanced EMNIST dataset. The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable. It contains 47 classes, has 112,800 size of training sample and 18,800 (15%) of testing sample. For the task of VIN codes characters classification it was removed such letters as Q(q) I(i) O(o) according to VIN codes [standart](https://uk.wikipedia.org/wiki/%D0%86%D0%B4%D0%B5%D0%BD%D1%82%D0%B8%D1%84%D1%96%D0%BA%D0%B0%D1%86%D1%96%D0%B9%D0%BD%D0%B8%D0%B9_%D0%BD%D0%BE%D0%BC%D0%B5%D1%80_%D1%82%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D1%80%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D0%B7%D0%B0%D1%81%D0%BE%D0%B1%D1%83).

## Methods

The CNN architecture was used. The model contains 3 convolutional layers with number of filters 32 64 128 correspondigly and 3 dense layers with 256 128 47(for each class). To prevent overfitting it was added dropout before the first dense layer.

## Ideas

For the model to be more robust data augmentation technique was used.

## Results

In the final result the accuracy 90.62% - training and 91.51% - validation was achieved. The reason behind higher accuracy on validation set could be in usage droupout and data augmentation techniques that were used for training process.

## Usage Example

### With Docker

```console
$ docker run -it --rm chi_vin_codes ls
Dockerfile    model      requirements.txt  train.ipynb inference.py  readme.md test_data      train.py
$ docker run -it --rm chi_vin_codes python inference.py --input ./test_data
077, test_data/M_01.jpg
056, test_data/8_01.jpg
049, test_data/1_01.jpg
116, test_data/T_01.jpg
050, test_data/2_01.jpg
072, test_data/H_01.jpeg
098, test_data/b_01.jpeg
050, test_data/2_02.jpg
069, test_data/E_01.jpg
065, test_data/A_02.png
065, test_data/A_01.jpg
078, test_data/N_01.jpg
100, test_data/d_01.jpeg
087, test_data/W_01.jpg
071, test_data/G_01.jpg
```
