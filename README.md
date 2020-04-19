# Multi-Class-Multi-Label-Image-Classification


This repository was prepared as a part of task given by Mr Akash Deep Singh(COO Tessellate Imaging) for multi class multi label image classification.

Datasets were downloaded in google drive and then accesses via google colab.

Initially the labels and images were taken from  "Data_Entry_2017.csv" file

Then the labels were encoded into respective label numbers and by this we found out that there 15 labels.

Labels which were specified some disease should also be classified as "Abnormal"

Labels which were specified with "No Finding" should be classified as "Normal".

In Dataset.py, the train_dataset returns a dictionary of image and labels of class_type(Normal/Abnormal) and Disease(15 Labels)

2 pretrained models were choosen i.e. alexnet and mobilenet.

This repository is a Pure PyTorch Code...
