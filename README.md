# Fashion-MNIST Image Classification

#### This repository is a collection of TensorFlow machine learning models for Fashion-MNIST image classification benchmarking.

## Available models

### Convolutional Neural Networks
* [AlexNet](https://dl.acm.org/doi/10.1145/3065386) 
* [LeNet](https://ieeexplore.ieee.org/document/726791)
* [ResNet](https://ieeexplore.ieee.org/document/7780459)

## How to run
```python main.py <model> [pretrained_model]```

Example: 

```python main.py LeNet pretrained_LeNet```

Run without arguments to get a list of available models.

## Results

Based on training with 60,000 images and testing with 10,000 images from the Fashion-MNIST dataset.

| Model | Accuracy | Precision | Recall |  
| ----- | -------- | --------- | ------ |
| AlexNet | 0.9060 | 0.9104 | 0.9010 |
| LeNet-5 | 0.8926 | 0.8991 | 0.8880 |
| ResNet  | 0.9111 | 0.9133 | 0.9096 |git 

