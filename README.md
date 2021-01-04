# Fashion-MNIST Image Classification

#### This repository is a collection of TensorFlow machine learning models for Fashion-MNIST image classification benchmarking.

## Available models

### Convolutional Neural Networks
* AlexNet
* LeNet-5

## How to run
```python main.py <model> [pretrained_model]```

Example: 

```python main.py LeNet pretrained_LeNet```

Run without a model argument to get a list of available models.

## Results

Based on training with 60,000 images and testing with 10,000 images from the Fashion-MNIST dataset.

| Model | Accuracy | Precision | Recall |  
| ----- | -------- | --------- | ------ |
| AlexNet | 0.9135 | 0.9162 | 0.9118 |
| LeNet-5 | 0.8910 | 0.9002 | 0.8835 |

