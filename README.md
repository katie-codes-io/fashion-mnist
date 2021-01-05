# Fashion-MNIST Image Classification

#### This repository is a collection of TensorFlow machine learning models for Fashion-MNIST image classification benchmarking.

## Available models

### Convolutional Neural Networks
* [AlexNet](https://dl.acm.org/doi/10.1145/3065386) 
* [LeNet](https://ieeexplore.ieee.org/document/726791)
* [ResNet](https://ieeexplore.ieee.org/document/7780459)

### Recurrent Neural Networks
* [LSTM](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)

## How to run
```python main.py <model> [pretrained_model]```

Example: 

```python main.py LeNet pretrained_LeNet```

Run without arguments to get a list of available models.

## Results

Based on training with 60,000 images and testing with 10,000 images from the Fashion-MNIST dataset (sorted by accuracy).

| Model   | Type | Accuracy | Precision | Recall |  
| ------- | ---- | -------- | --------- | ------ |
| ResNet  | CNN  | 0.9111   | 0.9133    | 0.9096 |
| AlexNet | CNN  | 0.9060   | 0.9104    | 0.9010 |
| LSTM    | RNN  | 0.8959   | 0.9047    | 0.8884 |
| LeNet-5 | CNN  | 0.8882   | 0.8960    | 0.8839 |

