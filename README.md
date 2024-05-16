# FeatureCloud CNN Quantization
### Image Classification with Quantization

This application facilitates image classification using deep neural network models with the added capability of model quantization. It employs both post-static quantization and quantization-aware training (QAT) techniques from [PyTorch](https://pytorch.org/docs/stable/quantization.html) to compress the models, thereby reducing memory and computational requirements while maintaining performance.

## How Post-Static Quantization and QAT are Utilized

### Post-Static Quantization: 
In this approach, the model is trained using traditional methods and then quantized after training. This involves preparing the model for quantization, calibrating it with the training data, and converting it to a quantized form.
### Quantization-Aware Training (QAT):
QAT is a technique where the model is trained with quantization-aware optimization. This allows the model to learn and adapt to the quantization process during training, resulting in improved performance post-quantization.

Image classification is a fundamental task in computer vision, and this app caters to datasets like CIFAR and MNIST. CIFAR-10 and MNIST are widely used benchmark datasets for image classification tasks. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, while MNIST comprises 28x28 grayscale images of handwritten digits.




## Config Settings
### Training Settings
```python
model: mnist_model.py
train_dataset: "train_dataset.pth"
test_dataset: "test_dataset.pth"

batch_size: 256
learning_rate: 0.001
epochs: 1
max_iter: 10
```
### Training Options
#### Model
File name will provided as generic data to clients, which later will be imported by the app. The model class should have the name 'Model' and include the forward method. For more details, please refer to the example provided in [models/pytorch/models](/data/sample_data/generic/cnn.py) 

`model`: This field should specify the Python file containing the model implementation. It is expected to be in the .py format.

#### Local dataset

`train_dataset` :  Path to the training dataset.

`test_dataset:` :  Path to the training dataset.

These datasets will be loaded using the `torch.utils.data.DataLoader` class.


#### Training config
`batch_size`: Specifies the number of samples in each training batch.

`learning_rate`: Determines the rate at which the model's parameters are updated during training.

`epochs`: Number of epochs for training.

`max_iter` : Defines the maximum number of communication rounds.

This training configurations will also be used in the training stage of Quantization Aware Training but easily be replaced by custom training functions.




### Quantization Settings 
```python

backend: 'qnnpack'  or   'fbgemm'
quant_type: 'post_static'  or  'qat'

```

#### Quantization Parameters config

`backend`: This parameter specifies the backend engine used for quantization. It can take the values 'qnnpack' or 'fbgemm'. 'qnnpack' is optimized for ARM CPUs, while 'fbgemm' is optimized for x86 CPUs.

`quant_type`: This parameter determines the type of quantization to be applied. It can be either 'post_static' or 'qat'. 'post_static' refers to post-training static quantization, where weights are quantized after training. 'qat' stands for quantization-aware training, where the model is trained with quantization in mind, allowing for more accurate quantization during inference.



### Limitations

The quantization methods implemented in this application are designed specifically for Convolutional Neural Networks (CNNs) and support layers such as convolutional, linear, dropout and pooling layers. 

This app provides a training function which can be adjusted. If you want to use your own training function for QAT, please use:

```python
out = forward_quant(mnist_model, data)
```
as forward pass. This applies quantization and dequantization to the input before and after passing it through a given model and thus it is trained with the awareness of the quantization process.

### Run app

#### Prerequisite

To run the quantization app, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud
```

Then either download the quantization app image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/fc-quantization-app
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/fc-quantization-app
```

