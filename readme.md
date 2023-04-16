# CIFAR-10 Image Classification using LeNet
This project aims to classify images from the CIFAR-10 dataset using the LeNet architecture. The LeNet architecture is a convolutional neural network (CNN) that was introduced in 1998 by Yann LeCun et al. for handwritten digit recognition. It consists of two sets of convolutional and average pooling layers, followed by a flattening layer, two fully connected layers, and a softmax classifier.

## Ray Tune
Ray Tune is a Python library that accelerates hyperparameter tuning at any scale. In this project, `learning rate`, `batch size`, and `dropout rate` are hyperparameters that will be tuned in the search space. The search space is specified by a discrete batch size [4, 8, 16, 32] and a learning rate that spans from 1e-4 to 1e-2. The range of dropout rate in the search space is from 0.2 to 0.5 with a 0.05 increment. The goal of the Ray Tune search algorithm is to find optimal hyperparmeter values where the validation accuracy is highest among tuning experiment results. 

## Files
- `main.py`
: The main script for training and testing the LeNet model on the CIFAR-10 dataset.

- `tuning.py`
: The script for hyperparameter tuning using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

- `model.py`
: The script defining the modified LeNet model architecture.

## Usage
**Training and Testing**

To train and test the LeNet model on the CIFAR-10 dataset, run the following command:

```console 
python main.py
```
This will train the model for 20 epochs and test it on the test set. The training and testing results will be printed to the console.

**Hyperparameter Tuning**

To perform hyperparameter tuning using Ray Tune, run the following command:

```console 
python tuning.py
```
This will perform a hyperparameter search using the ASHA algorithm and report the best hyperparameters found.

Conclusion
In this project, we implemented the LeNet architecture for image classification on the CIFAR-10 dataset. I achieved an accuracy of 73% after hyperparameter tuning using Ray Tune. This project demonstrates the effectiveness of hyperparameter tuning in improving the performance of deep learning models.
