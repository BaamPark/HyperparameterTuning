# CIFAR-10 Image Classification using LeNet
This project aims to classify images from the CIFAR-10 dataset using the LeNet architecture. The LeNet architecture is a convolutional neural network (CNN) that was introduced in 1998 by Yann LeCun et al. for handwritten digit recognition. It consists of two sets of convolutional and average pooling layers, followed by a flattening layer, two fully connected layers, and a softmax classifier.

## Files
- `main.py`
: The main script for training and testing the LeNet model on the CIFAR-10 dataset.

- `tuning.py`
: The script for hyperparameter tuning using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)

- `model.py`
: The script defining the modified LeNet model architecture.

## Usage
Training and Testing
To train and test the LeNet model on the CIFAR-10 dataset, run the following command:

Copy codepython main.py
This will train the model for 20 epochs and test it on the test set. The training and testing results will be printed to the console.

Hyperparameter Tuning
To perform hyperparameter tuning using Ray Tune, run the following command:

Copy codepython tuning.py
This will perform a hyperparameter search using the ASHA algorithm and report the best hyperparameters found.

Dependencies
Python 3.6+
PyTorch 1.9.0
torchvision 0.10.0
numpy 1.19.5
matplotlib 3.4.2
Ray 1.6.0
Results
The LeNet model achieved an accuracy of 70% on the test set after Best trial config: {'lr': 0.0018573094008814584, 'batch_size': 16, 'dropout_rate': 0.25}

Hyperparameter tuning using Ray Tune improved the accuracy to 73% by selecting a learning rate of 0.0001, a batch size of 16, and a dropout rate of 0.25.

Conclusion
In this project, we implemented the LeNet architecture for image classification on the CIFAR-10 dataset. We achieved an accuracy of 70% on the test set using the default hyperparameters and 73% after hyperparameter tuning using Ray Tune. This project demonstrates the effectiveness of hyperparameter tuning in improving the performance of deep learning models.

## Reference
