# Genetic Algorithm for Hyperparameter Optimization of a Convolutional Neural Network

This project implements a **Genetic Algorithm (GA)** to optimize the hyperparameters of a **Convolutional Neural Network (CNN)**. The CNN is trained on the **Sign Language MNIST (sign_mnist)** dataset, which contains images of signed letters. The objective is to find the best combination of hyperparameters that maximizes the classification accuracy on the test dataset within a fixed number of epochs.

## Features
- **Hyperparameter Optimization**: Utilizes a genetic algorithm to search for optimal hyperparameters.
- **Deep Learning Framework**: The CNN is built using [PyTorch](https://pytorch.org/).
- **Dataset**: The `sign_mnist` dataset is used, consisting of hand-signed letters for image classification tasks.
- **Performance-Oriented**: The algorithm focuses on maximizing the test accuracy by fine-tuning critical hyperparameters.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/genetic-algorithm-cnn.git
   cd genetic-algorithm-cnn```

  2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt```
## Dataset

The `sign_mnist` dataset can be downloaded from Kaggle. Place the dataset in the appropriate directory (e.g., `./data`) before running the training script.










