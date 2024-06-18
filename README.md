# Federated Learning with PyTorch on MNIST Dataset

This repository contains an implementation of a simple federated learning setup using PyTorch on the MNIST dataset. The goal is to simulate a federated learning scenario where multiple clients train on their local data and then send their updates to a central server for aggregation.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
  - [Device Setup](#device-setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Model Architecture](#model-architecture)
  - [Client Class](#client-class)
  - [Federated Learning Process](#federated-learning-process)
  - [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction

Federated learning is a machine learning setting where multiple clients (e.g., mobile devices or organizations) collaboratively train a model under the orchestration of a central server while keeping the training data decentralized. This implementation demonstrates a basic federated learning workflow using the MNIST dataset.

## Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- PyTorch
- torchvision
- numpy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Surajrs812/Federated-Learning-with-PyTorch-on-MNIST-Dataset.git
   cd federated-learning-mnist
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision numpy
   ```

## Usage

Run the Python script to start the federated learning simulation:
```bash
python federated_learning.py
```

## Code Overview

### Device Setup

The code automatically selects GPU if available, otherwise, it uses CPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Dataset Preparation

The MNIST dataset is downloaded and split into training, validation, and test sets. Additionally, the training set is further split among multiple clients:
```python
train_dataset = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
train_dataset, dev_dataset = random_split(train_dataset, [50000, 10000])
```

### Model Architecture

A simple convolutional neural network (CNN) is defined to classify MNIST digits:
```python
class FederatedNet(nn.Module):
    def __init__(self):
        super(FederatedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        ...
```

### Client Class

Each client has its local dataset and training loop:
```python
class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset

    def train(self, global_model, epochs, lr):
        ...
        return model.state_dict()
```

### Federated Learning Process

The central server aggregates the updates from the clients and updates the global model:
```python
global_model = FederatedNet().to(device)
for round_num in range(rounds):
    ...
    global_model.load_state_dict(new_global_state_dict)
```

### Evaluation

After each round, the global model is evaluated on the training and validation sets:
```python
def evaluate(model, dataset):
    ...
train_loss, train_acc = evaluate(global_model, train_dataset)
dev_loss, dev_acc = evaluate(global_model, dev_dataset)
```

## Results

The results include training loss, accuracy, and validation loss, accuracy for each round. The final test performance is also evaluated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
