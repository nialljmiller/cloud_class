Cloud Classification Neural Network

This repository contains a neural network designed for cloud classification using convolutional neural networks.
Introduction

This code implements a convolutional neural network (CNN) architecture for cloud classification from FITS files and JPEG images. The network architecture comprises several convolutional and linear layers.
Requirements

    Python 3.x
    PyTorch
    NumPy
    Matplotlib
    Astropy

Installation

    Clone the repository:

    bash

git clone https://github.com/your_username/your_repo.git

Install the required dependencies:

bash

    pip install -r requirements.txt

Usage

Ensure you have the necessary dataset structure and paths set up. Adjust the clear_paths, notclear_paths, and fits_paths variables in the Trainer class according to your dataset locations.

Run the trainer.train() method to start the training process.

python

from cloud_classification import Trainer

trainer = Trainer()
trainer.train()

Architecture

The neural network architecture consists of several convolutional layers followed by linear layers, defining the feature extraction and classification process.
Customization

    Adjust batch size, learning rate, number of steps, and other training parameters in the Trainer class initialization.
    Modify the network architecture in the Net class according to specific requirements.

Acknowledgments

    This code was developed by [Your Name].
    Dataset used: [Dataset Source/Description].

License

This project is licensed under [License Name]. See the LICENSE file for details.

Feel free to replace placeholders like [Your Name], [Dataset Source/Description], and [License Name] with the appropriate information. This template provides a basic structure for users to understand your code repository. Adjust and expand it as needed!
