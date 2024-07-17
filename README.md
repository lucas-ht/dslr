# DSLR: DataScience x Logistic Regression

Welcome to the DSLR project! This project is part of the 42 School curriculum and focuses on exploring data and implementing a logistic regression model for classification tasks. This README will provide an overview of the project, the theory behind logistic regression, and links to guides for practical usage.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Guides](#guides)
- [Theory](#theory)
  - [Logistic Regression](#logistic-regression)
  - [Sigmoid Function](#sigmoid-function)
  - [Perceptron](#perceptron)
  - [Gradient Descent](#gradient-descent)
- [Subject](#Subject)
- [Acknowledgements](#acknowledgements)


## Introduction

The DSLR project aims to introduce you to the basics of data science and machine learning, specifically focusing on data exploration and logistic regression. While this project does not cover the entire scope of data science, it provides a solid foundation for understanding and implementing logistic regression models.


## Installation

To get started with the DSLR project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/lucas-ht/dslr.git
cd dslr
pip install -r requirements.txt
```

## Guides

For detailed instructions on data exploration and model training, refer to the following guides:

* [Data Exploration Guide](/docs/data_exploration.md): Explains the different ways to explore and visualize your dataset.

* [Model Training and Evaluation Guide](/docs/model.md): Details the process of training your logistic regression model, making predictions, and evaluating accuracy.


## Theory


### Logistic Regression

Logistic regression is a statistical method used for binary classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a certain class. The output of logistic regression is a value between 0 and 1, which can be interpreted as the probability of the positive class.

The logistic regression model is defined as:

$` P(y=1|X) = \sigma(\beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n) `$

where:
* $` P(y=1|X) `$ is the probability of the positive class given the input features $` X `$.

* $` \sigma `$ is the [sigmoid function](#sigmoid-function).

* $` \beta_0, \beta_1, \ldots, \beta_n `$ are the model parameters.


### Sigmoid Function

The sigmoid function, denoted as $` \sigma `$, is used in logistic regression to map predicted values to probabilities. It is defined as:

$` \sigma(z) = \frac{1}{1 + e^{-z}} `$

where $` z `$ is the linear combination of input features and model parameters. The sigmoid function outputs a value between 0 and 1, making it suitable for binary classification.

The limits of the sigmoid function are:
* As $` z `$ approaches $` +\infty `$, $` \sigma(z) `$ approaches 1.
* As $` z `$ approaches $` -\infty `$, $` \sigma(z) `$ approaches 0.


### Perceptron

The perceptron is a type of artificial neuron used in machine learning for binary classification tasks. It is one of the simplest types of artificial neural networks and forms the basis for more complex neural network architectures.

A perceptron makes its predictions based on a linear predictor function combining a set of weights with the feature vector, followed by an activation function.

> [!NOTE]
> In this case, the activation function is the sigmoid function, this is done so that the perceptron output can be interpreted as a probability.

The perceptron model is defined as:

$` y = \sigma(\mathbf{w} \cdot \mathbf{x} + b) `$

where:
* $` \mathbf{w} `$ is the weight vector.
* $` \mathbf{x} `$ is the input feature vector.
* $` b `$ is the bias term.
* $` \sigma `$ is the [sigmoid function](#sigmoid-function).

The perceptron algorithm updates the weights and bias based on the prediction error:

$` \mathbf{w} := \mathbf{w} + \Delta \mathbf{w} `$

$` \Delta \mathbf{w} = \eta (y_{true} - y_{pred}) \mathbf{x} `$

$` b := b + \eta (y_{true} - y_{pred}) `$

where:
* $` \eta `$ is the learning rate.
* $` y_{true} `$ is the true label.
* $` y_{pred} `$ is the predicted label.


### Gradient Descent

Logistic regression models are typically trained using gradient descent optimization. There are two main types of gradient descent used in this project:
* **Batch Gradient Descent**: Computes the gradient using the entire dataset.
* **Stochastic Gradient Descent**: Computes the gradient using a single data point at each iteration.


## Subject

[The subject can be found here.](/assets/subject.pdf)


## Acknowledgements

This project is part of the 42 School curriculum. Special thanks to the 42 School community for their support and resources.
