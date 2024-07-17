# Model Training and Evaluation Guide

Welcome to the Model Training and Evaluation Guide for the DSLR project. This guide will help you understand the different ways to train your logistic regression model, make predictions, and evaluate the model's accuracy using the provided scripts.


## Table of Contents

- [Introduction](#introduction)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Evaluating Model Accuracy](#evaluating-model-accuracy)


## Introduction

In this project, we focus on training a logistic regression model to solve classification problems. Logistic regression is a linear model for binary classification that estimates the probability that a given input belongs to a certain class.

We provide three scripts to help you train, predict, and evaluate your logistic regression model:

1. `logreg_train.py` - Trains the logistic regression model.
2. `logreg_predict.py` - Uses the trained model to make predictions on new data.
3. `logreg_accuracy.py` - Evaluates the accuracy of the trained model.


## Training the Model

The `logreg_train.py` script trains the logistic regression model using the specified dataset. You can choose between batch gradient descent and stochastic gradient descent for training.

#### Usage

```bash
python logreg_train.py [dataset] [--model : batch | stochastic = batch]
```

#### Options

* dataset: Path to the dataset file (CSV format).
* --model: Optional parameter to specify the training method. Choices are batch (default) or stochastic.

#### Example

```bash
python logreg_train.py datasets/dataset_train.csv --model stochastic
```

This command trains the logistic regression model using stochastic gradient descent on the provided dataset. It then saves the weights and biases to a `model.json` file.


## Making Predictions

The logreg_predict.py script uses the trained model to make predictions on a new dataset.

#### Usage

```bash
python logreg_predict.py [dataset] [model_file] [--model : batch | stochastic = batch]
```

#### Options

* dataset: Path to the dataset file (CSV format) for making predictions.
* model_file: Path to the trained model file (JSON format).
* --model: Optional parameter to specify the model type. Choices are batch (default) or stochastic.

#### Example

```bash
python logreg_predict.py datasets/dataset_test.csv model.json --model stochastic
```

This command uses the stochastic gradient descent-trained model to make predictions on the dataset.


## Evaluating Model Accuracy

The logreg_accuracy.py script evaluates the accuracy of the trained model using a specified dataset.

#### Usage

```bash
python logreg_accuracy.py [dataset] [model_file] [--model : batch | stochastic = batch]
```

#### Options

* dataset: Path to the dataset file (CSV format) for evaluation.
* model_file: Path to the trained model file (JSON format).
* --model: Optional parameter to specify the model type. Choices are batch (default) or stochastic.

#### Example

```bash
python logreg_accuracy.py datasets/dataset_train.csv model.json --model stochastic
```

## Conclusion

By using these scripts, you can effectively train, predict, and evaluate your logistic regression model. This guide provides the necessary commands and options to help you navigate through each process.
