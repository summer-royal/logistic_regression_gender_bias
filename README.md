# Logistic Regression for Gender Prediction

This code implements logistic regression using word embeddings for gender prediction. It uses pre-trained word vectors from the GloVe word embedding model.

## Dependencies
- Python 3.x
- torchtext (for accessing GloVe word vectors)
- numpy
- requests
- zipfile
- io

## Dataset
The code expects a text file `train.txt` to be present in the `unit1_resources` directory. This file contains labeled examples for training logistic regression.

## Word Vectors
The code downloads the pre-trained GloVe word vectors from the Stanford University website and loads them using the `torchtext.vocab.GloVe` class. The word vectors used in this code have a size of 300.

## Functions
The code includes several functions to perform logistic regression and make predictions.

### `get_word_vector(word)`
This function retrieves the word vector for a given word from the GloVe word vectors.

### `read_train_examples()`
This function reads the labeled training examples from the `train.txt` file. It returns a list of tuples, where each tuple contains a word and its corresponding label.

### `sigmoid(z)`
This function applies the sigmoid activation function to a given input `z` and returns the result.

### `compute_logistic_regression(word, weights, bias)`
This function computes the logistic regression prediction for a given word using the provided weights and bias. It calculates the dot product of the word vector and the weights, applies the sigmoid function, and returns the result.

### `fit_logistic_regression(O000O0OO0OOOOO000, OOO000O0O000OOO00=1000, O00O0O0OOO00O00OO=0.001)`
This function performs logistic regression training using the provided training examples. It initializes the weights randomly, iterates over the training examples, updates the weights using gradient descent, and prints the loss at regular intervals. It returns the learned weights and bias.

### `print_test_output(test_examples, weights, bias)`
This function makes predictions for the provided test examples using the logistic regression model. It prints the predicted gender for each test example based on the computed logistic regression values.

## Usage
To use this code, ensure that the required dependencies are installed. Place the `train.txt` file in the `unit1_resources` directory. Then, run the code to train the logistic regression model and make predictions on the test examples.

Feel free to modify the code as needed for your specific use case.
