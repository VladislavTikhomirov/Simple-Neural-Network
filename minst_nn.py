import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import randint
from PIL import Image
from image_input import load_image
import os

# mnist number set
data = pd.read_csv('datasets/mnist_train.csv')
test_data = pd.read_csv('datasets/mnist_test.csv')
#  mnist fashion sert
# data = pd.read_csv('datasets/fashion-mnist_train.csv')

# Display the first few rows
print(data.head())

# alternate_data = pd.read_csv('datasets/')
# print(alternate_data.head())

# Convert data to numpy array
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
print(data)

# Split into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Initialize parameters for the network: 784 -> 24 -> 24 -> 10
def init_params():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(24, 128) - 0.5
    b2 = np.random.rand(24, 1) - 0.5
    W3 = np.random.rand(10, 24) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# ReLU derivative for backpropagation
def ReLU_deriv(Z):
    return Z > 0

# One-hot encoding for labels
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backpropagation
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

# Prediction and accuracy functions
def get_predictions(A3):
    return np.argmax(A3, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 100 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}, Accuracy: {accuracy}")
    return W1, b1, W2, b2, W3, b3

# Train the model
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.10, 1000)

# Function to make predictions
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

# Testing a prediction
def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, True Value: {label} (I AM TRAIN DATA!)")
    plt.show()

# Check accuracy on the development set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
print("Development set accuracy:", get_accuracy(dev_predictions, Y_dev))


# Load and preprocess test data
test_data = np.array(test_data)
m_test, n_test = test_data.shape

# Separate test data into labels and inputs
Y_test = test_data[:, 0]  # First column is the label
X_test = test_data[:, 1:n_test]  # Remaining columns are the input features
X_test = X_test.T / 255.  # Normalize and transpose

# Function to display and compare test images with predictions
def display_test_predictions(index, W1, b1, W2, b2, W3, b3, X_test, Y_test):
    # Extract the image and its true label from the test set
    current_image = X_test[:, index, None]
    true_label = Y_test[index]
    
    # Make a prediction using the trained model
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
    
    # Reshape and scale the image for display (28x28 grayscale)
    current_image = current_image.reshape((28, 28)) * 255
    
    # Display the image and the predicted vs. actual label
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, True Value: {true_label} (I AM TEST DATA!)")
    plt.show()

def display_test_results():
    # Example usage: Display a few predictions from the test set
    for i in range(5):  # Display the first 5 test samples
        display_test_predictions(i, W1, b1, W2, b2, W3, b3, X_test, Y_test)

    # Make predictions on test set using the trained model
    test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)

    # Evaluate the accuracy on test data
    test_accuracy = get_accuracy(test_predictions, Y_test)
    print("Test set accuracy:", test_accuracy)

display_test_results()

image_folder = 'utils/'
def test_image_results(W1, b1, W2, b2, W3, b3):
    img_data = ['utils/' + file for file in os.listdir(image_folder) if file.endswith('.png')]
    random_choice = randint(0, len(img_data)-1)
    cur_image = img_data[random_choice]
    prediction = make_predictions(cur_image, W1, b1, W2, b2, W3, b3)
    cur_image = cur_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(cur_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, True Value: {cur_image}")
    plt.show()

test_image_results(W1,b1,W2,b2,W3,b3)





plt.figure(data)



