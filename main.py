import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math


def create_weights(X, Y):
    W = X.transpose() @ Y
    return W


def sign_function(vector):
    res = np.array([], int)
    for i in vector:
        # res = np.append(res, 1/(1+math.exp(2*i)))
        if i > 0:
            res = np.append(res, 1)
        elif i < 0:
            res = np.append(res, -1)
        elif i == 0:
            res = np.append(res, i)

    return res


def convert_to_bipolar(matrix):
    if len(matrix.shape) == 1:
        for i in range(matrix.shape[0]):
            matrix[i] = 1 if matrix[i] > 0 else -1
    else:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = 1 if matrix[i, j] > 0 else -1
    return matrix


def input_data_to_vector(matrix):
    res = np.array([])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j, 0] == 255:
                res = np.append(res, 1)
            else:
                res = np.append(res, 0)
    return res


def vector_to_output_data(vector):

    pass


def main_work(input_data, W):
    W_1 = W.transpose()
    Y_prev = []
    Y = sign_function(input_data @ W)
    X_prev = input_data
    X = sign_function(Y @ W_1)
    while True:
        if (np.array_equal(Y, Y_prev) and np.array_equal(X, X_prev)):
            break
        print("iteration")
        Y_prev = Y
        Y = sign_function(X @ W)
        X_prev = X
        X = sign_function(Y @ W_1)
    return Y, X

def digit_to_bin_vec(digit):
    digit = bin(digit)[2:].zfill(4)
    res = [int(i) for i in digit]
    return res

def vec_in_digit(vec):
    digit = 0
    for i in range(len(vec)):
        if vec[i] == 1:
            digit += 2**(len(vec)-i-1)
    return digit
# X = np.array([[-1, 1, -1], [1, -1, 1 ]], int)
# Y = np.array([[-1, 1, -1, -1], [1, -1, 1, 1]], int)
X = []
Y = []

for i in range(0, 3):
    image = mpimg.imread(f"reference_data/{str(i)}.bmp")
    raw_data = np.array(image)
    input_vector_data = input_data_to_vector(raw_data)
    X.append(input_vector_data)
    Y.append(digit_to_bin_vec(i))

X = convert_to_bipolar(np.array(X))
Y = convert_to_bipolar(np.array(Y))

W = create_weights(X, Y)

image = mpimg.imread("reference_data/0.bmp")
raw_data = np.array(image)
input_vector_data = convert_to_bipolar(input_data_to_vector(raw_data))
# print(np.array_equal(X[1], input_vector_data))
y, x = main_work(input_vector_data, W)
plt.imshow(x.reshape(74, 51))
plt.show()
print(y)
