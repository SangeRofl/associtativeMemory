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



# import pprint
#
# def init_BAM(data):
# 	ab = []
# 	for ele in data:
# 		ab.append([generate_bipolar_form(ele[0]), generate_bipolar_form(ele[1])])
#
# 	x_length = len(ab[0][1])
# 	y_length = len(ab[0][0])
#
# 	_bam = [] #initialise empty bam array
# 	temp = []
# 	for ele in range(y_length):
# 		temp = [0] * x_length
# 		_bam.append(temp)
#
# 	return ab, x_length, y_length, _bam
#
# def create_BAM(ab, _bam):
# 	for ele in ab:
# 		X = ele[0]
# 		Y = ele[1]
# 		for ix, xi in enumerate(X):
# 			for iy, yi in enumerate(Y):
# 				_bam[ix][iy] += xi * yi
# 	return _bam
#
# def get_associations(A, _bam, x_length, y_length):
# 	A = multiply_vec(A, _bam, x_length, y_length)
# 	return threshold(A)
#
# def multiply_vec(vec, _bam, x_length, y_length):
# 	result = [0] * x_length
# 	for x in range(x_length):
# 		for y in range(y_length):
# 			result[x] += vec[y] * _bam[y][x]
# 	return result
#
# def generate_bipolar_form(vec):
# 	result = []
# 	for ele in vec:
# 		if ele == 0:
# 			result.append(-1)
# 		else:
# 			result.append(1)
# 	return result
#
# def threshold(vec):
# 	result = []
# 	for ele in vec:
# 		if ele < 0:
# 			result.append(0)
# 		else:
# 			result.append(1)
# 	return result
#
#
# def main():
# 	A = [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]]
# 	B = [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]]
#
# 	data_associations = [A, B]
# 	ab, x_length, y_length, bam = init_BAM(data_associations)
# 	bam_matrix = create_BAM(ab, bam)
# 	pp = pprint.PrettyPrinter(indent = 4)
# 	print("Bam Matrix: ")
# 	pp.pprint(bam_matrix)
# 	print("\n")
# 	print("[1, 0, 1, 0, 1, 0] :- ", get_associations([1, 0, 1, 0, 1, 0], bam_matrix, x_length, y_length))
# 	print("\n")
# 	print("[1, 1, 1, 0, 0, 0] :- ", get_associations([1, 1, 1, 0, 0, 0], bam_matrix, x_length, y_length))
#
# if __name__ == '__main__':
# 	main()