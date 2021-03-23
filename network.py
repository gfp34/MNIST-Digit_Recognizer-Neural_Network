import numpy as np
import math
import random
import datetime

from mnist import MNIST


def main():
    data = MNIST('data')
    images, labels = data.load_training()
    test_images, test_labels = data.load_testing()
    training_data = [i for i in zip(images, labels)]
    to_ratios_and_vectors(training_data)
    test_data = [i for i in zip(test_images, test_labels)]
    to_ratios_and_vectors(test_data)

    net = Network([784, 50, 50, 10])
    net.train(training_data, 30, 100, 2.0, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def to_ratios_and_vectors(data):
    for i in range(len(data)):
        image = data[i]
        a_0 = np.zeros((len(image[0]), 1))
        for k in range(len(image[0])):
            a_0[k][0] = image[0][k] / 255
        y = vectorized_result(image[1])
        data[i] = (a_0, y)


class Network:

    def __init__(self, sizes: list):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.activations = [np.zeros((y, 1)) for y in sizes]
        self.z = [np.zeros(a.shape) for a in self.activations][1: self.num_layers]

    def train(self, training_data, epochs, mini_batch_size, eta, test_data):
        grade = self.evaluate(test_data) / len(test_data)
        print(f"Random: grade = {grade * 100}%")
        print("Start Training!")
        for epoch in range(1, epochs + 1):
            start = datetime.datetime.now()
            random.shuffle(training_data)
            mini_batches = [training_data[i: i + mini_batch_size] for i in range(0, len(training_data), mini_batch_size)]
            i = 1
            for mini_batch in mini_batches:
                cost = self.update_mini_batch(mini_batch, eta)
                # print(f"Epoch {epoch}, Mini-batch {i} / {len(mini_batches)}: Average Cost - {cost}")
                i += 1
            grade = self.evaluate(test_data) / len(test_data)
            end = datetime.datetime.now()
            print(f"Epoch {epoch} complete: grade = {grade * 100}%, time = {end - start}")

    def feed_forward(self, x, y):
        self.activations[0] = x
        for l in range(self.num_layers - 1):
            self.z[l] = self.weights[l] @ self.activations[l] + self.biases[l]
            self.activations[l+1] = sigmoid_v(self.z[l])
        return sum((self.activations[-1] - y) ** 2)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        costs = []

        for x, y in mini_batch:
            cost = self.feed_forward(x, y)
            costs.append(cost)

            delta_nabla_b, delta_nabla_w = self.backprop(y)
            for i in range(len(nabla_b)):
                nabla_b[i] += delta_nabla_b[i]
                nabla_w[i] += delta_nabla_w[i]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

        return sum(costs) / len(costs)

    def backprop(self, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        deltas = 2 * (self.activations[-1] - y) * sigmoid_prime_v(self.z[-1])
        for l in range(self.num_layers-2, -1, -1):
            nabla_b[l] = deltas
            nabla_w[l] = deltas @ self.activations[l].T
            if l > 0:
                deltas = (self.weights[l].T @ deltas) * sigmoid_prime_v(self.z[l-1])

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        correct = 0
        for image in test_data:
            self.feed_forward(image[0], image[1])
            if np.argmax(self.activations[-1]) == np.argmax(image[1]):
                correct += 1
        return correct


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


"""Sigmoid function for running on vectors"""
sigmoid_v = np.vectorize(sigmoid)

"""Sigmoid Prime for running on vectors"""
sigmoid_prime_v = np.vectorize(sigmoid_prime)

if __name__ == "__main__":
    main()
