import numpy as np


class Neuron:
    def __init__(self, n_inputs=2):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1

    def backpropagation(self, X, y, prediction, learning_rate):
        bias_gradient = sigmoid_derivative(prediction) * binary_cross_entropy_loss_derivative(prediction, y)
        weight_gradients = np.array([x * bias_gradient for x in X])
        self.w -= learning_rate * weight_gradients
        self.b -= learning_rate * bias_gradient

    def train (self, X, y, epochs=10, learning_rate=0.05):
        for epoch in range(epochs + 1):
            losses = []
            for Xi, yi in zip(X, y):
                prediction, loss = self.forward_pass(Xi, yi)
                losses.append(loss)
                self.backpropagation(Xi, yi, prediction, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")

    def forward_pass(self, X, y=None):
        prediction = sigmoid_activation(np.dot(self.w, X) + self.b)
        loss = binary_cross_entropy_loss(prediction, y) if y is not None else None
        return prediction, loss


def sigmoid_activation(n):
    return 1 / (1 + np.exp(-n))
    
def sigmoid_derivative(n):
    return n * (1 - n)

def binary_cross_entropy_loss(prediction, target):
    return -target * np.log(prediction) - (1 - target) * np.log(1 - prediction)

def binary_cross_entropy_loss_derivative(prediction, target):
    return -target / prediction + (1 - target) / (1 - prediction)