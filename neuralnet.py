import random
import math
import copy

# used for neuron output. If math.exp gives and
# extremely large number, the result would approach
# 0, but if would have an overflow error, so
# just in case I would return 0
def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0

# uses a bunch of weights and inputs and gives
# a result (fires?) based on the sigmoid function
class Neuron:
    weights = []

    def __init__(self, weights):
        self.weights = weights

    def simulate(self, inputs):
        result = 0
        for i in range(len(inputs)):
            result += inputs[i] * self.weights[i]
        return sigmoid(result)

# has a bunch of neurons that feed input to each other to
# produce an output
class NeuralNet:
    layers = []
    learning_rate = .1

    # initialize with random weights
    def __init__(self, learn_rate, input_size, neuron_nums):
        self.learning_rate = learn_rate
        for neuron_num in neuron_nums:
            layer = []
            for i in range(neuron_num):
                weights = [random.uniform(-1.0, 1.0) for _ in range(input_size + 1)]
                layer.append(Neuron(weights))
            self.layers.append(layer)
            input_size = neuron_num

    # gets the outputs of all the nodes based on an input
    def test(self, inputs):
        row_input = copy.copy(inputs)
        output = []
        for i in range(len(self.layers)):
            row_input.append(-1)
            layer = []
            for j in range(len(self.layers[i])):
                layer.append(self.layers[i][j].simulate(row_input))
            output.append(layer)
            row_input = copy.copy(layer)
        return output

    # tests the input, and gives a prediction based on it
    def predict(self, inputs):
        results = self.test(inputs)
        outputs = results[-1]
        return outputs.index(max(outputs))

    # compares to the output and does back propagation
    def calc_errors(self, output, targets):
        errors = []
        for i in range(1, len(output) + 1):
            error_layer = []
            output_layer = output[-i]
            for j in range(len(output_layer)):
                if i == 1:
                    error = output_layer[j] - targets[j]
                else:
                    error = 0
                    for k in range(len(output[-i+1])):
                        error += self.layers[-i+1][k].weights[j] * output[-i+1][k]
                error *= output_layer[j] * (1 - output_layer[j])
                error_layer.append(error)
            errors.insert(0, error_layer)
        return errors

    # updates weights based on errors
    def fix_errors(self, inputs, output, errors):
        for i in range(len(errors)):
            for j in range(len(errors[i])):
                if i == 0:
                    activations = inputs
                else:
                    activations = output[i-1]
                for k in range(len(self.layers[i][j].weights) - 1):
                    change = self.learning_rate * errors[i][j] * activations[k]
                    self.layers[i][j].weights[k] -= change
                self.layers[i][j].weights[-1] += self.learning_rate * errors[i][j] * -1

    # trains based on a single input
    def train(self, inputs, target):
        targets = [0] * len(self.layers[-1])
        targets[target] = 1
        output = self.test(inputs)
        errors = self.calc_errors(output, targets)
        self.fix_errors(inputs, output, errors)
