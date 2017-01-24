"""
The Neuron and Network classes for the neural network

Create a basic network -
network = Network({'neurons': 2, 'neurons': 3, 'neurons': 1})

This will create a basic feedforward network with
- 1 input layer of 2 neurons
- 1 hidden layer of 3 neurons
- 1 output layer of 1 neuron
with the weights and biases randomly set using a random uniform initialiser
"""

import numpy as np

class Neuron:
  def __init__(self, input_count, init='DEFAULT', bias='DEFAULT', initialiser='uniform'):
    self.input_count = input_count
    self.initialiser = initialiser
    self.weights = []
    self.bias = -1
    self.bias_setting = bias
    if init == 'DEFAULT':
      self.init_bias()
      self.init_weights()
    else:
      self.weights = init['weights']
      self.bias = init['bias']
      self.bias_setting = init['bias']

  def init_bias(self):
    if self.input_count > 0:
      self.bias = self.init_weight() if self.bias_setting == 'DEFAULT' else self.bias_setting

  def init_weight(self):
    if self.initialiser == 'gaussian':
      return np.random.normal(0, (1 / np.sqrt(self.input_count)))
    else:
      return np.random.uniform(-(2 / self.input_count), (2 / self.input_count))

  def init_weights(self):
    if self.input_count > 0:
      self.weights = [self.init_weight() for x in range(0, self.input_count)]

  def reinitialise(self):
    self.init_bias()
    self.init_weights()
    return self

  def __str__(self):
    return 'Weights: ' + str(self.weights) + '\n' + 'Bias: ' + str(self.bias)

class Network:
  def __init__(self, layers, weights_initialiser='uniform'):
    self.activation_function = 'sigmoid'
    self.weights_initialiser = weights_initialiser
    self.momentum = 0.9
    self.step = 0.1
    self.assessment = 'RMSE'
    self.layers = []
    self.previous_weights = []
    self.configure(layers)

  def configure(self, layers):
    for (i, layer) in enumerate(reversed(layers)):
      input_count = 0 if (i == 0) else len(self.layers[-1])
      self.layers.append(
        [Neuron(
          input_count,
          init=(self.config_get_initial(layer, x))) for x in range(0, layer['neurons'])
        ])
    self.layers.reverse()

  def config_get_initial(self, layer, neuron_index):
    return layer['init'][neuron_index] if 'init' in layer else 'DEFAULT'

  def reinitialise(self):
    for layer in self.layers:
      layer.reinitialise()

  def forward_pass(self, inputs):
    output = []
    count = 1
    activations = []
    for (layer_index, layer) in enumerate(self.layers):
      output.append([])
      activations.append([])
      for neuron_index in enumerate(layer):
        if layer_index == 0:
          output[layer_index].append(
            {
              'i': count,
              'sum': 0,
              'activation': inputs[neuron_index],
              'comment': 'input'
            })
          count += 1
          activations[layer_index].append(inputs[neuron_index])
        else:
          sum_value = self.forward_pass_sum(layer_index, neuron_index, activations)
          activation = self.activate(sum_value)
          activations[layer_index].append(activation)
          output[layer_index].append(
            {
              'i': count,
              'sum': sum_value,
              'activation': activation,
              'comment': 'hidden' if (layer_index != (len(self.layers) - 1)) else 'output'
            })
          count += 1
    return output

  def forward_pass_sum(self, layer_index, neuron_index, activations):
    sum_value = self.layers[layer_index][neuron_index].bias * 1
    for (i, neuron) in enumerate(self.layers[layer_index - 1]):
      sum_value += neuron.weights[neuron_index] * activations[-2][i]
    return sum_value

  def backward_pass(self, forward_pass, expected):
    output = []
    for (layer_index, layer) in enumerate(reversed(forward_pass)):
      output.insert(0, [])
      for (neuron_index, pass_value) in enumerate(reversed(layer)):
        if layer_index == 0:
          output[0].append(self.backward_pass_input(pass_value, expected[neuron_index]))
        else:
          output[0].insert(
            0, self.backward_pass_hidden(layer_index, neuron_index, forward_pass, output))
    return output

  def backward_pass_input(self, pass_value, expected):
    return (expected - pass_value['activation']) * self.activate(pass_value['activation'], True)

  def backward_pass_hidden(self, layer_index, neuron_index, forward_pass, output):
    actual_l = (len(self.layers) - 1) - layer_index
    actual_n = (len(self.layers[actual_l]) - 1) - neuron_index
    neuron = self.layers[actual_l][actual_n]
    pass_value = forward_pass[actual_l][actual_n]
    sum_value = 0
    derivative = self.activate(pass_value['activation'], True)
    for (i, weight) in enumerate(neuron.weights):
      sum_value += (weight * output[1][i])
    sum_value = sum_value * derivative
    return sum_value

  def calculate_new_weight(self, weight, delta, activation):
    weight_change = (self.step * delta * activation)
    return weight + weight_change

  def update_weights(self, forward_pass, deltas):
    new_previous_weights = []
    # weight_deltas = []
    for (l_index, layer) in enumerate(self.layers):
      for (n_index, neuron) in enumerate(layer):
        for (w_index, weight) in enumerate(neuron.weights):
          if l_index != (len(self.layers) - 1):
            weight_activation = forward_pass[l_index][n_index]['activation']
            weight_delta = deltas[l_index + 1][w_index]
            # if forward_pass[l][n]['i'] <= len(weight_deltas):
            #   new_weight += (self.momentum * weight_deltas[forward_pass[l][n]['i'] - 1])
            # new_previous_weights.append(new_weight)
            # if len(new_previous_weights) <= len(self.previous_weights):
            #   previous_weight = self.previous_weights[len(new_previous_weights) - 1]
            #   weight_deltas.append(previous_weight - new_weight)
            neuron.weights[w_index] = self.calculate_new_weight(
              weight, weight_delta, weight_activation)
        bias_delta = deltas[l_index][n_index]
        bias_activation = forward_pass[l_index][n_index]['activation']
        neuron.bias = self.calculate_new_weight(neuron.bias, bias_delta, bias_activation)
    self.previous_weights = new_previous_weights

  def solve(self, inputs):
    return [pass_value['activation'] for pass_value in self.forward_pass(inputs)[-1]]

  def train(self, inputs, expected):
    forwards = self.forward_pass(inputs)
    backwards = self.backward_pass(forwards, expected)
    self.update_weights(forwards, backwards)
    return self

  def error(self):
    return 0

  def activate(self, i, deriv=False):
    return self.sigmoid(i, deriv)

  def sigmoid(self, i, deriv):
    if deriv:
      return i*(1-i)
    return 1/(1+np.exp(-i))

  def __str__(self):
    return '\n'.join(
      ['Layer ' + str(i + 1) + '\n-------\n' + str(
        '\n'.join(
          ['Neuron ' + str(n + 1) + '\n' + str(neuron) + '\n' for (n, neuron) in enumerate(layer)
          ])
        ) for (i, layer) in enumerate(self.layers)
      ]
    )
