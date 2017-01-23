import numpy as np
from ..neuralnetwork import Neuron, Network

def test_neuron_simple_instance():
  input_count = 2
  neuron = Neuron(input_count)
  assert isinstance(neuron, Neuron)
  assert neuron.inputCount == input_count
  assert neuron.biasSetting == 'DEFAULT'
  assert neuron.initialiser == 'uniform'
  assert isinstance(neuron.weights, list)
  assert len(neuron.weights) == input_count
  for weight in neuron.weights:
    assert weight >= -1
    assert weight <= 1
  assert neuron.bias >= -1
  assert neuron.bias <= 1

def test_neuron_initialiser_gaussian():
  input_count = 2
  min_weight = 0 - (4 * (1 / np.sqrt(input_count)))
  max_weight = 0 + (4 * (1 / np.sqrt(input_count)))
  neuron = Neuron(input_count, initialiser='gaussian')
  for weight in neuron.weights:
    assert weight >= min_weight
    assert weight <= max_weight
  assert neuron.bias >= min_weight
  assert neuron.bias <= max_weight

def test_neuron_reinitialise():
  input_count = 2
  neuron = Neuron(input_count)
  original_weights = list(neuron.weights)
  original_bias = neuron.bias
  neuron.reinitialise()
  assert neuron.weights != original_weights
  assert neuron.bias != original_bias

def test_neuron_str():
  input_count = 2
  neuron = Neuron(input_count)
  assert str(neuron) == 'Weights: ' + str(neuron.weights) + '\n' + 'Bias: ' + str(neuron.bias)

def test_network_simple_instance():
  network = Network([{'neurons': 1}, {'neurons': 2}, {'neurons': 3}])
  assert isinstance(network, Network)
  assert network.weightsInitialiser == 'uniform'
  assert network.momentum == 0.9
  assert network.step == 0.1
  assert isinstance(network.layers, list)
  assert len(network.layers) == 3
  for (i, layer) in enumerate(network.layers):
    assert isinstance(layer, list)
    assert len(layer) == (i + 1)
    for neuron in layer:
      assert isinstance(neuron, Neuron)
      assert isinstance(neuron.weights, list)
      if (i + 1) == len(network.layers):
        assert neuron.inputCount == 0

def test_network_str():
  network = Network([{'neurons': 1}, {'neurons': 2}, {'neurons': 3}])
  assert str(network) == '\n'.join(
    ['Layer ' + str(i + 1) + '\n-------\n' + str(
      '\n'.join(
        ['Neuron ' + str(n + 1) + '\n' + str(neuron) + '\n' for (n, neuron) in enumerate(layer)
        ])
      ) for (i, layer) in enumerate(network.layers)
    ]
  )
