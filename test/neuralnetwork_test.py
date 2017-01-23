import numpy as np
from ..neuralnetwork import Neuron, Network

def test_neuron_simple_instance():
  inputCount = 2
  neuron = Neuron(inputCount)
  assert isinstance(neuron, Neuron)
  assert neuron.inputCount == inputCount
  assert neuron.biasSetting == 'DEFAULT'
  assert neuron.initialiser == 'uniform'
  assert isinstance(neuron.weights, list)
  assert len(neuron.weights) == inputCount
  for weight in neuron.weights:
    assert weight >= -1
    assert weight <= 1
  assert neuron.bias >= -1
  assert neuron.bias <= 1

def test_neuron_initialiser_gaussian():
  inputCount = 2
  minWeight = 0 - (4 * (1 / np.sqrt(inputCount)))
  maxWeight = 0 + (4 * (1 / np.sqrt(inputCount)))
  neuron = Neuron(inputCount, initialiser='gaussian')
  for weight in neuron.weights:
    assert weight >= minWeight
    assert weight <= maxWeight
  assert neuron.bias >= minWeight
  assert neuron.bias <= maxWeight

def test_neuron_reinitialise():
  inputCount = 2
  neuron = Neuron(inputCount)
  originalWeights = list(neuron.weights)
  originalBias = neuron.bias
  neuron.reinitialise()
  assert(neuron.weights != originalWeights)
  assert(neuron.bias != originalBias)

def test_neuron_str():
  inputCount = 2
  neuron = Neuron(inputCount)
  assert str(neuron) == 'Weights: ' + str(neuron.weights) + '\n' + 'Bias: ' + str(neuron.bias)

def test_network_simple_instance():
  network = Network([ { 'neurons': 1 }, { 'neurons': 2 }, { 'neurons': 3 } ])
  assert isinstance(network, Network)
  assert network.weightsInitialiser == 'uniform'
  assert network.momentum == 0.9
  assert network.step == 0.1
  assert isinstance(network.layers, list)
  assert len(network.layers) == 3
  for (i, layer) in enumerate(network.layers):
    assert isinstance(layer, list)
    assert len(layer) == (i + 1)
    for n in layer:
      assert isinstance(n, Neuron)
      assert isinstance(n.weights, list)
      if ((i + 1) == len(network.layers)):
        assert n.inputCount == 0

def test_network_str():
  network = Network([ { 'neurons': 1 }, { 'neurons': 2 }, { 'neurons': 3 } ])
  assert str(network) == '\n'.join(
    [ 'Layer ' + str(i + 1) + '\n-------\n' + str(
        '\n'.join(
          [ 'Neuron ' + str(n + 1) + '\n' + str(neuron) + '\n' for (n, neuron) in enumerate(layer)
          ])
      ) for (i, layer) in enumerate(network.layers)
    ]
  )
