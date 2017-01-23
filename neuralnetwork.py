import numpy as np

class Neuron:
  def __init__(self, inputCount, init='DEFAULT', bias='DEFAULT', initialiser='uniform'):
    self.inputCount = inputCount
    self.initialiser = initialiser
    self.weights = []
    self.bias = -1
    self.biasSetting = bias
    if init == 'DEFAULT':
      self.initBias()
      self.initWeights()
    else:
      self.weights = init['weights']
      self.bias = init['bias']
      self.biasSetting = init['bias']

  def initBias(self):
    if self.inputCount > 0:
      self.bias = self.initWeight() if self.biasSetting == 'DEFAULT' else self.biasSetting

  def initWeight(self):
    if self.initialiser == 'gaussian':
      return np.random.normal(0, (1 / np.sqrt(self.inputCount)))
    else:
      return np.random.uniform(-(2 / self.inputCount), (2 / self.inputCount))

  def initWeights(self):
    if self.inputCount > 0:
      self.weights = [self.initWeight() for x in range(0, self.inputCount)]

  def reinitialise(self):
    self.initBias()
    self.initWeights()
    return self

  def __str__(self):
    return 'Weights: ' + str(self.weights) + '\n' + 'Bias: ' + str(self.bias)

class Network:
  def __init__(self, layers, weightsInitialiser='uniform'):
    self.activationFunction = 'sigmoid'
    self.weightsInitialiser = weightsInitialiser
    self.momentum = 0.9
    self.step = 0.1
    self.assessment = 'RMSE'
    self.layers = []
    self.previousWeights = []
    self.configure(layers)

  def configure(self, layers):
    for (i, layer) in enumerate(reversed(layers)):
      inputCount = 0 if (i == 0) else len(self.layers[-1])
      self.layers.append(
        [Neuron(
          inputCount,
          init=(self.configGetInitial(layer, x))) for x in range(0, layer['neurons'])
        ])
    self.layers.reverse()

  def configGetInitial(self, layer, x):
    return layer['init'][x] if 'init' in layer else 'DEFAULT'

  def reinitialise(self):
    for layer in self.layers: layer.reinitialise()

  def forwardPass(self, inputs):
    output = []
    count = 1
    activations = []
    for (l, layer) in enumerate(self.layers):
      output.append([])
      activations.append([])
      for (n, neuron) in enumerate(layer):
        if l == 0:
          output[l].append(
            { 'i': count,
              'sum': 0,
              'activation': inputs[n],
              'comment': 'input'
            })
          count += 1
          activations[l].append(inputs[n])
        else:
          sumValue = self.fpSum(l, n, activations)
          activation = self.activate(sumValue)
          activations[l].append(activation)
          output[l].append(
            { 'i': count,
              'sum': sumValue,
              'activation': activation,
              'comment': 'hidden' if (l != (len(self.layers) - 1)) else 'output'
            })
          count += 1
    return output

  def fpSum(self, l, n, activations):
    sumValue = self.layers[l][n].bias * 1
    for (i, neuron) in enumerate(self.layers[l - 1]):
      sumValue += neuron.weights[n] * activations[-2][i]
    return sumValue

  def backwardPass(self, forwardPass, expected):
    output = []
    for (l, layer) in enumerate(reversed(forwardPass)):
      output.insert(0, [])
      for (n, passValue) in enumerate(reversed(layer)):
        if l == 0:
          output[0].append(self.bpInput(passValue, expected[n]))
        else:
          output[0].insert(0, self.bpHidden(l, n, passValue, forwardPass, output))
    return output

  def bpInput(self, passValue, expected):
    return (expected - passValue['activation']) * self.activate(passValue['activation'], True)

  def bpHidden(self, l, n, passValue, forwardPass, output):
    actualL = (len(self.layers) - 1) - l
    actualN = (len(self.layers[actualL]) - 1) - n
    neuron = self.layers[actualL][actualN]
    passValue = forwardPass[actualL][actualN]
    sumValue = 0
    derivative = self.activate(passValue['activation'], True)
    for (w, weight) in enumerate(neuron.weights):
      sumValue += (weight * output[1][w])
    sumValue = sumValue * derivative
    return sumValue

  def updateWeights(self, forwardPass, deltas):
    newPreviousWeights = []
    weightDeltas = []
    for (l, layer) in enumerate(self.layers):
      for (n, neuron) in enumerate(layer):
        for (w, weight) in enumerate(neuron.weights):
          if l != (len(self.layers) - 1):
            weightChange = (self.step * deltas[l + 1][w] * forwardPass[l][n]['activation'])
            newWeight = neuron.weights[w] + weightChange
            # if forwardPass[l][n]['i'] <= len(weightDeltas):
            #   newWeight += (self.momentum * weightDeltas[forwardPass[l][n]['i'] - 1])
            # newPreviousWeights.append(newWeight)
            # if len(newPreviousWeights) <= len(self.previousWeights):
            #   weightDeltas.append(self.previousWeights[len(newPreviousWeights) - 1] - newWeight)
            neuron.weights[w] = newWeight
        neuron.bias += (self.step * deltas[l][n] * forwardPass[l][n]['activation'])
    self.previousWeights = newPreviousWeights

  def solve(self, inputs):
    return [passValue['activation'] for passValue in self.forwardPass(inputs)[-1]]

  def train(self, inputs, expected):
    forwards = self.forwardPass(inputs)
    backwards = self.backwardPass(forwards, expected)
    self.updateWeights(forwards, backwards)
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
