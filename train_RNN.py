#!/usr/bin/python

#####
# Description: Main train script for GRU network
# Author: Heike Adel
# Date: 2016
#####

import sys
import os
import time
import numpy
import theano
import theano.tensor as T
import cPickle

from blocks.roles import add_role, WEIGHT, BIAS
from blocks.filter import VariableFilter
from blocks.model import Model, ComputationGraph
from blocks.algorithms import GradientDescent, Scale, AdaGrad, AdaDelta, Momentum, StepClipping, CompositeRule
from blocks.extensions import FinishAfter
from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from layers import BidirectionalEncoderSigmoid, LogisticRegression, HiddenLayer, AttentionLayer
from utilities_uncertainty import readConfig, readWordvectors
from blocks_fuel_classes import F1Extension, ModelResults, ShuffledExampleSchemeBatch

if len(sys.argv) != 2:
  print "please pass the config file as parameters"
  exit(0)

time1 = time.time()

configfile = sys.argv[1]
config = readConfig(configfile)

for c in sorted(config.keys()):
  print c + "\t" + config[c]

datafile = config["file"]
vectorsize = int(config["vectorsize"])
wordvectorfile = config["wordvectors"]
networkfile = config["net"]
learning_rate = float(config["lrate"])
batch_size = int(config["batchsize"])
contextsize = int(config["contextsize"])
combinationMethod = "concatenation"
if "combinationMethod" in config:
  combinationMethod = config["combinationMethod"]
attentionMethod = "internalOnH"
if "attentionMethod" in config:
  attentionMethod = config["attentionMethod"]
kattention = 1
if "Kmax" in attentionMethod:
  if "kattention" in config:
    kattention = int(config["kattention"])
  else:
    print "no k for attention sequence defined. Setting to default 1"
useHiddenLayer = True
if "noHidden" in config:
  useHiddenLayer = False
  print "using no hidden layer"
else:
  hiddenunits = int(config["hidden"])
rnnH = 50
if "rnnH" in config:
  rnnH = int(config["rnnH"])

iterationSeed = -1
if "iterationSeed" in config:
  iterationSeed = int(config["iterationSeed"])

dt = theano.config.floatX
wordvectors, vectorsize = readWordvectors(wordvectorfile)
representationsize = vectorsize + 1

if "external" in attentionMethod:
  cuevectorfile = "cues_wiki"
  if "cuevectorfile" in config:
    cuevectorfile = config["cuevectorfile"]

  cf = open(cuevectorfile, 'rb')
  cueVectorMatrix = cPickle.load(cf) # numpy matrix of shape (number of vectors, dimension of vectors)
  cf.close()

  cue_vectors = theano.shared(numpy.array(cueVectorMatrix, dtype = dt), borrow=True)

  cueHiddenunits = 100
  if "cueHidden" in config:
    cueHiddenunits = int(config["cueHidden"])

time2 = time.time()
print "time for reading data: " + str(time2 - time1)

# train network
curSeed = 23455
if "seed" in config:
  curSeed = int(config["seed"])
rng = numpy.random.RandomState(curSeed)
seed = rng.get_state()[1][0]
print "seed: " + str(seed)

time1 = time.time()

######## FUEL #################
# Define "data_stream"
# The names here (e.g. 'name1') need to match the names of the variables which
#  are the roots of the computational graph for the cost.

if "gpu" in theano.config.device:
  train_set = H5PYDataset(datafile, which_sets = ('train',))
  test_set = H5PYDataset(datafile, which_sets = ('test',))
else:
  train_set = H5PYDataset(datafile, which_sets = ('train',), load_in_memory=True)
  test_set = H5PYDataset(datafile, which_sets = ('test',), load_in_memory=True)

numSamplesTest = test_set.num_examples
numTrainingBatches = train_set.num_examples / batch_size
numTestBatches = numSamplesTest

print "got " + str(numSamplesTest) + " test examples"

print "batch size for training", batch_size
print "batch size for testing", 1

if iterationSeed != -1:
  data_stream = DataStream(train_set, iteration_scheme = ShuffledExampleSchemeBatch(numTrainingBatches * batch_size, batch_size, iterationSeed))
else:
  data_stream = DataStream(train_set, iteration_scheme = ShuffledExampleSchemeBatch(numTrainingBatches * batch_size, batch_size))
data_stream_test = DataStream(test_set, iteration_scheme=SequentialScheme(
                                       numTestBatches, 1))

################################

# allocate symbolic variables for the data
x = T.matrix('x')   # the data is presented as rasterized images
y = T.imatrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
length = T.imatrix('length')

time2 = time.time()
print "time for preparing data structures: " + str(time2 - time1)

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'
time1 = time.time()

batch_size_var = x.shape[0] # different for train and test

length2 = length.reshape((batch_size_var, ))
length2 = T.minimum(length2, contextsize - 1)

# prepare input for GRU layer
x2 = x.reshape((batch_size_var, representationsize, contextsize))
layer0_input = x2.dimshuffle(1,0,2)
#layer0_input = layer0_input[:,:,:contextsize]

# create mask for GRU layer
layer0_mask = T.zeros((batch_size_var, contextsize))
layer0_mask_tmp, _ = theano.scan(fn = lambda b,m: T.set_subtensor(m[b,:length2[b]+1], 1), outputs_info = layer0_mask, sequences = T.arange(batch_size_var))
layer0_mask = layer0_mask_tmp[-1]

# indices for last hidden state
ii = length2 - 1
jj = T.arange(batch_size_var)

y = y.reshape((batch_size_var, ))

layer0 = BidirectionalEncoderSigmoid(representationsize, rnnH)

layer0representations = layer0.apply(layer0_input, layer0_mask)
layer0outputsize = 2 * rnnH
if combinationMethod != "onlyAtt":
  layer0output = layer0representations[ii,jj,:] # take last hidden state as sentence representation
  layer0flattened = layer0output.flatten(2).reshape((batch_size_var, layer0outputsize))

if "internalOnH" in attentionMethod:
  layer1input = layer0representations.dimshuffle(1,2,0)
  layer1 = AttentionLayer(rng, thisInput=layer1input, batchsize=batch_size_var, dim1=layer0outputsize, dim2 = contextsize, method = attentionMethod, k = kattention)
  layer1outputsize = 2 * rnnH
elif "internalOnW" in attentionMethod:
  layer1input = T.tanh(x2)
  layer1 = AttentionLayer(rng, thisInput=layer1input, batchsize=batch_size_var, dim1=representationsize, dim2 = contextsize, method = attentionMethod, k = kattention)
  layer1outputsize = representationsize
elif "externalOnH" in attentionMethod:
  layer1input = layer0representations.dimshuffle(1,2,0)
  layer1 = AttentionLayer(rng, thisInput=layer1input, batchsize=batch_size_var, dim1=layer0outputsize, dim2 = contextsize, method = attentionMethod, embeddings = cue_vectors, n_in = vectorsize, hiddenunits=cueHiddenunits, k = kattention)
  layer1outputsize = 2 * rnnH
elif "externalOnW" in attentionMethod:
  layer1input = T.tanh(x2)
  layer1 = AttentionLayer(rng, thisInput=layer1input, batchsize=batch_size_var, dim1=representationsize, dim2 = contextsize, method = attentionMethod, embeddings = cue_vectors, n_in = vectorsize, hiddenunits=cueHiddenunits, k = kattention)
  layer1outputsize = representationsize
else:
  print "ERROR: unknown attentionMethod - skipping attention"
  combinationMethod = "noAtt"

if "Kmax" in attentionMethod and "Sequence" in attentionMethod:
  layer1outputsize = layer1outputsize * kattention

layer1flattened = layer1.output.flatten(2).reshape((batch_size_var, layer1outputsize))

if combinationMethod == "onlyAtt":
  layer2_inputSize = layer1outputsize
  layer2_input = layer1flattened
elif combinationMethod == "noAtt":
  layer2_inputSize = layer0outputsize
  layer2_input = layer0flattened
else: # concatenation
  layer2_inputSize = layer0outputsize + layer1outputsize
  layer2_input = T.concatenate([layer0flattened, layer1flattened], axis = 1)

if useHiddenLayer: 
  # construct a fully-connected sigmoidal layer
  layer2 = HiddenLayer(rng, input=layer2_input, n_in=layer2_inputSize,
                         n_out=hiddenunits, activation=T.tanh)

  # classify the values of the fully-connected sigmoidal layer
  layer3 = LogisticRegression(input=layer2.output, n_in=hiddenunits, n_out=2)
else:
  # classify the values of the fully-connected sigmoidal layer
  layer3 = LogisticRegression(input=layer2_input, n_in=layer2_inputSize, n_out=2)

# create a list of all model non-bricks parameters
paramList = [layer3.params]
if useHiddenLayer:
  paramList.append(layer2.params)
if combinationMethod != "noAtt":
  paramList.append(layer1.params)
# params from layer0 already have the blocks role
params = []
for p in paramList:
  for i, p_part in enumerate(p):
    if i == 0:
      add_role(p_part, WEIGHT)
    else:
      add_role(p_part, BIAS)
  params += p

cost = layer3.negative_log_likelihood(y)
# regularization is added below

cost.name = 'cost'

lr = T.scalar('lr', dt)

time2 = time.time()
print "time for building the model: " + str(time2 - time1)

############ BLOCKS ########################
# wrap everything in Blocks objects and run!
######### training ##################

cg = ComputationGraph(cost)
# add regularization
reg2 = 0.0
weightList = VariableFilter(roles=[WEIGHT])(cg.variables)
for p in weightList:
  reg2 += T.sum(p ** 2)
cost += 0.00001 * reg2

n_epochs = 15
if "n_epochs" in config:
  n_epochs = int(config["n_epochs"])

params = cg.parameters
model = Model([cost])
print "model parameters:"
print model.get_parameter_dict()

if "adagrad" in config:
  print "using adagrad"
  thisRule=AdaGrad(learning_rate=learning_rate)
elif "adadelta" in config:
  print "using adadelta"
  thisRule=AdaDelta()
elif "momentum" in config:
  print "using momentum"
  mWeight = float(config["momentum"])
  thisRule=Momentum(learning_rate=learning_rate, momentum=mWeight)
else:
  print "using traditional SGD"
  thisRule=Scale(learning_rate=learning_rate)

if "gradientClipping" in config:
  threshold = float(config["gradientClipping"])
  print "using gradient clipping with threshold ", threshold
  thisRule=CompositeRule([StepClipping(threshold), thisRule])

#step_rule=CompositeRule([StepClipping(config['step_clipping']),
#                                 eval(config['step_rule'])()])

algorithm = GradientDescent(cost=cost,
                            parameters=params,
                            step_rule=thisRule,
                            on_unused_sources='warn')
extensions = []
if "saveModel" in config:
  print "saving model after training"
  extensions.append(Checkpoint(path=networkfile, after_n_epochs=n_epochs))
extensions.append(FinishAfter(after_n_epochs=n_epochs))
extensions.append(F1Extension(layer3 = layer3, y = y, model = model, data_stream = data_stream_test, num_samples = numSamplesTest, batch_size = 1, every_n_epochs=1))
extensions.append(ModelResults(layer3 = layer3, y = y, model = model, data_stream = data_stream_test, num_samples = numSamplesTest, batch_size = 1, after_n_epochs=n_epochs))
my_loop = MainLoop(model=model,
                   data_stream=data_stream,
                   algorithm=algorithm,
                   extensions=extensions)
time1 = time.time()
my_loop.run()
time2 = time.time()
print "time for training network: " + str(time2 - time1)

