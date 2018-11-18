#!/usr/bin/python

#####
# Description: Main train script for CNN network
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

from blocks.roles import add_role, WEIGHT
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale, AdaGrad, AdaDelta, Momentum
from blocks.extensions import FinishAfter
from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from layers import LeNetConvPoolLayer, LogisticRegression, HiddenLayer, AttentionLayer
from utilities_uncertainty import readConfig, readWordvectors, openTokenizedFile
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
filtersize = [1,int(config["filtersize"])]
nkerns = [int(config["nkerns"])]
pool = [1, int(config["kmax"])]
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

iterationSeed = -1
if "iterationSeed" in config:
  iterationSeed = int(config["iterationSeed"])

dt = theano.config.floatX

wordvectors, vectorsize = readWordvectors(wordvectorfile)

representationsize = vectorsize + 1

# some sanity checks
if contextsize < filtersize[1]:
  print "INFO: setting filtersize to ", contextsize
  filtersize[1] = contextsize

sizeAfterConv = contextsize - filtersize[1] + 1

sizeAfterPooling = -1
if sizeAfterConv < pool[1]:
  print "INFO: setting poolsize to ", sizeAfterConv
  pool[1] = sizeAfterConv
sizeAfterPooling = pool[1]

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
ishape = [representationsize, contextsize]  # this is the size of context matrizes

time2 = time.time()
print "time for preparing data structures: " + str(time2 - time1)

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'
time1 = time.time()
# Reshape input matrix to a 4D tensor, compatible with our LeNetConvPoolLayer
batch_size_var = x.shape[0] # different for training and test set

layer0_input = x.reshape((batch_size_var, 1, ishape[0], ishape[1]))

y = y.reshape((batch_size_var, ))

# Construct the first convolutional pooling layer:
filter_shape = (nkerns[0], 1, representationsize, filtersize[1])
poolsize=(pool[0], pool[1])

fan_in = numpy.prod(filter_shape[1:])
fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
              numpy.prod(poolsize))

W_bound = numpy.sqrt(6. / (fan_in + fan_out))
# the convolution weight matrix
convW = theano.shared(numpy.asarray(
           rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
           dtype=theano.config.floatX), name='conv_W',
                               borrow=True)

# the bias is a 1D tensor -- one bias per output feature map
b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
convB = theano.shared(value=b_values, name='conv_b', borrow=True)

layer0 = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0_input,
            filter_shape=filter_shape, poolsize=poolsize)

layer0flattened = layer0.output.flatten(2).reshape((batch_size_var, nkerns[0] * sizeAfterPooling))
layer0outputsize = nkerns[0] * sizeAfterPooling

if "internalOnH" in attentionMethod:
  layer1 = AttentionLayer(rng, thisInput=layer0.conv_out_tanh, batchsize=batch_size_var, dim1=nkerns[0], dim2 = sizeAfterConv, method = attentionMethod, k = kattention)
  layer1outputsize = nkerns[0]
elif "internalOnW" in attentionMethod:
  layer1 = AttentionLayer(rng, thisInput=x.reshape((batch_size_var, ishape[0], ishape[1])), batchsize=batch_size_var, dim1=ishape[0], dim2 = ishape[1], method = attentionMethod, k = kattention)
  layer1outputsize = ishape[0]
elif "externalOnH" in attentionMethod:
  layer1 = AttentionLayer(rng, thisInput=layer0.conv_out_tanh, batchsize=batch_size_var, dim1=nkerns[0], dim2 = sizeAfterConv, method = attentionMethod, embeddings = cue_vectors, n_in = vectorsize, hiddenunits=cueHiddenunits, k = kattention)
  layer1outputsize = nkerns[0]
elif "externalOnW" in attentionMethod:
  layer1 = AttentionLayer(rng, thisInput=x.reshape((batch_size_var, ishape[0], ishape[1])), batchsize=batch_size_var, dim1=ishape[0], dim2 = ishape[1], method = attentionMethod, embeddings = cue_vectors, n_in = vectorsize, hiddenunits=cueHiddenunits, k = kattention)
  layer1outputsize = ishape[0]
else:
  print "ERROR: unknown attentionMethod - skipping attention"
  combinationMethod = "noAtt"

if "Kmax" in attentionMethod and "Sequence" in attentionMethod:
  layer1outputsize = layer1outputsize * kattention

layer1flattened = layer1.output.flatten(2)
layer1flattened = layer1flattened.reshape((batch_size_var, layer1outputsize))

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

# create a list of all model parameters to be fit by gradient descent
paramList = [layer3.params]
if useHiddenLayer:
  paramList.append(layer2.params)
if combinationMethod != "noAtt":
  paramList.append(layer1.params)
paramList.append(layer0.params)

params = []
for p in paramList:
  for p_part in p:
    add_role(p_part, WEIGHT) # make parameters visible to Blocks
  params += p

cost = layer3.negative_log_likelihood(y)

# L2 regularization
reg2 = 0
for p in paramList:
  reg2 += T.sum(p[0] ** 2)
cost += 0.00001 * reg2

cost.name = 'cost'

lr = T.scalar('lr', dt)

time2 = time.time()
print "time for building the model: " + str(time2 - time1)

############ BLOCKS ########################
# wrap everything in Blocks objects and run!
######### training ##################
n_epochs = 15
if "n_epochs" in config:
  n_epochs = int(config["n_epochs"])
print "number of training epochs: ", n_epochs

model = Model([cost])
print "model parameters:"
print model.get_parameter_dict()
if "adagrad" in config:
  print "using adagrad"
  algorithm = GradientDescent(cost=cost,
                              parameters=params,
                              step_rule=AdaGrad(learning_rate=learning_rate),
                              on_unused_sources='warn')
elif "adadelta" in config:
  print "using adadelta"
  algorithm = GradientDescent(cost=cost,
                              parameters=params,
                              step_rule=AdaDelta(),
                              on_unused_sources='warn')
elif "momentum" in config:
  print "using momentum"
  mWeight = float(config["momentum"])
  algorithm = GradientDescent(cost=cost,
                              parameters=params,
                              step_rule=Momentum(learning_rate=learning_rate,
                                                 momentum=mWeight),
                              on_unused_sources='warn')
else:
  print "using traditional SGD"
  algorithm = GradientDescent(cost=cost,
                            parameters=params,
                            step_rule=Scale(learning_rate=learning_rate),
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
