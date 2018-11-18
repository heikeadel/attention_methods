#!/usr/bin/python

#####
# Description: Script for preprocessing data and storing data stream for RNN
# Author: Heike Adel
# Date: 2016
#####

import sys
import os
import time
import numpy
from utilities_uncertainty import readConfig, readWordvectors, openTokenizedFileWithoutPadding
import theano

if len(sys.argv) != 2:
  print "please pass the config file as parameters"
  exit(0)

time1 = time.time()

configfile = sys.argv[1]
config = readConfig(configfile)

trainfile = config["train"]
print "trainfile\t" + trainfile
testfile = config["test"]
print "testfile\t" + testfile

# format of train and test file:
# text file where each row consists of one instance
# format of instance: two columns, separated by :: as delimiter as follows:
# label (0 or 1 with 1 meaning uncertainty) :: tokenized sentence
# two example instances:
# 1 :: We infer that the murine and human GATA-3 proteins play a central and highly conserved role in vertebrate T-cell-specific transcriptional regulation .
# 0 :: CD3-epsilon gene expression is confined to the T cell lineage .

wordvectorfile = config["wordvectors"]
print "wordvectorfile\t" + wordvectorfile

# format of wordvector file: output format of word2vec

filename = config["file"]
print "filename for storing data\t" + filename
contextsize = int(config["contextsize"])
print "contextsize\t" + str(contextsize)

time1 = time.time()

# reading word vectors
wordvectors, vectorsize = readWordvectors(wordvectorfile)

representationsize = vectorsize + 1 # additional dimension for capitalization feature; not used in this work

def getMatrixForContext(context, vectorsize, contextsize):
  global representationsize
  global config
  matrix = numpy.zeros(shape = (representationsize, contextsize))
  i = 0
  nextIndex = 0
  numOOV = 0

  while i < len(context):
    word = context[i]
    nextIndex = 0
    # current word
    if word != "<empty>":
      if not word in wordvectors:
        word = "<unk>"
        numOOV += 1
        #word = "UNKNOWN"
      curVector = wordvectors[word]
      for j in range(0, vectorsize):
        if j > len(curVector):
          print "ERROR: mismatch in word vector lengths: " + str(len(curVector)) + " vs " + vectorsize
          exit()
        elem = float(curVector[j])
        matrix[j + nextIndex, i] = elem
    nextIndex += vectorsize
    i += 1

  return matrix

# read train file
inputListTrain, resultVectorTrain = openTokenizedFileWithoutPadding(trainfile, contextsize)
numSamples = len(inputListTrain)
if numSamples == 0:
  print "no train examples for this slot: no training possible"
  exit()

inputMatrixTrain = numpy.empty(shape = (numSamples, representationsize * contextsize))
inputLengthsTrain = numpy.empty(shape = (numSamples, 1))
for sample in range(0, numSamples):
  context = inputListTrain[sample]
  lastIndex = context.index("<empty>")
  if lastIndex == -1:
    lastIndex = contextsize - 1
  else:
    lastIndex -= 1 # index should point to last word, not to first <empty>
  matrix = getMatrixForContext(context, vectorsize, contextsize)
  matrix = numpy.reshape(matrix, representationsize * contextsize)
  inputMatrixTrain[sample,:] = matrix
  inputLengthsTrain[sample,:] = lastIndex

# read test file
inputListTest, resultVectorTest = openTokenizedFileWithoutPadding(testfile, contextsize)
numSamplesTest = len(inputListTest)
if numSamplesTest == 0:
  print "no test examples for this slot: no training possible"
  exit()

inputMatrixTest = numpy.empty(shape = (numSamplesTest, representationsize * contextsize))
inputLengthsTest = numpy.empty(shape = (numSamplesTest, 1))
for sample in range(0, numSamplesTest):
  context = inputListTest[sample]
  lastIndex = context.index("<empty>")
  if lastIndex == -1:
    lastIndex = contextsize - 1
  else:
    lastIndex -= 1 # index should point to last word, not to first <empty>
  matrix = getMatrixForContext(context, vectorsize, contextsize)
  matrix = numpy.reshape(matrix, representationsize * contextsize)
  inputMatrixTest[sample,:] = matrix
  inputLengthsTest[sample,:] = lastIndex

if "dev" in config:
  devfile = config["dev"]
  # read dev file
  inputListDev, resultVectorDev = openTokenizedFileWithoutPadding(devfile, contextsize)
  numSamplesDev = len(inputListDev)
  if numSamplesDev == 0:
    print "no dev examples for this slot: no training possible"
    exit()

  inputMatrixDev = numpy.empty(shape = (numSamplesDev, representationsize * contextsize))
  inputLengthsDev = numpy.empty(shape = (numSamplesDev, 1))
  for sample in range(0, numSamplesDev):
    context = inputListDev[sample]
    lastIndex = context.index("<empty>")
    if lastIndex == -1:
      lastIndex = contextsize - 1
    else:
      lastIndex -= 1 # index should point to last word, not to first <empty>
    matrix = getMatrixForContext(context, vectorsize, contextsize)
    matrix = numpy.reshape(matrix, representationsize * contextsize)
    inputMatrixDev[sample,:] = matrix
    inputLengthsDev[sample,:] = lastIndex

time2 = time.time()
print "time for reading data: " + str(time2 - time1)

time1 = time.time()
dt = theano.config.floatX

################ FUEL #################
import h5py
from fuel.datasets.hdf5 import H5PYDataset

f = h5py.File(filename, mode='w')

if "dev" in config:
  features = f.create_dataset('x', (numSamples + numSamplesDev + numSamplesTest, representationsize * contextsize), dtype = dt, compression='gzip')
  targets = f.create_dataset('y', (numSamples + numSamplesDev + numSamplesTest, 1), dtype=numpy.dtype(numpy.int32), compression='gzip')
  lengths = f.create_dataset('length', (numSamples + numSamplesDev + numSamplesTest, 1), dtype=numpy.dtype(numpy.int32), compression='gzip')
  features[...] = numpy.vstack([inputMatrixTrain, inputMatrixDev, inputMatrixTest])
  targets[...] = numpy.array(resultVectorTrain + resultVectorDev + resultVectorTest).reshape(numSamples + numSamplesDev + numSamplesTest, 1)
  lengths[...] = numpy.vstack([inputLengthsTrain, inputLengthsDev, inputLengthTest])
  features.dims[0].label='batch'
  features.dims[1].label='feature'
  targets.dims[0].label='batch'
  targets.dims[1].label='label'
  lengths.dims[0].label='batch'
  lengths.dims[1].label='length'
  split_dict = {'train' : {'x':(0,numSamples), 'y':(0,numSamples), 'length':(0,numSamples)}, 
                'dev' : {'x':(numSamples,numSamples+numSamplesDev), 'y':(numSamples,numSamples+numSamplesDev), 'length':(numSamples,numSamples+numSamplesDev)}, 
                'test' : {'x':(numSamples+numSamplesDev,numSamples+numSamplesDev+numSamplesTest), 'y':(numSamples+numSamplesDev,numSamples+numSamplesDev+numSamplesTest), 'length':(numSamples+numSamplesDev, numSamples+numSamplesDev+numSamplesTest)}}
  f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
else:
  features = f.create_dataset('x', (numSamples + numSamplesTest, representationsize * contextsize), dtype = dt, compression='gzip')
  targets = f.create_dataset('y', (numSamples + numSamplesTest, 1), dtype=numpy.dtype(numpy.int32), compression='gzip')
  lengths = f.create_dataset('length', (numSamples + numSamplesTest, 1), dtype=numpy.dtype(numpy.int32), compression='gzip')
  features[...] = numpy.vstack([inputMatrixTrain, inputMatrixTest])
  targets[...] = numpy.array(resultVectorTrain + resultVectorTest).reshape(numSamples + numSamplesTest, 1)
  lengths[...] = numpy.vstack([inputLengthsTrain, inputLengthsTest])
  features.dims[0].label='batch'
  features.dims[1].label='feature'
  targets.dims[0].label='batch'
  targets.dims[1].label='label'
  lengths.dims[0].label='batch'
  lengths.dims[1].label='length'
  split_dict = {'train' : {'x':(0,numSamples), 'y':(0,numSamples), 'length':(0,numSamples)}, 'test' : {'x':(numSamples,numSamples+numSamplesTest), 'y':(numSamples,numSamples+numSamplesTest), 'length':(numSamples,numSamples+numSamplesTest)}}
  f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()

time2 = time.time()
print "time for creating dataset: " + str(time2 - time1)
