#!/usr/bin/python

#####
# Description: Blocks Extensions and Fuel classes
# Author: Heike Adel
# Date: 2016
#####


from blocks.extensions import SimpleExtension
import time
import theano
import numpy
from fuel.schemes import BatchScheme
from picklable_itertools import imap
from picklable_itertools.extras import partition_all

################# FUEL #############################################
class ShuffledExampleSchemeBatch(BatchScheme):
  def __init__(self, examples, batch_size, seed = 987654):
    super(ShuffledExampleSchemeBatch, self).__init__(examples, batch_size)
    self.batch_size = batch_size
    numpy.random.seed(seed)

  def get_request_iterator(self):
    indices = list(self.indices)
    # shuffle indices
    indicesShuffled = []
    permutation = numpy.random.permutation(len(indices))
    return imap(list, partition_all(self.batch_size, permutation))

################# BLOCKS ###########################################
def getF1(tp, numHypo, numRef):
    precision = 1
    recall = 0
    f1 = 0
    if numHypo > 0:
      precision = 1.0 * tp / numHypo
    if numRef > 0:
      recall = 1.0 * tp / numRef
    if precision + recall > 0:
      f1 = 2 * precision * recall / (precision + recall)
    print str(time.ctime()) + "\tP = " + str(precision) + ", R = " + str(recall) + ", F1 = " + str(f1)
    return f1

class PrintStatus(SimpleExtension):
  def __init__(self, **kwargs):
    super(PrintStatus, self).__init__(**kwargs)
  def do(self, which_callback, *args):
    print self.main_loop.log.status['iterations_done']

class ModelResults(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(ModelResults, self).__init__(**kwargs)
    y_hat = layer3.results()
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    print "ref\thypo\tconfidence for 1"
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      probs = self.predict(*inp)
      y_curr = src2vals['y']
      for j in range(self.batch_size):
        hypo = probs[0][j]
        ref = y_curr[j][0]
        conf = probs[2][j][1]
        print str(ref) + "\t" + str(hypo) + "\t" + str(conf)

class F1Extension(SimpleExtension):
  def __init__(self, layer3, y, model, data_stream, num_samples, batch_size, **kwargs):
    super(F1Extension, self).__init__(**kwargs)
    y_hat = layer3.results()
    self.theinputs = [inp for inp in model.inputs if inp.name != 'y'] # name of input variables in graph
    self.predict = theano.function(self.theinputs, y_hat)
    self.data_stream = data_stream
    self.num_samples = num_samples
    self.batch_size = batch_size

  def do(self, which_callback, *args):
    num_batches = self.num_samples / self.batch_size
    epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)
    tp = 0
    numHypo = 0
    numRef = 0
    total = 0
    tp_tn = 0
    for i in range(num_batches):
      src2vals = epoch_iter.next()
      inp = [src2vals[src.name] for src in self.theinputs]
      probs = self.predict(*inp)
      y_curr = src2vals['y']

      for j in range(self.batch_size):
        index = i * self.batch_size + j
        hypo = probs[0][j]
        ref = y_curr[j][0]
        if hypo == 1:
          numHypo += 1
          if hypo == ref:
            tp += 1
        if ref == 1:
          numRef += 1
        if hypo == ref:
          tp_tn += 1
        total += 1
    print "accurracy: ", 100.0 * tp_tn / total
    getF1(tp, numHypo, numRef)

