#!/usr/bin/python

#####
# Description: Implementation of network layers
# Author: Heike Adel
# Date: 2016
# Code for GRU, HiddenLayer and LogisticRegression: based on Theano/Blocks tutorials
#####

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN
from toolz import merge
from picklable_itertools.extras import equizip
from blocks.bricks import (Tanh, Linear, Initializable, Logistic)
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional, recurrent, LSTM
from blocks.roles import add_role, WEIGHT
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant

class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""
    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [T.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]

class BidirectionalEncoderSigmoid(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoderSigmoid, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        curSeed = 1791095845
        self.rng = numpy.random.RandomState(curSeed)

        self.bidir = BidirectionalWMT15(
            GatedRecurrentWithZerosAtMask(activation=Logistic(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        #self.children = [self.lookup, self.bidir,
        self.children = [self.bidir,
                         self.fwd_fork, self.back_fork]

        self._push_allocation_config() # maybe not necessary? (maybe only necessary for decoder)

        print "RNN seed: " + str(self.rng.get_state()[1][0])
        # initialization of parameters
        self.weights_init = IsotropicGaussian()
        self.biases_init = Constant(0)
        self.push_initialization_config()
        self.bidir.prototype.weights_init = Orthogonal()
        self.initialize()

    def _push_allocation_config(self):
        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = source_sentence

        representation = self.bidir.apply(
            # Conversion to embedding representation here.
            # TODO: Less than the current number of dimensions should be totally fine.
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        self.representation = representation
        return representation


class GatedRecurrentWithZerosAtMask(GatedRecurrent):
  
      @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
      def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states)
        return next_states


#############################################################################################

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=""):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if name != "":
          prefix = name
        else:
          prefix = "mlp_"

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

#########################################################################################

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None, rng = None, dropout_rate = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        if W == None:
          # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
          self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='softmax_W', borrow=True)
        else:
          self.W = W

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='softmax_b', borrow=True)
        else:
          self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def results(self):
      return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]

####################################################################################

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def preparePooling(self, conv_out):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling
      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      neighborsArgSorted = neighborsArgSorted
      return neighborsForPooling, neighborsArgSorted

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling, neighborsArgSorted = self.preparePooling(conv_out)
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape = None, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        self.conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        k = poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

###################################################################################

class AttentionLayer:

  def getExternalAttentionWeights(self, cueVectors):
    # create an MLP with one hidden layer to compute a score between each input token and cue vector
    # get hidden layer vectors for cue vectors
    print "external attention with " + str(self.hiddenunits) + " hiddenunits"
    hiddenCue = T.dot(cueVectors, self.Wcue)
    # repeat this vector for each input word
    hiddenCueRepeated = T.repeat(hiddenCue.reshape((1,cueVectors.shape[0],self.hiddenunits)), self.batchsize * self.dim2, axis = 0)
    hiddenCueRepeated2 = hiddenCueRepeated.reshape((self.batchsize, self.dim2, cueVectors.shape[0], self.hiddenunits)).dimshuffle(0,2,1,3)
    # get hidden layer vectors for input vectors
    hiddenInp = T.dot(T.transpose(self.Winp), self.input).dimshuffle((1,2,0))
    hiddenInpRepeated = T.repeat(hiddenInp, cueVectors.shape[0], axis = 0).reshape((self.batchsize, cueVectors.shape[0], self.dim2, self.hiddenunits))
    hiddenTmp = hiddenCueRepeated2 + hiddenInpRepeated
    # hidden.shape = batchsize x numberOfCues x contextsize (dim2) x hiddenunits
    hidden = hiddenTmp.reshape((self.batchsize, cueVectors.shape[0], self.dim2, self.hiddenunits))
    # now: for each hidden units vector in hidden (last dimension): calculate one final score (=> number of final scores: batchsize x numberOfCues x contextsize)
    hiddenOut = T.tanh(hidden)
    alphas = T.dot(hiddenOut, self.W2).flatten(3)
    self.weights = alphas.sum(axis = 1)
    return self.weights

  def externalAttention(self, cueVectors):
    weights = self.getExternalAttentionWeights(cueVectors)
    weightsNormed = self.normalizeZeroWeights(weights)
    weightsSoftmax = self.doSoftmax(weightsNormed)
    self.attentionWeightsOwn = weightsSoftmax.dimshuffle((0,'x',1))
    self.weightedInput = self.attentionWeightsOwn * self.input
    return self.attentionWeightsOwn, self.weightedInput

  def externalAttentionKmaxSequence(self, cueVectors, k):
    self.attentionWeightsOwn, self.weightedInput = self.externalAttention(cueVectors)
    return self.getKmaxSequence(self.attentionWeightsOwn, self.weightedInput, k)

  def normalizeZeroWeights(self, weights):
    zeroWeight = weights[0][-1] # assumption: always the same weight for zero embeddings
    minWeight = T.min(weights)
    offset = - minWeight + 0.00000001
    weights2 = weights + offset # all weights are > 0 now
    zeroWeight2 = zeroWeight + offset
    weights3 = T.switch(T.eq(weights2, zeroWeight2), 0.0, weights2)
    return weights3

  def externalAttentionKmaxAverage(self, cueVectors, k):
    self.attentionWeightsOwn, self.weightedInput = self.externalAttention(cueVectors)
    return self.getKmaxAverage(self.attentionWeightsOwn, self.weightedInput, k)

  def getKmaxIndices(self, weights, k):
    maxIndices = T.argsort(weights, axis = 2)[:,:,-k:]
    maxIndicesSorted = T.sort(maxIndices, axis=2)
    ii = T.repeat(T.arange(self.batchsize), k)
    jj = maxIndicesSorted.flatten()
    return ii, jj

  def getKmaxAverage(self, weights, weightedInput, k):
    # right now it's exactly the same as getKmaxSequence, the averaging is done later
    return self.getKmaxSequence(weights, weightedInput, k)

  def getKmaxSequence(self, weights, weightedInput, k):
    # get k max values as a sequence
    ii, jj = self.getKmaxIndices(weights, k)
    kmaxSequence_tmp = weightedInput[ii,:,jj]
    new_shape = T.cast(T.join(0, T.as_tensor([self.batchsize]), T.as_tensor([k]), T.as_tensor([self.dim1])), 'int32')
    self.kmaxSequence = T.reshape(kmaxSequence_tmp, new_shape, ndim=3)
    self.kmaxSequence = self.kmaxSequence.dimshuffle(0,2,1)
    return self.kmaxSequence

  def getKmaxPooledSequence(self, input3D, k):
    neighborsForPooling, neighborsArgSorted = self.preparePooling(input3D)
    kNeighborsArg = neighborsArgSorted[:,-k:]
    maxIndicesSorted = T.sort(kNeighborsArg, axis=1)
    ii = T.repeat(T.arange(self.batchsize * self.dim1), k)
    jj = maxIndicesSorted.flatten()
    kmaxSequence_tmp = neighborsForPooling[ii,jj]
    new_shape = T.cast(T.join(0, T.as_tensor([self.batchsize]), T.as_tensor([self.dim1]), T.as_tensor([k])), 'int32')
    self.kmaxSequence = T.reshape(kmaxSequence_tmp, new_shape, ndim=3)
    return self.kmaxSequence

  def externalAttentionKmaxPooledSequence(self, cueVectors, k):
    _, self.weightedInput = self.externalAttention(cueVectors)
    return self.getKmaxPooledSequence(self.weightedInput, k)

  def preparePooling(self, input3D):
    neighborsForPooling = TSN.images2neibs(ten4=input3D.reshape((1, input3D.shape[0], input3D.shape[1], input3D.shape[2])), neib_shape=(1,input3D.shape[2]), mode='ignore_borders')
    neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
    return neighborsForPooling, neighborsArgSorted

  def internalAttention(self):
    weightAllDot = self.getInternalAttentionWeights()
    weightAllDotNormed = self.normalizeZeroWeights(weightAllDot)
    self.attentionWeightsOwn = self.doSoftmax(weightAllDotNormed)
    self.attentionWeightsOwn = self.attentionWeightsOwn.dimshuffle((0,'x',1))
    self.weightedInput = self.attentionWeightsOwn * self.input
    return self.attentionWeightsOwn, self.weightedInput

  def internalAttentionKmaxSequence(self, k):
    self.attentionWeightsOwn, self.weightedInput = self.internalAttention()
    return self.getKmaxSequence(self.attentionWeightsOwn, self.weightedInput, k)

  def internalAttentionKmaxPooledSequence(self, k):
    _, self.weightedInput = self.internalAttention()
    return self.getKmaxPooledSequence(self.weightedInput, k)

  def doSoftmax(self, weights):
    expWeights = T.exp(weights)
    expWeights = T.switch(T.eq(expWeights, 1.0), 0.0, expWeights)
    sumWeights = T.sum(expWeights, axis=1)
    result = expWeights / sumWeights.dimshuffle(0,'x')
    return result

  def getInternalAttentionWeights(self):
    weightsReshaped = self.W.reshape((1,self.dim1)).dimshuffle(('x', 0, 1))
    result = T.dot(weightsReshaped, self.input).reshape((self.batchsize, self.dim2))
    return result

  def internalAttentionKmaxAverage(self, k):
    self.attentionWeightsOwn, self.weightedInput = self.internalAttention()
    return self.getKmaxAverage(self.attentionWeightsOwn, self.weightedInput, k)

  def lengthNormalization(self, vector):
    norm = T.sqrt((vector ** 2).sum(axis = 1))
    return vector / norm.dimshuffle(0,'x')

  def __init__(self, rng, thisInput, batchsize, dim1, dim2, method="internalOnH", embeddings=None, n_in=0, hiddenunits=100, k = 1, name = None):
    self.input = thisInput.flatten(3) # dimensions: batchsize * dim1 * contextsize
    self.batchsize = batchsize
    self.dim1 = dim1
    self.dim2 = dim2
    self.n_in = n_in

    if method == "internalOnH" or method == "internalOnW":
      W_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim1)),
               high=numpy.sqrt(6. / (self.dim1)),
               size=(self.dim1)), dtype=theano.config.floatX)
      Wname = 'internal_W'
      if name != None:
        Wname += '_' + name
      self.W = theano.shared(value=W_values, name=Wname, borrow=True)
      _, weightedInputG = self.internalAttention()
      outputTmp = T.sum(weightedInputG, axis = 2).flatten(2)
      self.output = self.lengthNormalization(outputTmp)
      self.params = [self.W]
    elif method == "internalOnH_KmaxAverage" or method == "internalOnW_KmaxAverage":
      W_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim1)),
               high=numpy.sqrt(6. / (self.dim1)),
               size=(self.dim1)), dtype=theano.config.floatX)
      Wname = 'internal_W'
      if name != None:
        Wname += '_' + name
      self.W = theano.shared(value=W_values, name=Wname, borrow=True)
      outputTmp = T.sum(self.internalAttentionKmaxAverage(k), axis = 2).flatten(2)
      self.output = self.lengthNormalization(outputTmp)
      self.params = [self.W]
    elif method == "internalOnH_KmaxSequence" or method == "internalOnW_KmaxAverage":
      W_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim1)),
               high=numpy.sqrt(6. / (self.dim1)),
               size=(self.dim1)), dtype=theano.config.floatX)
      Wname = 'internal_W'
      if name != None:
        Wname += '_' + name
      self.W = theano.shared(value=W_values, name=Wname, borrow=True)
      self.output = self.internalAttentionKmaxSequence(k)
      self.params = [self.W]
    elif method == "internalOnH_KmaxPooledSequence" or method == "internalOnW_KmaxPooledSequence":
      W_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim1)),
               high=numpy.sqrt(6. / (self.dim1)),
               size=(self.dim1)), dtype=theano.config.floatX)
      Wname = 'internal_W'
      if name != None:
        Wname += '_' + name
      self.W = theano.shared(value=W_values, name=Wname, borrow=True)
      self.output = self.internalAttentionKmaxPooledSequence(k)
      self.params = [self.W]
    elif method == "externalOnW" or method == "externalOnH" or method == "externalOnW_KmaxAverage" or method == "externalOnH_KmaxAverage":
      self.hiddenunits = hiddenunits
      Wcue_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim1 + self.hiddenunits)),
               high=numpy.sqrt(6. / (self.dim1 + self.hiddenunits)),
               size=(n_in, self.hiddenunits)), dtype=theano.config.floatX)
      Wname = 'Wcue'
      if name != None:
        Wname += '_' + name
      self.Wcue = theano.shared(value=Wcue_values, name=Wname, borrow=True)
      Winp_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.dim2 + self.hiddenunits)),
               high=numpy.sqrt(6. / (self.dim2 + self.hiddenunits)),
               size=(self.dim1, self.hiddenunits)), dtype=theano.config.floatX)
      Wname = 'Winp'
      if name != None:
        Wname += '_' + name
      self.Winp = theano.shared(value=Winp_values, name=Wname, borrow=True)
      W2_values = numpy.asarray(rng.uniform(
               low=-numpy.sqrt(6. / (self.hiddenunits)),
               high=numpy.sqrt(6. / (self.hiddenunits)),
               size=(self.hiddenunits, 1)), dtype=theano.config.floatX)
      Wname = 'W2'
      if name != None:
        Wname += '_' + name
      self.W2 = theano.shared(value=W2_values, name=Wname, borrow=True)
      if method == "externalOnW" or method == "externalOnH":
        _, outputTmp = self.externalAttention(embeddings)
      else:
        outputTmp = self.externalAttentionKmaxAverage(embeddings, k)
      outputTmp = outputTmp.sum(axis = 2).flatten(2)
      self.output = self.lengthNormalization(outputTmp)
      self.params = [self.Wcue, self.Winp, self.W2]
    elif method == "externalOnW_KmaxSequence" or method == "externalOnH_KmaxSequence" or method == "externalOnW_KmaxPooledSequence" or method == "externalOnH_KmaxPooledSequence":
      self.hiddenunits = hiddenunits
      Wcue_values = numpy.asarray(rng.uniform(
                 low=-numpy.sqrt(6. / (self.dim1 + self.hiddenunits)),
                 high=numpy.sqrt(6. / (self.dim1 + self.hiddenunits)),
                 size=(n_in, self.hiddenunits)), dtype=theano.config.floatX)
      Wname = 'Wcue'
      if name != None:
        Wname += '_' + name
      self.Wcue = theano.shared(value=Wcue_values, name=Wname, borrow=True)
      Winp_values = numpy.asarray(rng.uniform(
                 low=-numpy.sqrt(6. / (self.dim2 + self.hiddenunits)),
                 high=numpy.sqrt(6. / (self.dim2 + self.hiddenunits)),
                 size=(self.dim1, self.hiddenunits)), dtype=theano.config.floatX)
      Wname = 'Winp'
      if name != None:
        Wname += '_' + name
      self.Winp = theano.shared(value=Winp_values, name=Wname, borrow=True)
      W2_values = numpy.asarray(rng.uniform(
                 low=-numpy.sqrt(6. / (self.hiddenunits)),
                 high=numpy.sqrt(6. / (self.hiddenunits)),
                 size=(self.hiddenunits, 1)), dtype=theano.config.floatX)
      Wname = 'W2'
      if name != None:
        Wname += '_' + name
      self.W2 = theano.shared(value=W2_values, name=Wname, borrow=True)
      if "KmaxSequence" in method:
        self.output = self.externalAttentionKmaxSequence(embeddings, k)
      elif "KmaxPooledSequence" in method:
        self.output = self.externalAttentionKmaxPooledSequence(embeddings, k)
      self.params = [self.Wcue, self.Winp, self.W2]
    else:
      print "ERROR: unknown attention method", method

