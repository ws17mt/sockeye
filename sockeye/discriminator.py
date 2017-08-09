# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
# 
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
GAN Discriminators
"""
from typing import Callable, List, NamedTuple, Tuple
from typing import Optional

import mxnet as mx

import sockeye.constants as C
from sockeye.grl import *
import sockeye.utils

def get_discriminator(act: str,
                      num_hidden: int,
                      num_layers: int,
                      dropout: float,
                      loss_lambda: float,
                      batch_norm: bool,
                      prefix: str,
                      cell_type: str,
                      disc_type: str) -> 'Discriminator':
    """
    Returns a Discriminator with the following properties.
    
    :param act: Activation function.
    :param num_hidden: Number of hidden units.
    :param num_layers: Number of hidden layers.
    :param dropout: Dropout probability.
    :param batch_norm: Whether to use batch normalization.
    :param prefix: Symbol prefix for MLP.
    :param cell_type: RNN cell type (GRU or LSTM).
    :param disc_type: Discriminator type (rnn or mlp)
    :returns: Discriminator instance.
    """
    if disc_type == C.MLP_DISC_TYPE:
        return MLPDiscriminator(act, num_hidden, num_layers, dropout,
                                loss_lambda, batch_norm, prefix)
    elif disc_type == C.RNN_DISC_TYPE:
        return RNNDiscriminator(num_hidden, num_layers, dropout, loss_lambda,
                                batch_norm, cell_type, prefix)
    else:
        raise NotImplementedError()

class Discriminator:
    """
    Generic discriminator interface.
    """
    
    def get_num_hidden(self) -> int:
        """
        Returns the representation size of this decoder.

        :raises: NotImplementedError
        """
        raise NotImplementedError()

class RNNDiscriminator(Discriminator):
    """
    Class to generate an RNN descriminator for the computation graph in GAN models.

    :param num_hidden: Number of hidden units in the discriminator.
    :param num_layers: Number of layers in the discriminator.
    :param dropout: Dropout probability on discriminator outputs.
    :param loss_lambda: Weight for the discriminator loss.
    :param batch_norm: Whether to perform batch normalization.
    :param cell_type: RNN cell type for discriminator (GRU or LSTM).
    :param prefix: Discriminator symbol prefix.
    """

    def __init__(self,
                 num_hidden: int,
                 num_layers: int,
                 dropout: float,
                 loss_lambda: float,
                 batch_norm: bool,
                 cell_type: str,
                 prefix: str):
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.loss_lambda = loss_lambda
        # TODO add batch normalization!
        self.batch_norm = batch_norm
        self.cell_type = cell_type
        # initialize weights and biases (input and output layer only)
        self.in_w = mx.sym.Variable('%sin_weight' % prefix)
        self.in_b = mx.sym.Variable('%sin_bias' % prefix)
        self.out_w = mx.sym.Variable('%sout_weight' % prefix)
        self.out_b = mx.sym.Variable('%sout_bias' % prefix)
        # create the actual RNN
        self.rnn = sockeye.rnn.get_stacked_rnn(self.cell_type, self.num_hidden, self.num_layers,
                                              self.dropout, self.prefix)
        # TODO do we need all those arguments in our case??

    def discriminate(self,
                     data: mx.sym.Symbol,
                     target_seq_len: int,
                     target_vocab_size: int,
                     target_length: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Given a sequence of decoder hidden states, decide whether they are from the real or generated data.

        :param data: Input data (hidden states from decoder). Shape: (batch_size, tar_seq_len, rnn_num_hidden).
        :param target_seq_len: Maximum length of target sequences.
        :param target_vocab_size: Target vocabulary size.
        :param target_length: Lengths of target sequences. Shape: (batch_size,).
        :return: Logits of discriminator decision for target sequence. Shape: (batch_size, 2).
        """
        # TODO: check __init__, add batch norm, get rid of params we don't actually use (if any)
        # start with a gradient reversal layer
        reverse_grad = mx.symbol.Custom(data=data, op_type='gradientreversallayer',
                                        loss_lambda=self.loss_lambda)
        # apply tanh to the decoder hidden states
        target = mx.sym.tanh(reverse_grad)
        # unroll the RNN to get the outputs of shape (batch_size, max_len, num_hidden)
        outputs, _ = self.rnn.unroll(target_seq_len, inputs=target, merge_outputs=True, layout='NTC')
        # to classify, apply a fully connected layer and a sigmoid activation to SequenceLast
        # for SequenceLast, need (max_len, batch_size, num_hidden)
        logits = mx.sym.swapaxes(data=outputs, dim1=0, dim2=1)
        logits = mx.sym.SequenceLast(data=logits, sequence_length=target_length, use_sequence_length=True)
        logits = mx.sym.FullyConnected(data=logits, num_hidden=2, weight=self.out_w, bias=self.out_b)
        logits = mx.sym.sigmoid(data=logits)
        return logits


class MLPDiscriminator(Discriminator):
    """
    Class to generate an MLP discriminator for the computation graph in GAN models.
    
    :param act: Activation function.
    :param num_hidden: Number of hidden units in the discriminator.
    :param num_layers: Number of hidden layers in the discriminator.
    :param dropout: Dropout probability on discriminator outputs.
    :param prefix: Discriminator symbol prefix.
    """
    
    def __init__(self,
                 act: str,
                 num_hidden: int,
                 num_layers: int,
                 dropout: float,
                 loss_lambda: float,
                 batch_norm: bool,
                 prefix: str) -> None:
        self.act = act
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout # TODO add dropout (currently don't use this at all)
        self.loss_lambda = loss_lambda
        self.batch_norm = batch_norm
        # initialize weights and biases (note no weights or biases on the GRL)
        # input layer
        self.in_w = mx.sym.Variable('%sin_weight' % self.prefix)
        self.in_b = mx.sym.Variable('%sin_bias' % self.prefix)
        if self.batch_norm:
            self.in_gamma = mx.sym.Variable('%sin_gamma' % self.prefix)
            self.in_beta = mx.sym.Variable('%sin_beta' % self.prefix)
            self.in_mm = mx.sym.Variable('%sin_mm' % self.prefix)
            self.in_mv = mx.sym.Variable('%sin_mv' % self.prefix)
        # hidden layers
        self.weight_dict = {}
        self.bias_dict = {}
        if self.batch_norm:
            self.gamma_dict = {}
            self.beta_dict = {}
            self.mm_dict = {}
            self.mv_dict = {}
        for layer in range(self.num_layers):
            self.weight_dict[layer] = mx.sym.Variable('%slayer%d_weight' % (self.prefix, layer))
            self.bias_dict[layer] = mx.sym.Variable('%slayer%d_bias' % (self.prefix, layer))
            if self.batch_norm:
                self.gamma_dict[layer] = mx.sym.Variable('%slayer%d_gamma' % (self.prefix, layer))
                self.beta_dict[layer] = mx.sym.Variable('%slayer%d_beta' % (self.prefix, layer))
                self.mm_dict[layer] = mx.sym.Variable('%slayer%d_mm' % (self.prefix, layer))
                self.mv_dict[layer] = mx.sym.Variable('%slayer%d_mv' % (self.prefix, layer))
        # output layer
        self.out_w = mx.sym.Variable('%sout_weight' % prefix)
        self.out_b = mx.sym.Variable('%sout_bias' % prefix)
        if self.batch_norm:
            self.out_gamma = mx.sym.Variable('%sout_gamma' % self.prefix)
            self.out_beta = mx.sym.Variable('%sout_beta' % self.prefix)
            self.out_mm = mx.sym.Variable('%sout_mm' % self.prefix)
            self.out_mv = mx.sym.Variable('%sout_mv' % self.prefix)

    def discriminate(self,
                     data: mx.sym.Symbol,
                     target_seq_len: int,
                     target_vocab_size: int,
                     target_length: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Given a sequence of decoder hidden states, decide whether they are from the real or generated data.
        
        :param data: Input data (hidden states from decoder). Shape: (batch_size, tar_seq_len, rnn_num_hidden).
        :param target_seq_len: Maximum length of target sequences.
        :param target_vocab_size: Target vocabulary size.
        :param target_length: Lengths of target sequences. Shape: (batch_size,).
        :return: Logits of discriminator decision for target sequence. Shape: (batch_size, 2).
        """
        # reshape the data so it's (max_len, batch_size, rnn_num_hidden)
        target = mx.sym.swapaxes(data=data, dim1=0, dim2=1)
        decoder_last_state = mx.sym.SequenceLast(data=target, sequence_length=target_length,
                                                 use_sequence_length=True)
        # add a gradient reversal layer before the discriminators
        reverse_grad = mx.symbol.Custom(data=decoder_last_state, op_type='gradientreversallayer',
                                        loss_lambda=self.loss_lambda)
        # Apply tanh to the input of the GAN: https://github.com/soumith/ganhacks
        reverse_grad = mx.sym.tanh(reverse_grad)
        # input layer
        logits = mx.sym.FullyConnected(data=reverse_grad, num_hidden=self.num_hidden,
                                       weight=self.in_w, bias=self.in_b)
        if self.batch_norm:
            logits = mx.sym.BatchNorm(data=logits, gamma=self.in_gamma, beta=self.in_beta,
                                      moving_mean=self.in_mm, moving_var=self.in_mv)
        logits = mx.sym.Activation(data=logits, act_type=self.act)
        # hidden layers
        for layer in range(self.num_layers):
            logits = mx.sym.FullyConnected(data=logits, num_hidden=self.num_hidden, weight=self.weight_dict[layer],
                                           bias=self.bias_dict[layer])
            if self.batch_norm:
                logits = mx.sym.BatchNorm(data=logits, gamma=self.gamma_dict[layer], beta=self.beta_dict[layer],
                                          moving_mean=self.mm_dict[layer], moving_var=self.mv_dict[layer])
            logits = mx.sym.Activation(data=logits, act_type=self.act)
        # output layer
        logits = mx.sym.FullyConnected(data=logits, num_hidden=2, weight=self.out_w, bias=self.out_b)
        if self.batch_norm:
            logits = mx.sym.BatchNorm(data=logits, gamma=self.out_gamma, beta=self.out_beta,
                                      moving_mean=self.out_mm, moving_var=self.out_mv)
        logits = mx.sym.sigmoid(data=logits)
        return logits
    
    def get_num_hidden(self) -> int:
        """
        Return the number of hidden units of this discriminator.
        """ 
        return self.num_hidden

    def get_activation(self) -> str:
        """
        Return the activation of this discriminator.
        """
        return self.act
