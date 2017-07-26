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
                      prefix: str) -> 'Discriminator':
    """
    Returns a MLPDiscriminator with the following properties.
    
    :param act: Activation function.
    :param num_hidden: Number of hidden units.
    :param num_layers: Number of hidden layers.
    :param dropout: Dropout probability.
    :param prefix: Symbol prefix for MLP.
    :returns: Discriminator instance.
    """
    return MLPDiscriminator(act, num_hidden, num_layers, dropout, prefix)


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


class MLPDiscriminator(Discriminator):
    """
    Class to generate the discriminator part of the computation graph in GAN models.
    Currently can only use an MLP; later will add CNN or RNN.
    
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
                 prefix: str) -> None:
        self.act = act
        self.prefix = prefix
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout # TODO add dropout (currently don't use this at all)
        # initialize weights and biases (note no weights or biases on the GRL)
        # input layer
        self.in_w = mx.sym.Variable('%sin_weight' % self.prefix)
        self.in_b = mx.sym.Variable('%sin_bias' % self.prefix)
        # hidden layers
        self.weight_dict = {}
        self.bias_dict = {}
        for layer in range(self.num_layers):
            self.weight_dict[layer] = mx.sym.Variable('%slayer%d_weight' % (self.prefix, layer))
            self.bias_dict[layer] = mx.sym.Variable('%slayer%d_bias' % (self.prefix, layer))
        # output layer
        self.out_w = mx.sym.Variable('%sout_weight' % prefix)
        self.out_b = mx.sym.Variable('%sout_bias' % prefix)

    def discriminate(self,
                     data: mx.sym.Symbol,
                     target_seq_len: int,
                     target_vocab_size: int,
                     target_length: mx.sym.Symbol,
                     loss_lambda: float) -> mx.sym.Symbol:
        """
        Given a sequence of decoder hidden states, decide whether they are from the real or generated data.
        
        :param data: Input data. Shape: (target_seq_len*batch_size, target_vocab_size).
        :param target_seq_len: Maximum length of target sequences.
        :param target_vocab_size: Target vocabulary size.
        :param target_length: Lengths of target sequences. Shape: (batch_size,).
        :param loss_lambda: Weight parameter for discriminators.
        :return: Logits of discriminator decision for target sequence. Shape: (batch_size, 2).
        """
        # reshape the data so it's max len x batch size x vocab size
        target = mx.sym.reshape(data=data, shape=(-1, target_seq_len, target_vocab_size))
        target = mx.sym.swapaxes(data=target, dim1=0, dim2=1)
        decoder_last_state = mx.sym.SequenceLast(data=target, sequence_length=target_length,
                                                 use_sequence_length=True)
        # add a gradient reversal layer before the discriminators
        reverse_grad = mx.symbol.Custom(data=decoder_last_state, op_type='gradientreversallayer',
                                        loss_lambda=loss_lambda)
        # input layer
        logits = mx.sym.FullyConnected(data=reverse_grad, num_hidden=self.num_hidden, weight=self.in_w, bias=self.in_b)
        logits = mx.sym.Activation(data=logits, act_type=self.act)
        # hidden layers
        for layer in range(self.num_layers):
            logits = mx.sym.FullyConnected(data=logits, num_hidden=self.num_hidden, weight=self.weight_dict[layer],
                                           bias=self.bias_dict[layer])
            logits = mx.sym.Activation(data=logits, act_type=self.act)
        # output layer
        logits = mx.sym.FullyConnected(data=logits, num_hidden=2, weight=self.out_w, bias=self.out_b)
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
