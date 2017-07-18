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
import sockeye.mlp
import sockeye.utils


def get_discriminator(act: str,
                      data_length: int,
                      num_hidden: int,
                      num_layers: int,
                      dropout=0.,
                      prefix: str) -> 'Discriminator':
    """
    Returns a MLPDiscriminator with the following properties.
    
    :param act: Activation function.
    :param data_length: Maximum length of the input data.
    :param num_hidden: Number of hidden units.
    :param num_layers: Number of hidden layers.
    :param dropout: Dropout probability.
    :param prefix: Symbol prefix for MLP.
    :returns: Discriminator instance.
    """
    return MLPDiscriminator(act, data_length, num_hidden, num_layers, dropout, prefix)


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
    :param data_length: Maximum input data length.
    :param num_hidden: Number of hidden units in the discriminator.
    :param num_layers: Number of hidden layers in the discriminator.
    :param dropout: Dropout probability on discriminator outputs.
    :param prefix: Discriminator symbol prefix.
    """
    
    def __init__(self,
                 data_length,
                 act='relu',
                 num_hidden: int,
                 num_layers: int,
                 dropout=0.,
                 prefix: str) -> None:
    self.act = act
    self.prefix = prefix
    self.data_length = data_length
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.dropout = dropout # TODO add dropout (currently don't use this at all)
    
    # Discriminator MLP
    self.mlp = sockeye.mlp.get_mlp(act, data_length, num_hidden, num_layers, dropout, prefix)
    
    def discriminate(self, 
                     data: mx.sym.Symbol,
                     target_seq_len: int,
                     target_vocab_size: int,
                     target_length: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Given a sequence of decoder hidden states, decide whether they are from the real or generated data.
        
        :param data: Input data. Shape: (target_seq_len*batch_size, target_vocab_size).
        :param target_seq_len: Maximum length of target sequences.
        :param target_vocab_size: Target vocabulary size.
        :param target_length: Lengths of target sequences. Shape: (batch_size, target_seq_len).
        :return: Logits of discriminator decision for target sequence. Shape: (batch_size, 2).
        """
        # TODO make this more flexible..
        # reshape the data so it's max len x batch size x vocab size
        target = mx.sym.reshape(data=data, shape=(target_seq_len, -1, target_vocab_size))
        decoder_last_state = mx.sym.SequenceLast(data=target, sequence_length=target_length,
                                                 use_sequence_length=True)
        # input layer
        logits = mx.sym.FullyConnected(data=data, num_hidden=self.num_hidden)
        logits = mx.sym.Activation(data=logits, act_type=self.act)
        # hidden layers
        for layer in range(num_layers):
            logits = mx.sym.FullyConnected(data=logits, num_hidden=self.num_hidden)
            logits = mx.sym.Activation(data=logits, act_type=self.act)
        # output layer
        logits = mx.sym.FullyConnected(data=logits, num_hidden=2)
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
