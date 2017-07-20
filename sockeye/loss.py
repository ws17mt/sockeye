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
Functions to generate loss symbols for sequence-to-sequence models.
"""
from typing import Tuple

import mxnet as mx

import sockeye.constants as C
import sockeye.model


def get_loss(config: sockeye.model.ModelConfig) -> 'Loss':
    """
    Returns Loss instance given loss_name.

    :param config: Model configuration.
    """
    if config.loss == C.CROSS_ENTROPY:
        return CrossEntropyLoss(config.normalize_loss)
    elif config.loss == C.SMOOTHED_CROSS_ENTROPY:
        return SmoothedCrossEntropyLoss(config.smoothed_cross_entropy_alpha, config.vocab_target_size,
                                        config.normalize_loss)
    elif config.loss == C.GAN_LOSS:
        return GANLoss(config.normalize_loss)
    else:
        raise ValueError("unknown loss name")


class Loss:
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.
        
        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: Loss and softmax output symbols.
        """
        raise NotImplementedError()

# NOTE: moved mas_labels_after_EOS to style_training.py since it is used in discriminating, not loss

class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """

    def __init__(self, normalize: bool = False):
        self._normalize = normalize

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: Loss and softmax output symbols.
        """
        if self._normalize:
            normalization = "valid"
        else:
            normalization = "null"
        return mx.sym.SoftmaxOutput(data=logits,
                                    label=labels,
                                    ignore_label=C.PAD_ID,
                                    use_ignore=True,
                                    normalization=normalization,
                                    name=C.SOFTMAX_NAME)


def _normalize(loss: mx.sym.Symbol, labels: mx.sym.Symbol):
    """
    Normalize loss by the number of non-PAD tokens.

    :param loss: A loss value for each label.
    :param labels: A label for each loss entry (potentially containing PAD tokens).
    :return: The normalized loss.
    """
    return mx.sym.broadcast_div(loss, mx.sym.sum(labels != C.PAD_ID))


class SmoothedCrossEntropyLoss(Loss):
    """
    Computes a smoothed cross-entropy loss. Smoothing is defined by alpha which indicates the
    amount of probability mass subtracted from the true label probability (1-alpha).
    Alpha is then uniformly distributed across other labels.

    :param alpha: Smoothing value.
    :param vocab_size: Size of the target vocabulary.
    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """

    def __init__(self, alpha: float, vocab_size: int, normalize: bool = False):
        assert alpha >= 0, "alpha must be >= 0"
        self._alpha = alpha
        self._vocab_size = vocab_size
        self._normalize = normalize

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> Tuple[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: Loss and softmax output symbols.
        """
        probs = mx.sym.softmax(data=logits)

        on_value = 1.0 - self._alpha
        off_value = self._alpha / (self._vocab_size - 1.0)
        cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                       depth=self._vocab_size,
                                       on_value=on_value,
                                       off_value=off_value)

        # zero out pad symbols (0)
        cross_entropy = mx.sym.where(labels, cross_entropy, mx.sym.zeros((0, self._vocab_size)))

        # compute cross_entropy
        cross_entropy *= - mx.sym.log(data=probs + 1e-10)
        cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

        if self._normalize:
            cross_entropy = _normalize(cross_entropy, labels)

        cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.SMOOTHED_CROSS_ENTROPY)
        probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
        return cross_entropy, probs

class GANLoss(Loss):
    """
    Computes a loss for G, D_e, and D_f for GANs.
    
    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """
    
    def __init__(self, normalize: bool = False):
        self._normalize = normalize
    
    def get_loss_G(self, logits_e: mx.sym.Symbol, logits_f: mx.sym.Symbol,
                   labels_e: mx.sym.Symbol, labels_f: mx.sym.Symbol,
                   loss_De: mx.sym.Symbol, loss_Df: mx.sym.Symbol, lambd: float) -> mx.sym.Symbol:
        """
        Returns generator loss for GAN given logits and discriminator loss.
        
        :param logits_e: Logits from e autoencode minibatch. Shape: (batch_size * e_sequence_len, e_vocab_size).
        :param logits_f: Logits from f autoencode minibatch. Shape: (batch_size * f_sequence_len, f_vocab_size).
        :param labels_e: Labels from e autoencode minibatch. Shape: (batch_size * e_sequence_len, ).
        :param labels_f: Labels from f autoencode minibatch. Shape: (batch_size * f_sequence_len, ).
        :param loss_De: Loss symbol for e discriminator.
        :param loss_Df: Loss symbol for f discriminator.
        :param lambd: Weight for the discriminator losses.
        """
        if self._normalize:
            normalization = "valid"
        else:
            normalization = "null"
        # TODO need to subtract lambda * (loss_De + loss_Df)
        return mx.sym.SoftmaxOutput(data=mx.sym.concat(logits_e, logits_f, dim=0),
                                   label=mx.sym.concat(labels_e, labels_f, dim=0),
                                   ignore_label=C.PAD_ID, use_ignore=True,
                                   normalization=normalization)
        # - lambd * (loss_De + loss_Df)
        # TODO name
        # TODO is the shape of this right for adding them? 
        # TODO maybe recalculate loss_d instead of passing it in? so we can block the gradients?
    
    def get_loss_D(self, logits_ae: mx.sym.Symbol, logits_trans: mx.sym.Symbol,
                   labels_ae: mx.sym.Symbol, labels_trans: mx.sym.Symbol, name: str) -> mx.sym.Symbol:
        """
        Returns discriminator loss and softmax output symbols for GAN given logits.
        
        :param logits_ae: Logits from autoencoding minibatch. Shape: (batch_size, 2).
        :param logits_trans: Logits from translation minibatch. Shape: (batch_size, 2).
        :param labels_ae: Labels from autoencoding minibatch. Shape: (batch_size,).
        :param labels_trans: Labels from translation minibatch. Shape: (batch_size,).
        :param name: Desired name for softmax symbol (since this is used for both discriminators).
        :return: Loss and softmax output symbols.
        """
        if self._normalize:
            normalization = "valid"
        else:
            normalization = "null"
        # TODO this is not quite right since we need to feed the autoencoder logits into the discriminator..
        return mx.sym.SoftmaxOutput(data=mx.sym.concat(logits_ae, logits_trans, dim=0),
                                    label=mx.sym.concat(labels_ae, labels_trans, dim=0),
                                    ignore_label=C.PAD_ID,
                                    use_ignore=True,
                                    normalization=normalization,
                                    name=name)
