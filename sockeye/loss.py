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


def get_loss(loss_type, config: sockeye.model.ModelConfig) -> 'Loss':
    """
    Returns Loss instance given loss_name.

    :param loss_type: which type of loss is to be used
    :param config: Model configuration.
    """
    if loss_type == C.CROSS_ENTROPY:
        return CrossEntropyLoss(config.normalize_loss)
    elif loss_type == C.SMOOTHED_CROSS_ENTROPY:
        return SmoothedCrossEntropyLoss(config.smoothed_cross_entropy_alpha, config.vocab_target_size,
                                        config.normalize_loss)
    elif loss_type == C.GAN_LOSS:
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

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol, name) -> mx.sym.Symbol:
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
                                    name=name)


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

    def get_loss(self, e_logits_autoencoder: mx.sym.Symbol, f_logits_autoencoder: mx.sym.Symbol,
                 e_labels: mx.sym.Symbol, f_labels: mx.sym.Symbol,
                 e_D_autoencoder: mx.sym.Symbol, e_D_transfer: mx.sym.Symbol,
                 e_labels_autoencoder: mx.sym.Symbol, e_labels_transfer: mx.sym.Symbol,
                 f_D_autoencoder: mx.sym.Symbol, f_D_transfer: mx.sym.Symbol,
                 f_labels_autoencoder: mx.sym.Symbol, f_labels_transfer: mx.sym.Symbol,
                 g_loss_weight: float) -> mx.sym.Symbol:
        """
        Returns generator and discriminator loss for GAN assuming gradient reversal layer.

        :param e_logits_autoencoder: Logits from e autoencode step. Shape: (e_batch_size * e_seq_len, e_vocab_size).
        :param f_logits_autonecoder: Logits from f autoencoder step. Shape: (f_batch_size * f_seq_len, f_vocab_size).
        :param e_labels: Labels from e autoencode step. Shape: (batch_size * e_seq_len).
        :param f_labels: Labels from f autoencode step. Shape: (batch_size * f_seq_len).
        :param e_D_autoencoder: Logits from discriminating autoencoded e. Shape: (e_batch_size, 2).
        :param e_D_transfer: Logits from discriminating transferred e (from f). Shape: (f_batch_size, 2).
        :param e_labels_autoencoder: Labels from discriminating autoencoded e. Shape: (e_batch_size, ).
        :param e_labels_transfer: Labels from discriminating transferred e. Shape(f_batch_size, ).
        :param f_D_autoencoder: Logits from discriminating autoencoded f. Shape: (f_batch_size, 2).
        :param f_D_transfer: Logits from discriminating transferred f (from e). Shape: (e_batch_size, 2).
        :param f_labels_autoencoder: Labels from discriminating autoencoded f. Shape: (f_batch_size, ).
        :param f_labels_transfer: Labels from discriminating transferred f. Shape(e_batch_size, ).
        :param g_loss_weight: weight for loss_G.
        """
        if self._normalize:
            normalization = 'valid'
        else:
            normalization = 'null'

        # get reconstruction and discriminator losses
        # TODO why does loss_G have the wrong name??
        loss_G = mx.sym.SoftmaxOutput(data=mx.sym.concat(e_logits_autoencoder, f_logits_autoencoder, dim=0),
                                      label=mx.sym.concat(e_labels, f_labels, dim=0),
                                      ignore_label=C.PAD_ID, use_ignore=True, normalization=normalization,
                                      name=C.GAN_LOSS + '_g')
        # TODO should e_loss_D and f_loss_D have ignore_label?
        e_loss_D = mx.sym.SoftmaxOutput(data=mx.sym.concat(e_D_autoencoder, e_D_transfer, dim=0),
                                        label=mx.sym.concat(e_labels_autoencoder, e_labels_transfer, dim=0),
                                        normalization=normalization, name=C.GAN_LOSS + '_e')
        f_loss_D = mx.sym.SoftmaxOutput(data=mx.sym.concat(f_D_autoencoder, f_D_transfer, dim=0),
                                        label=mx.sym.concat(f_labels_autoencoder, f_labels_transfer, dim=0),
                                        normalization=normalization, name=C.GAN_LOSS + '_f')

        # now combine them
        loss_D = mx.sym.concat(e_loss_D, f_loss_D, dim=0, name=C.GAN_LOSS + '_d')
        # weight loss_G
        loss_G = g_loss_weight * loss_G
        # NOTE: the GRL reverses the gradients and adds the lambda, so we add the three losses here
        return loss_G, loss_D
