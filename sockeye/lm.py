"""
Language model for multi-task learning, designed to share parameters
with sockeye components
"""

import logging
import mxnet as mx

import sockeye.rnn
import sockeye.encoder
import sockeye.decoder
import sockeye.constants as C

logger = logging.getLogger(__name__)


def get_lm_from_encoder(encoder,
                        dir="l2r") -> 'SharedLanguageModel':
    return None


def get_lm_l2r_from_deccoder(decoder) -> 'SharedLanguageModel':
    return None


def get_lm_from_options(
        num_embed,
        vocab_size,
        dropout,
        rnn_num_layers,
        rnn_num_hidden,
        rnn_cell_type,
        rnn_residual_connections,
        rnn_forget_bias
) -> 'SharedLanguageModel':

    return SharedLanguageModel(
        num_embed,
        vocab_size,
        dropout,
        rnn_num_layers,
        rnn_num_hidden,
        rnn_cell_type,
        rnn_residual_connections,
        rnn_forget_bias
        )


class SharedLanguageModel:
    """
    Language model that shares parameters with a
    sockeye encoder or decoder
    """
    def __init__(self,
                 num_embed,
                 vocab_size,
                 dropout,
                 rnn_num_layers,
                 rnn_num_hidden,
                 rnn_cell_type,
                 rnn_residual_connections,
                 rnn_forget_bias,
                 embedding_params=None,
                 rnn_params=None,
                 cls_w_params=None,
                 cls_b_params=None):
        # Collect settings
        self.num_embed = num_embed
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.rnn_num_layers = rnn_num_layers
        self.rnn_num_hidden = rnn_num_hidden
        self.rnn_cell_type = rnn_cell_type
        self.rnn_residual_connections = rnn_residual_connections
        self.rnn_forget_bias = rnn_forget_bias

        # Build parameters - embedding
        self.embedding = sockeye.encoder.Embedding(
            num_embed=self.num_embed,
            vocab_size=self.vocab_size,
            prefix=C.SOURCE_EMBEDDING_PREFIX,  # TODO: revisit these prefix decisions
            dropout=self.dropout,
            params=embedding_params
        )

        # Build parameters - rnn
        self.rnn = sockeye.rnn.get_stacked_rnn(
            cell_type=self.rnn_cell_type,
            num_hidden=self.rnn_num_hidden,
            num_layers=self.rnn_num_layers,
            dropout=self.dropout,
            prefix=C.STACKEDRNN_PREFIX,  # TODO: revisit these prefix decisions
            residual=self.rnn_residual_connections,
            forget_bias=self.rnn_forget_bias,
            params=rnn_params
        )

        # Build paramers - output layer
        if cls_w_params is not None:
            self.cls_w = cls_w_params
        else:
            self.cls_w = mx.sym.Variable("cls_weight")  # TODO: revisit prefix
        if cls_b_params is not None:
            self.cls_b = cls_b_params
        else:
            self.cls_b = mx.sym.Variable("cls_bias")  # TODO: revisit prefix


def encode(self, data, data_length, seq_len):

    data = self.embedding.encode(data, data_length, seq_len)

    self.rnn.reset()
    outputs, states = self.rnn.unroll(seq_len, inputs=data, merge_outputs=True)

    pred = mx.sym.reshape(outputs, shape=(-1, self.rnn_num_hidden))
    pred = mx.sym.FullyConnected(data=pred,
                                 num_hidden=self.vocab_size,
                                 weight=self.cls_w,
                                 bias=self.cls_b,
                                 name=C.LOGITS_NAME)
    return pred
