"""
Language model for multi-task learning, designed to share parameters
with sockeye components
"""

import logging
import mxnet as mx

import sockeye.model
import sockeye.rnn
import sockeye.encoder
import sockeye.decoder
import sockeye.constants as C

logger = logging.getLogger(__name__)


def get_lm_from_encoder(config: sockeye.model.ModelConfig,
                        encoder,
                        fused,
                        rnn_forget_bias) -> 'SharedLanguageModel':
    """
    Language model that shares weights with an encoder
    """
    assert config.lm_pretrain_layers_source > 0
    assert encoder.embed.embed_weight is not None
    assert encoder.lm_pre_rnn.rnn.params is not None
    # Also tie source LM output weights if we are doing tying
    cls_w = None
    if config.weight_tying:
        cls_w = encoder.embed.embed_weight
    return SharedLanguageModel(
        num_embed=config.num_embed_source,
        vocab_size=config.vocab_source_size,
        dropout=config.dropout,
        rnn_num_layers=config.lm_pretrain_layers_source,
        rnn_num_hidden=config.rnn_num_hidden,
        rnn_cell_type=config.rnn_cell_type,
        rnn_residual_connections=config.rnn_residual_connections,
        rnn_forget_bias=rnn_forget_bias,
        embedding_prefix=C.SOURCE_EMBEDDING_PREFIX,
        rnn_prefix=C.STACKEDRNN_PREFIX+C.LM_SOURCE_PREFIX,
        embedding_params=encoder.embed.embed_weight,
        rnn_params=encoder.lm_pre_rnn.rnn.params,
        cls_w_params=cls_w
        )


def get_lm_from_decoder(config,
                        decoder,
                        rnn_forget_bias) -> 'SharedLanguageModel':
    """
    Language model that shares weights with a decoder
    """
    assert config.lm_pretrain_layers_target > 0
    assert decoder.embedding.embed_weight is not None
    assert decoder.lm_pre_rnn is not None
    lmodel = SharedLanguageModel(
        num_embed=config.num_embed_target,
        vocab_size=config.vocab_target_size,
        dropout=config.dropout,
        rnn_num_layers=config.lm_pretrain_layers_target,
        rnn_num_hidden=config.rnn_num_hidden,
        rnn_cell_type=config.rnn_cell_type,
        rnn_residual_connections=config.rnn_residual_connections,
        rnn_forget_bias=rnn_forget_bias,
        # Weight sharing happens here
        embedding_prefix=C.TARGET_EMBEDDING_PREFIX,
        rnn_prefix=C.DECODER_PREFIX+C.LM_TARGET_PREFIX,
        embedding_params=decoder.embedding.embed_weight,
        rnn_params=decoder.lm_pre_rnn.params,
        cls_w_params=decoder.cls_w,
        cls_b_params=decoder.cls_b
        )
    return lmodel


def get_lm_from_options(
        num_embed,
        vocab_size,
        dropout,
        rnn_num_layers,
        rnn_num_hidden,
        rnn_cell_type,
        rnn_residual_connections,
        rnn_forget_bias,
        weight_tying) -> 'SharedLanguageModel':
    """
    Language model with no weight sharing
    """
    return SharedLanguageModel(
        num_embed,
        vocab_size,
        dropout,
        rnn_num_layers,
        rnn_num_hidden,
        rnn_cell_type,
        rnn_residual_connections,
        rnn_forget_bias,
        weight_tying=weight_tying
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
                 weight_tying=False,
                 embedding_prefix=C.SOURCE_EMBEDDING_PREFIX,
                 rnn_prefix=C.STACKEDRNN_PREFIX,
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
            prefix=embedding_prefix,
            dropout=self.dropout,
            params=embedding_params
        )

        # Build parameters - rnn
        self.rnn = sockeye.rnn.get_stacked_rnn(
            cell_type=self.rnn_cell_type,
            num_hidden=self.rnn_num_hidden,
            num_layers=self.rnn_num_layers,
            dropout=self.dropout,
            prefix=rnn_prefix,
            residual=self.rnn_residual_connections,
            forget_bias=self.rnn_forget_bias,
            params=rnn_params
        )

        # Build paramers - output layer
        if cls_w_params is not None:
            self.cls_w = cls_w_params
        elif weight_tying:
            self.cls_w = self.embedding.embed_weight
        else:
            self.cls_w = mx.sym.Variable("lm_cls_weight")  # TODO: revisit prefix

        if cls_b_params is not None:
            self.cls_b = cls_b_params
        else:
            self.cls_b = mx.sym.Variable("lm_cls_bias")  # TODO: revisit prefix

    def encode(self, data, seq_len):

        data, _, _ = self.embedding.encode(data, None, seq_len)

        self.rnn.reset()
        outputs, states = self.rnn.unroll(seq_len, inputs=data, merge_outputs=True)

        pred = mx.sym.reshape(outputs, shape=(-1, self.rnn_num_hidden))
        pred = mx.sym.FullyConnected(data=pred,
                                     num_hidden=self.vocab_size,
                                     weight=self.cls_w,
                                     bias=self.cls_b,
                                     name=C.LOGITS_NAME)
        return pred
