# This is an implementation of the following work:
# Dual Learning for Machine Translation
# Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
# https: //arxiv.org/abs/1611.00179 (accepted at NIPS 2016)
# Developed by Vu Cong Duy Hoang (vhoang2@student.unimelb.edu.au)
# Date: July 2017
# ---------------------------------------------------------------------------------

"""
Code for dual learning framework
"""
import logging
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import mxnet as mx
import numpy as np

import sockeye.bleu
import sockeye.constants as C
import sockeye.data_io
import sockeye.model
import sockeye.utils
import sockeye.vocab
from sockeye.attention import AttentionState
from sockeye.decoder import DecoderState

logger = logging.getLogger(__name__)

class InferenceLModel(sockeye.training.TrainingModel):
    """
    InferenceLModel is a SockeyeModel that supports inference functionalities for RNNLM.
    This object is orthogonal to sockeye.traininglm.TrainingLModel which is used for training only.

    :param model_folder: Folder to load model from.
    :param context: MXNet context to bind modules to.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param max_input_len: Maximum input length.   
    """
    def __init__(self,
                 model_folder: str,
                 context: mx.context.Context,
                 fused: bool,
                 max_input_len: Optional[int],
                 rnn_forget_bias: Optional[float] = None,
                 bucketing: Optional[bool] = True) -> None:
        super(sockeye.training.TrainingModel, self).__init__(sockeye.model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME)))
        
        self.context = context
        
        self.bucketing = bucketing
        
        self._build_model_components(self.config.max_seq_len, fused, rnn_forget_bias)
        
        self.module = self._build_module(train_iter, self.config.max_seq_len)
        self.module_list = [self.module]
        
        self.lm_source_module = None
        self.lm_target_module = None
        
        self.training_monitor = None

    def _build_model_components(self, 
                                fused, 
                                rnn_forget_bias):
        self.lm = sockeye.lm.get_lm_from_options(self.config.num_embed_source,
                                                 self.config.vocab_source_size,
                                                 self.config.dropout,
                                                 self.config.rnn_num_layers,
                                                 self.config.rnn_num_hidden,
                                                 self.config.rnn_cell_type,
                                                 self.config.rnn_residual_connections,
                                                 rnn_forget_bias)

        self.rnn_cells = [self.lm.rnn]
        self.built = True

    def _build_module(self,
                      max_seq_len: int):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.MONO_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.MONO_LABEL_NAME), shape=(-1,))

        loss = sockeye.loss.get_loss(self.config)

        data_names = [C.MONO_NAME]
        label_names = [C.MONO_LABEL_NAME]

        def sym_gen(seq_len):

            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            logits = self.lm.encode(source, seq_len=seq_len)

            outputs = loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        if self.bucketing:
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=max_seq_len,
                                          context=self.context)
        else:
            symbol, _, __ = sym_gen(max_seq_len)
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

    # get the negative log-likelihood given a batch of data
    def compute_nll(self, 
                   batch: mx.io.DataBatch, 
                   val_metric: mx.metric.CompositeEvalMetric):
        val_metric.reset()
        
        self.module.forward(data_batch=batch, is_train=False)  
        self.module.update_metric(val_metric, batch.label)
        
        total_loss = 0
        for name, val in val_metric.get_name_value():
            total_loss += val
        
        return -np.log(total_loss) # FIXME: conversion from perplexity to normalized log-likelihood, -logLL = log(perplexity). Smarter way?

class TrainableInferenceModel(sockeye.inference.InferenceModel):
    """
    TrainableInferenceModel is a SockeyeModel that supports both training and inference functionalities for attention-based encoder-decoder model.

    :param model_folder: Folder to load model from.
    :param context: MXNet context to bind modules to.
    :param fused: Whether to use FusedRNNCell (CuDNN). Only works with GPU context.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in model_folder.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    """
    def __init__(self,
                 model_folder: str,
                 context: mx.context.Context,
                 fused: bool,
                 beam_size: int,
                 max_input_len: Optional[int],   
                 checkpoint: Optional[int] = None,
                 softmax_temperature: Optional[float] = None,
                 bucketing: Optional[bool] = True):
        # inherit InferenceModel
        super().__init__(model_folder=model_folder,
                         context=context,
                         fused=False,
                         max_input_len=max_input_len,
                         beam_size=beam_size,
                         softmax_temperature=softmax_temperature,
                         checkpoint=checkpoint)
        
        # bucketing flag
        self.bucketing = bucketing

        # build module for learnable model(s)  
        self.module = self._build_module() # learning module

        # init model
        # bind the abstract data
        max_data_shapes = self._get_module_data_shapes(self.max_input_len)
        max_label_shapes = self._get_module_label_shapes(self.max_input_len)
        self.module.bind(data_shapes=max_data_shapes, 
                         label_shapes=max_label_shapes, 
                         for_training=True, 
                         grad_req='write')
        
        self.module.symbol.save(os.path.join(model_folder, C.SYMBOL_NAME))

        initializer = sockeye.initializer.get_initializer(C.RNN_INIT_ORTHOGONAL, lexicon=None) # FIXME: these values are set manually for now!
        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)

    def _get_module_data_shapes(self, 
                                max_seq_len: int):
        return [mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, max_seq_len), layout=C.BATCH_MAJOR),
                mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, max_seq_len), layout=C.BATCH_MAJOR)]

    def _get_module_label_shapes(self, 
                                 max_seq_len: int):
        return [mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, max_seq_len), layout=C.BATCH_MAJOR)]

    # self.encoder_module and self.decoder_module will be used for translation.
    # this self.module will be used for training/learning the model parameters.
    def _build_module(self):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        loss = sockeye.loss.get_loss(self.config)

        data_names = [C.SOURCE_NAME, C.SOURCE_LENGTH_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            source_encoded = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            logits = self.decoder.decode(source_encoded, source_seq_len, source_length,
                                         target, target_seq_len, source_lexicon)

            outputs = loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        if self.bucketing == False: # run-roll
            symbol, _, __ = sym_gen((self.max_input_len, self.max_input_len))
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)
        else:
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=(self.max_input_len, self.max_input_len),
                                          context=self.context)

    def setup_optimizer(self, initial_learning_rate: float, 
                        opt_configs: Tuple[str, float, float, float, 'sockeye.lr_scheduler.LearningRateScheduler']):
        optimizer = opt_configs[0]
        optimizer_params = {'wd': opt_configs[1],
                            "learning_rate": initial_learning_rate}
        if opt_configs[4] is not None:
            optimizer_params["lr_scheduler"] = opt_configs[4]
        clip_gradient = None if opt_configs[3] < 0 else opt_configs[3]
        if clip_gradient is not None:
            optimizer_params["clip_gradient"] = opt_configs[3]
        if opt_configs[2] is not None:
            optimizer_params["momentum"] = opt_configs[2]
        optimizer_params["rescale_grad"] = 1.0

        self.module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

    # get the log-likelihood given a batch of data
    def compute_nll(self, 
                     batch: mx.io.DataBatch, 
                     val_metric: mx.metric.CompositeEvalMetric):
        val_metric.reset()
        
        self.module.forward(data_batch=batch, is_train=True)  
        self.module.update_metric(val_metric, batch.label)
        
        total_val = 0
        for _, val in val_metric.get_name_value():
            total_val += val
        
        return -np.log(total_val) # FIXME: conversion from perplexity to normalized log-likelihood, -logLL = log(perplexity). Smarter way?

    def forward(self, 
                batch: mx.io.DataBatch):         
        self.module.forward(data_batch=batch, is_train=True)  
 
    # evaluate over a given development set
    def evaluate_dev(self, 
                     val_iter: sockeye.data_io.ParallelBucketSentenceIter,
                     val_metric: mx.metric.CompositeEvalMetric):
        val_iter.reset()
        val_metric.reset()

        for nbatch, eval_batch in enumerate(val_iter):
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(val_metric, eval_batch.label)

        total_val = 0.0
        for _, val in val_metric.get_name_value(): 
            total_val += val

        return total_val

    def update_params(self, 
                      reward_scale: float):
        self.module.backward() # backward step

        self.module._curr_module._optimizer.rescale_grad = reward_scale # FIXME: this is hacky, thinking another way?
        self.module.update() # update the parameters

    def save_params(self, output_folder: str, 
                    checkpoint: int):
        """
        Saves the parameters to disk.
        """
        arg_params, aux_params = self.module.get_params()  # sync aux params across devices
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        params_base_fname = C.PARAMS_NAME % checkpoint
        self.save_params_to_file(os.path.join(output_folder, params_base_fname))
