# This is an implementation of the following work:
# Dual Learning for Machine Translation
# Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
# https: //arxiv.org/abs/1611.00179 (accepted at NIPS 2016)
# Developed by Vu Cong Duy Hoang (vhoang2@student.unimelb.edu.au)
# Date: July 2017
# ---------------------------------------------------------------------------------

"""
Code for dual learning (aka round tripping) framework
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
                 rnn_forget_bias: Optional[float] = None,
                 bucketing: Optional[bool] = True) -> None:
        super(sockeye.training.TrainingModel, self).__init__(sockeye.model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME)))

        self.vocab = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_folder, C.VOCAB_SRC_NAME))
        
        self.context = context
        
        self.bucketing = bucketing
        self.buckets = sockeye.data_io.define_buckets(self.config.max_seq_len)
        
        self._build_model_components(rnn_forget_bias)    
        self.module = self._build_module(self.config.max_seq_len)
        self.module_list = [self.module]

        # pre-bind the module
        self.module.bind(data_shapes=self._get_module_data_shapes(self.config.max_seq_len), 
                         label_shapes=self._get_module_label_shapes(self.config.max_seq_len), 
                         for_training=False,
                         grad_req="null")
        self.load_params_from_file(os.path.join(model_folder, C.PARAMS_BEST_NAME))
        self.module.init_params(arg_params=self.params,
                                allow_missing=False) # just copy from pre-initialized params

        self.lm_source_module = None
        self.lm_target_module = None
        
        self.training_monitor = None

    def _get_module_data_shapes(self, 
                                max_seq_len: int):
        return [mx.io.DataDesc(name=C.MONO_NAME, shape=(1, max_seq_len), layout=C.BATCH_MAJOR)]

    def _get_module_label_shapes(self, 
                                 max_seq_len: int):
        return [mx.io.DataDesc(name=C.MONO_LABEL_NAME, shape=(1, max_seq_len), layout=C.BATCH_MAJOR)]

    def get_inference_input(self,
                            tokens: List[str]) -> Tuple[mx.nd.NDArray, Optional[int]]:
        """
        Returns NDArray of source ids, NDArray of sentence length, and corresponding bucket_key

        :param tokens: List of input tokens.
        """
        _, bucket_key = sockeye.data_io.get_bucket(len(tokens), self.buckets)
        if bucket_key is None:
            logger.warning("Input (%d) exceeds max bucket size (%d). Stripping", len(tokens), self.buckets[-1])
            bucket_key = self.buckets[-1]
            tokens = tokens[:bucket_key]

        source = mx.nd.zeros((1, bucket_key))
        ids = sockeye.data_io.tokens2ids(tokens, self.vocab)
        for i, wid in enumerate(ids):
            source[0, i] = wid
               
        return source, bucket_key 

    def _build_model_components(self, 
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

    # get the normalized log-likelihood given a sentence
    def compute_ll(self, 
                   batch: mx.io.DataBatch, 
                   val_metric: mx.metric.CompositeEvalMetric) -> float:
        val_metric.reset()
        
        self.module.forward(data_batch=batch, is_train=False)  
        self.module.update_metric(val_metric, batch.label)
        
        normLL = -np.log(val_metric.get_name_value()[0][1])
               
        return normLL # conversion from perplexity to normalized log-likelihood, normLL = -log(perplexity). Smarter way?
    
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
        self.module = self._build_train_module() # learning module

        # init model
        # bind the abstract data
        max_data_shapes = self._get_module_data_shapes(self.max_input_len)
        max_label_shapes = self._get_module_label_shapes(self.max_input_len)
        self.module.bind(data_shapes=max_data_shapes, 
                         label_shapes=max_label_shapes, 
                         force_rebind=False,
                         for_training=True, 
                         grad_req='write')
        
        self.module.symbol.save(os.path.join(model_folder, C.SYMBOL_NAME))

        initializer = sockeye.initializer.get_initializer(C.RNN_INIT_ORTHOGONAL, lexicon=None) # FIXME: these values are set manually for now!
        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False) # just copy from pre-initialized params (self.params)

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
    def _build_train_module(self):
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

            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source, source_length, seq_len=source_seq_len)
            source_lexicon = self.lexicon.lookup(source) if self.lexicon else None

            logits = self.decoder.decode(source_encoded, source_encoded_seq_len, source_encoded_length,
                                         target, target_seq_len, source_lexicon)

            outputs = loss.get_loss(logits, labels)

            return mx.sym.Group(outputs), data_names, label_names

        if self.bucketing == False: # un-roll
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

    def set_params_inference_modules(self):        
        # set params for inference module
        arg_params, _ = self.module.get_params()
        self.encoder_module.init_params(arg_params=arg_params, 
                                       aux_params=None, 
                                       allow_missing=False, 
                                       force_init=True)
        self.decoder_module.init_params(arg_params=arg_params, 
                                       aux_params=None, 
                                       allow_missing=False, 
                                       force_init=True)
        
    def setup_optimizer(self, initial_learning_rate: float, 
                        opt_configs: Tuple[str, float, float, float, 'sockeye.lr_scheduler.LearningRateScheduler']):
        self.learning_rate = initial_learning_rate
        
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

        self.module.init_optimizer(kvstore='local', 
                                   optimizer=optimizer, 
                                   optimizer_params=optimizer_params)

    # get the log-likelihood given a batch of data
    def compute_ll(self, 
                     batch: mx.io.DataBatch, 
                     val_metric: mx.metric.CompositeEvalMetric) -> float:
        val_metric.reset()
        
        self.module.forward(data_batch=batch, is_train=True)
        self.module.update_metric(val_metric, batch.label)
        
        normLL = -np.log(val_metric.get_name_value()[0][1])
        
        return normLL # conversion from perplexity to normalized log-likelihood, normLL = -log(perplexity). Smarter way?
    
    def forward(self, 
                batch: mx.io.DataBatch):         
        self.module.forward(data_batch=batch, is_train=True)  

    def backward_and_collect_gradients(self, 
                                       reward: float,
                                       agg_grads: List[List[mx.nd.NDArray]]) -> List[List[mx.nd.NDArray]]:
        # execute backward step
        self.module.backward()

        # collect and agggregate gradients
        if agg_grads == None:
            agg_grads = [[reward * grad.copy() for grad in grads] for grads in self.module._curr_module._exec_group.grad_arrays] # current gradients
        else:
            for gradsc, gradsp in zip(self.module._curr_module._exec_group.grad_arrays, agg_grads):
                for gradc, gradp in zip(gradsc, gradsp):
                    gradp += reward * gradc.copy() # aggregate gradients

        return agg_grads # FIXME: how to change agg_grads inside this function!
    
    # evaluate over a given development set
    def evaluate_dev(self, 
                     val_iter: sockeye.data_io.ParallelBucketSentenceIter,
                     val_metric: mx.metric.CompositeEvalMetric):
        val_iter.reset()
        val_metric.reset()

        for _, eval_batch in enumerate(val_iter):
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(val_metric, eval_batch.label)

        total_val = 0.0
        for _, val in val_metric.get_name_value(): 
            total_val += val

        return total_val

    def update_params(self, 
                      k: int,
                      agg_grads: List[List[mx.nd.NDArray]]):
        for param_list, grad_list in zip(self.module._curr_module._exec_group.param_arrays, agg_grads):
            for param, grad in zip(param_list, grad_list):
                mx.ndarray.sgd_update(weight=param, grad=grad/float(k), lr=self.learning_rate, out=param)
    
        arg_params, _ = self.module.get_params()  # sync aux params across devices
        self.params.update(arg_params)

    def save_params(self, output_folder: str, 
                    checkpoint: int):
        """
        Saves the parameters to disk.
        """
        # update the current params from self.module first
        self.save_params_to_file(os.path.join(output_folder, C.PARAMS_NAME % checkpoint))
