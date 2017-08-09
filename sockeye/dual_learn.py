# This is an implementation of the following work:
# Dual Learning for Machine Translation
# Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
# https: //arxiv.org/abs/1611.00179 (accepted at NIPS 2016)
# Developed by Vu Cong Duy Hoang (vhoang2@student.unimelb.edu.au)
# Date: July 2017
# ---------------------------------------------------------------------------------

"""
Simple Training CLI.
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import logging
from contextlib import ExitStack
from typing import Optional, Dict, List, Tuple, Iterator

import mxnet as mx
import numpy as np
import math

import sockeye.arguments as arguments
import sockeye.attention
import sockeye.constants as C
import sockeye.data_io
import sockeye.decoder
import sockeye.encoder
import sockeye.initializer
import sockeye.lexicon
import sockeye.lr_scheduler
import sockeye.model
import sockeye.training
import sockeye.inference
import sockeye.traininglm
import sockeye.dual_learning
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, expand_requested_device_ids

logger = logging.getLogger(__name__)

def _check_path(opath, logger, overwrite):
    training_state_dir = os.path.join(opath, C.TRAINING_STATE_DIRNAME)
    if os.path.exists(opath):
        if overwrite:
            logger.info("Removing existing output folder %s.", opath)
            shutil.rmtree(opath)
            os.makedirs(opath)
        elif os.path.exists(training_state_dir):
            with open(os.path.join(opath, C.ARGS_STATE_NAME), "r") as fp:
                old_args = json.load(fp)
            arg_diffs = _dict_difference(vars(args), old_args) | _dict_difference(old_args, vars(args))
            # Remove args that may differ without affecting the training.
            arg_diffs -= set(C.ARGS_MAY_DIFFER)
            if arg_diffs:
                # We do not have the logger yet
                logger.error("Mismatch in arguments for training continuation.")
                logger.error("Differing arguments: %s.", ", ".join(arg_diffs))
                sys.exit(1)
        else:
            logger.error("Refusing to overwrite existing output folder %s.", opath)
            sys.exit(1)
    else:
        os.makedirs(opath)

def _read_lines(path: str, len_limit=None , limit=None) -> Iterator[List[str]]:
    """
    Returns a list of lines in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of lines.
    """
    with sockeye.data_io.smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            if len_limit is not None and len(line.strip().split()) > len_limit: continue
            yield line.strip()

def _get_inputs(tokens: List[str],
                vocab: Dict[str, int],
                buckets: List[int]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray, Optional[int]]:
        """
        Returns NDArray of source ids, NDArray of sentence length, and corresponding bucket_key

        :param tokens: List of input tokens.
        """
        _, bucket_key = sockeye.data_io.get_bucket(len(tokens), buckets)
        if bucket_key is None:
            bucket_key = buckets[-1]
            tokens = tokens[:bucket_key]

        source = mx.nd.zeros((1, bucket_key))
        ids = sockeye.data_io.tokens2ids(tokens, vocab)
        for i, wid in enumerate(ids):
            source[0, i] = wid
        length = mx.nd.array([len(ids)])

        source_label = mx.nd.zeros((1, bucket_key)) # use batch_size=1
        for i in range(len(ids) - 1):
            source_label[0, i] = ids[i + 1]
        source_label[0, len(ids) - 1] = vocab[C.EOS_SYMBOL]
        
        return source, source_label, length, bucket_key

# This one is to implement the idea of soft translation in dual learning framework.
# (work in progress)
def _soft_dual_learn(context: mx.context.Context, 
                     vocab_source: Dict[str, int],
                     vocab_target: Dict[str, int],
                     all_data: Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', List[str], List[str]], 
                     models: List[sockeye.dual_learning.TrainableInferenceModel], 
                     opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                     grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                     lmon: Tuple[int, int], # extra stuffs for learning monitor
                     model_folders: Tuple[str, str],
                     k: int):
    logger.error("Soft dual learning not yet implemented yet!")
    sys.exit(1)

def _dual_learn_batch_soft_landing(context: mx.context.Context, 
                vocab_source: Dict[str, int],
                vocab_target: Dict[str, int],
                all_data: Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', 
                                'ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', 
                                List[str], List[str]], 
                models: List[sockeye.dual_learning.TrainableInferenceModel], 
                opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                lmon: Tuple[int, int], # extra stuffs for learning monitor
                model_folders: Tuple[str, str],
                k: int,
                minibatch_size: int):
    # set up decoders/translators
    logger.info("DEBUG - 8a")
    dec_s2t = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[0]], 
                                           vocab_source=vocab_source, 
                                           vocab_target=vocab_target)
    dec_t2s = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[1]], 
                                           vocab_source=vocab_target,
                                           vocab_target=vocab_source)
    logger.info("Passed!")

    # set up monolingual data access/ids
    logger.info("DEBUG - 8b")
    lsmono = len(all_data[4])
    ltmono = len(all_data[5])
    orders_s = list(range(2 * lsmono)) # double the length of sampling data
    orders_t = list(range(2 * ltmono)) # double the length of sampling data
    np.random.shuffle(orders_s)
    np.random.shuffle(orders_t)
    logger.info("Passed!")

    # set up optimizers
    logger.info("DEBUG - 8c")
    dec_s2t.models[0].setup_optimizer(initial_learning_rate=grad_alphas[1], opt_configs=opt_configs)
    dec_t2s.models[0].setup_optimizer(initial_learning_rate=grad_alphas[2], opt_configs=opt_configs)
    logger.info("Passed!")

    # create eval metric
    metric_val = mx.metric.create([mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])]) # FIXME: use cross-entropy loss instead

    # print the perplexities over dev (for debugging only)
    best_dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[2], metric_val)
    best_dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[3], metric_val)
    logger.info("Perplexity over development set from source-to-target model:" + str(best_dev_pplx_s2t))
    logger.info("Perplexity over development set from target-to-source model:" + str(best_dev_pplx_t2s))

    # start the dual learning algorithm
    logger.info("DEBUG - 8d (learning loop)")
    id_s = 0
    id_t = 0
    r = 0 # learning round
    e_s = 0 # epoch over source mono data
    e_t = 0 # epoch over target mono data
    flag = True # role of source and target
    while e_s < lmon[0] or e_t < lmon[0]: 
        if id_s >= len(orders_s): # source monolingual data
            # shuffle the data
            np.random.shuffle(orders_s)
            all_data[0].reset()
            
            # update epochs
            e_s += 1
            
            # reset the data ids
            id_s = id_s - len(orders_s)
        if id_t >= len(orders_t): # target monoingual data
            # shuffle the data
            np.random.shuffle(orders_t)
            all_data[1].reset()
                    
            # update epochs
            e_t += 1
            
            # reset the data ids
            id_t = id_t - len(orders_t)

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_t respectively
        sents = []
        if flag == True:
            sents = [(all_data[4][id_b], 0) if id_b < lsmono else ("", 1) for id_b in orders_s[id_s:id_s + minibatch_size]]
        else:
            sents = [(all_data[5][id_b], 0) if id_b < ltmono else ("", 1) for id_b in orders_t[id_t:id_t + minibatch_size]]
            
        logger.info("Sampled sentences: " + str(sents))

        def _process_samples(sents, 
                             strain_iter, ttrain_iter, 
                             tm_s2t, tm_t2s, lm):
            agg_grads_s2t = None
            agg_grads_t2s = None     
            for sent, sampl_type in sents:
                # sampling from real parallel data
                if sampl_type == 1: # sent will be "".
                    # fit the data with normal cross-entropy likelihood loss
                    logger.info("DEBUG - 8g (learning loop) - fit the real paralel data")
                    tm_s2t.models[0].forward(strain_iter.next())
                    tm_t2s.models[0].forward(ttrain_iter.next())

                    logger.info("DEBUG - 8g (learning loop) - backward and collect gradients")
                    agg_grads_s2t = tm_s2t.models[0].backward_and_collect_gradients(reward=1, 
                                                                                    agg_grads=agg_grads_s2t)
                    agg_grads_t2s = tm_t2s.models[0].backward_and_collect_gradients(reward=1,
                                                                                    agg_grads=agg_grads_t2s)
                    
                    continue;

                # sampling from monolingual data
                # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
                logger.info("DEBUG - 8d (learning loop) - K-best translation")
                trans_input = tm_s2t.make_input(0, sent) # 0: unused for now!
                trans_outputs = tm_s2t.translate_kbest(trans_input, k) # generate k-best translations
                mid_hyps = [list(sockeye.data_io.get_tokens(trans[1])) for trans in trans_outputs]
                mid_hyp_scores = [trans[4] for trans in trans_outputs]
                logger.info(str(k) +"-best translations: " + str(mid_hyps))#mid_hyps)
                logger.info("Scores: " + str(mid_hyp_scores))
                logger.info("Passed!")

                # create an input batch as input_iter
                for mid_hyp in mid_hyps:
                    if len(mid_hyp) == 0: continue
                    logger.info("DEBUG - 8d (learning loop) - create data batches")
                    infer_input_s2t = _get_inputs(trans_input[2], tm_s2t.vocab_source, tm_s2t.buckets)
                    infer_input_t2s = _get_inputs(mid_hyp, tm_t2s.vocab_source, tm_t2s.buckets) 
                    input_batch_s2t = mx.io.DataBatch(data=[infer_input_s2t[0], infer_input_s2t[2], infer_input_t2s[0]], 
                                                      label=[infer_input_t2s[1]], # slice one position for label seq
                                                      bucket_key=(infer_input_s2t[3],infer_input_t2s[3]),
                                                      provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                      provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                    input_batch_t2s = mx.io.DataBatch(data=[infer_input_t2s[0], infer_input_t2s[2], infer_input_s2t[0]], 
                                                      label=[infer_input_s2t[1]], #  slice one position for label seq
                                                      bucket_key=(infer_input_t2s[3],infer_input_s2t[3]),
                                                      provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)],
                                                      provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)])
                    input_batch_mono = mx.io.DataBatch(data=[infer_input_t2s[0]], 
                                                       label=[infer_input_t2s[1]], #  slice one position for label seq
                                                       bucket_key=infer_input_t2s[3],
                                                       provide_data=[mx.io.DataDesc(name=C.MONO_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                       provide_label=[mx.io.DataDesc(name=C.MONO_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                    logger.info("Passed!")

                    logger.info("DEBUG - 8e (learning loop) - computing rewards")
                    # set the language-model reward for currently-sampled sentence from P(mid_hyp; mod_mlm_t)
                    # 1) bind the data {(mid_hyp,mid_hyp)} into the model's module
                    # 2) do forward step --> get the r1 = P(mid_hyp; mod_mlm_t)
                    print("Computing reward_1")
                    reward_1 = lm.compute_ll(input_batch_mono, metric_val)
                    logger.info("reward_1=" + str(reward_1))
         
                    # set the communication reward for currently-sampled sentence from P(sentA|mid_hype; mod_am_t2s)
                    # do forward step --> get the r2 = P(sentA|mid_hyp; mod_am_t2s)
                    print("Computing reward_2") 
                    reward_2 = tm_t2s.models[0].compute_ll(input_batch_t2s, metric_val) # also includes forward step
                    logger.info("reward_2=" + str(reward_2))

                    # reward interpolation: r = alpha * r1 + (1 - alpha) * r2
                    reward = grad_alphas[0] * reward_1 + (1.0 - grad_alphas[0]) * reward_2
                    logger.info("total_reward=" + str(reward))
                    logger.info("Passed!")
        
                    # do forward step for s2t model - FIXME: this step is done twice (one for inference and one for computing loss). Better way? 
                    tm_s2t.models[0].forward(input_batch_s2t)

                    # do backward steps & collect gradients 
                    logger.info("DEBUG - 8g (learning loop) - backward and collect gradients")
                    agg_grads_s2t = tm_s2t.models[0].backward_and_collect_gradients(reward=-reward, # gradient ascent
                                                                                    agg_grads=agg_grads_s2t)
                    agg_grads_t2s = tm_t2s.models[0].backward_and_collect_gradients(reward=1.0 - grad_alphas[0], # gradient ascent
                                                                                    agg_grads=agg_grads_t2s)
                    logger.info("Passed!")
        
            if agg_grads_s2t != None and agg_grads_t2s != None:
                # update model parameters
                logger.info("DEBUG - 8h (learning loop) - update params for learning modules")
                tm_s2t.models[0].update_params(k=k, 
                                               agg_grads=agg_grads_s2t)
                tm_t2s.models[0].update_params(k=k, 
                                               agg_grads=agg_grads_t2s)
                logger.info("Passed!")
            
                # re-set the params for inference modules after each update of model parameters
                logger.info("DEBUG - 8i (learning loop) - update params for inference modules")
                tm_s2t.models[0].set_params_inference_modules()
                tm_t2s.models[0].set_params_inference_modules()
                logger.info("Passed!")

        if flag == False:   
            _process_samples(sents, 
                             all_data[1], 
                             all_data[0],
                             dec_t2s, 
                             dec_s2t, 
                             models[3])
            r += 1 # next round       
            id_t += minibatch_size
        else:
            _process_samples(sents, 
                             all_data[0], 
                             all_data[1],
                             dec_s2t, 
                             dec_t2s, 
                             models[2])
            id_s += minibatch_size
   
        # switch source and target roles
        flag = not flag
                                
        # testing over the development data (all_data[2] and all_data[3]) to check the improvements (after a desired number of rounds)
        if r >= lmon[1]:
            # s2t model
            dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[2], metric_val)

            # t2s model
            dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[3], metric_val)

            # print the perplexities to the consoles
            logger.info("-------------------------------------------------------------------------")
            logger.info("Perplexity over development set from source-to-target model:" + str(dev_pplx_s2t))
            logger.info("Perplexity over development set from target-to-source model:" + str(dev_pplx_t2s))
            logger.info("-------------------------------------------------------------------------")

            # save the better model(s) if losses over dev are decreasing!
            if best_dev_pplx_s2t > dev_pplx_s2t:
                dec_s2t.models[0].save_params(model_folders[0], 1) # FIXME: add checkpoints
                best_dev_pplx_s2t = dev_pplx_s2t
            if best_dev_pplx_t2s > dev_pplx_t2s:
                dec_t2s.models[0].save_params(model_folders[1], 1) # FIXME: add checkpoints
                best_dev_pplx_t2s = dev_pplx_t2s

            r = 0 # reset the round

        # FIXME: change the ratio of real parallel vs. synthetic parallel data in soft-landing strategy (question: when?)
        # ...

def _dual_learn_batch(context: mx.context.Context, 
                vocab_source: Dict[str, int],
                vocab_target: Dict[str, int],
                all_data: Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', List[str], List[str]], 
                models: List[sockeye.dual_learning.TrainableInferenceModel], 
                opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                lmon: Tuple[int, int], # extra stuffs for learning monitor
                model_folders: Tuple[str, str],
                k: int,
                minibatch_size: int):
    # set up decoders/translators
    logger.info("DEBUG - 8a")
    dec_s2t = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[0]], 
                                           vocab_source=vocab_source, 
                                           vocab_target=vocab_target)
    dec_t2s = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[1]], 
                                           vocab_source=vocab_target,
                                           vocab_target=vocab_source)
    logger.info("Passed!")

    # set up monolingual data access/ids
    logger.info("DEBUG - 8b")
    orders_s = list(range(len(all_data[2])))
    orders_t = list(range(len(all_data[3])))
    np.random.shuffle(orders_s)
    np.random.shuffle(orders_t)
    logger.info("Passed!")

    # set up optimizers
    logger.info("DEBUG - 8c")
    dec_s2t.models[0].setup_optimizer(initial_learning_rate=grad_alphas[1], opt_configs=opt_configs)
    dec_t2s.models[0].setup_optimizer(initial_learning_rate=grad_alphas[2], opt_configs=opt_configs)
    logger.info("Passed!")

    # create eval metric
    metric_val = mx.metric.create([mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])]) # FIXME: use cross-entropy loss instead

    # print the perplexities over dev (for debugging only)
    best_dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[0], metric_val)
    best_dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[1], metric_val)
    logger.info("Perplexity over development set from source-to-target model:" + str(best_dev_pplx_s2t))
    logger.info("Perplexity over development set from target-to-source model:" + str(best_dev_pplx_t2s))

    # start the dual learning algorithm
    logger.info("DEBUG - 8d (learning loop)")
    id_s = 0
    id_t = 0
    r = 0 # learning round
    e_s = 0 # epoch over source mono data
    e_t = 0 # epoch over target mono data
    flag = True # role of source and target
    while e_s < lmon[0] or e_t < lmon[0]: 
        if id_s >= len(orders_s): # source monolingual data
            # shuffle the data
            np.random.shuffle(orders_s)
            
            # update epochs
            e_s += 1
            
            # reset the data ids
            id_s = id_s - len(orders_s)
        if id_t >= len(orders_t): # target monoingual data
            # shuffle the data
            np.random.shuffle(orders_t)
                    
            # update epochs
            e_t += 1
            
            # reset the data ids
            id_t = id_t - len(orders_t)

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_t respectively
        sents = []
        if flag == True:
            sents = [all_data[2][id_b] for id_b in orders_s[id_s:id_s + minibatch_size]]
        else:
            sents = [all_data[3][id_b] for id_b in orders_t[id_t:id_t + minibatch_size]]
            
        logger.info("Sampled sentences: " + str(sents))

        def _process_samples(sents, tm_s2t, tm_t2s, lm):
            agg_grads_s2t = None
            agg_grads_t2s = None     
            for sent in sents:
                # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
                logger.info("DEBUG - 8d (learning loop) - K-best translation")
                trans_input = tm_s2t.make_input(0, sent) # 0: unused for now!
                trans_outputs = tm_s2t.translate_kbest(trans_input, k) # generate k-best translations
                mid_hyps = [list(sockeye.data_io.get_tokens(trans[1])) for trans in trans_outputs]
                mid_hyp_scores = [trans[4] for trans in trans_outputs]
                logger.info(str(k) +"-best translations: " + str(mid_hyps))#mid_hyps)
                logger.info("Scores: " + str(mid_hyp_scores))
                logger.info("Passed!")

                # create an input batch as input_iter
                for mid_hyp in mid_hyps:
                    if len(mid_hyp) == 0: continue
                    logger.info("DEBUG - 8d (learning loop) - create data batches")
                    infer_input_s2t = _get_inputs(trans_input[2], tm_s2t.vocab_source, tm_s2t.buckets)
                    infer_input_t2s = _get_inputs(mid_hyp, tm_t2s.vocab_source, tm_t2s.buckets) 
                    input_batch_s2t = mx.io.DataBatch(data=[infer_input_s2t[0], infer_input_s2t[2], infer_input_t2s[0]], 
                                                      label=[infer_input_t2s[1]], # slice one position for label seq
                                                      bucket_key=(infer_input_s2t[3],infer_input_t2s[3]),
                                                      provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                      provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                    input_batch_t2s = mx.io.DataBatch(data=[infer_input_t2s[0], infer_input_t2s[2], infer_input_s2t[0]], 
                                                      label=[infer_input_s2t[1]], #  slice one position for label seq
                                                      bucket_key=(infer_input_t2s[3],infer_input_s2t[3]),
                                                      provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                    mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)],
                                                      provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)])
                    input_batch_mono = mx.io.DataBatch(data=[infer_input_t2s[0]], 
                                                       label=[infer_input_t2s[1]], #  slice one position for label seq
                                                       bucket_key=infer_input_t2s[3],
                                                       provide_data=[mx.io.DataDesc(name=C.MONO_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                       provide_label=[mx.io.DataDesc(name=C.MONO_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                    logger.info("Passed!")

                    logger.info("DEBUG - 8e (learning loop) - computing rewards")
                    # set the language-model reward for currently-sampled sentence from P(mid_hyp; mod_mlm_t)
                    # 1) bind the data {(mid_hyp,mid_hyp)} into the model's module
                    # 2) do forward step --> get the r1 = P(mid_hyp; mod_mlm_t)
                    print("Computing reward_1")
                    reward_1 = lm.compute_ll(input_batch_mono, metric_val)
                    logger.info("reward_1=" + str(reward_1))
         
                    # set the communication reward for currently-sampled sentence from P(sentA|mid_hype; mod_am_t2s)
                    # do forward step --> get the r2 = P(sentA|mid_hyp; mod_am_t2s)
                    print("Computing reward_2") 
                    reward_2 = tm_t2s.models[0].compute_ll(input_batch_t2s, metric_val) # also includes forward step
                    logger.info("reward_2=" + str(reward_2))

                    # reward interpolation: r = alpha * r1 + (1 - alpha) * r2
                    reward = grad_alphas[0] * reward_1 + (1.0 - grad_alphas[0]) * reward_2
                    logger.info("total_reward=" + str(reward))
                    logger.info("Passed!")
        
                    # do forward step for s2t model - FIXME: this step is done twice (one for inference and one for computing loss). Better way? 
                    tm_s2t.models[0].forward(input_batch_s2t)

                    # do backward steps & collect gradients 
                    logger.info("DEBUG - 8g (learning loop) - backward and collect gradients")
                    agg_grads_s2t = tm_s2t.models[0].backward_and_collect_gradients(reward=-reward, # gradient ascent
                                                                                    agg_grads=agg_grads_s2t)
                    agg_grads_t2s = tm_t2s.models[0].backward_and_collect_gradients(reward=1.0 - grad_alphas[0], # gradient ascent
                                                                                    agg_grads=agg_grads_t2s)
                    logger.info("Passed!")
        
            if agg_grads_s2t != None and agg_grads_t2s != None:
                # update model parameters
                logger.info("DEBUG - 8h (learning loop) - update params for learning modules")
                tm_s2t.models[0].update_params(k=k*minibatch_size, 
                                               agg_grads=agg_grads_s2t)
                tm_t2s.models[0].update_params(k=k*minibatch_size, 
                                               agg_grads=agg_grads_t2s)
                logger.info("Passed!")
            
                # re-set the params for inference modules after each update of model parameters
                logger.info("DEBUG - 8i (learning loop) - update params for inference modules")
                tm_s2t.models[0].set_params_inference_modules()
                tm_t2s.models[0].set_params_inference_modules()
                logger.info("Passed!")

        if flag == False:   
            _process_samples(sents, 
                            dec_t2s, 
                            dec_s2t, 
                            models[3])
            r += 1 # next round       
            id_t += minibatch_size
        else:
            _process_samples(sents, 
                            dec_s2t, 
                            dec_t2s, 
                            models[2])
            id_s += minibatch_size
   
        # switch source and target roles
        flag = not flag
                                
        # testing over the development data (all_data[0] and all_data[1]) to check the improvements (after a desired number of rounds)
        if r >= lmon[1]:
            # s2t model
            dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[0], metric_val)

            # t2s model
            dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[1], metric_val)

            # print the perplexities to the consoles
            logger.info("-------------------------------------------------------------------------")
            logger.info("Perplexity over development set from source-to-target model:" + str(dev_pplx_s2t))
            logger.info("Perplexity over development set from target-to-source model:" + str(dev_pplx_t2s))
            logger.info("-------------------------------------------------------------------------")

            # save the better model(s) if losses over dev are decreasing!
            if best_dev_pplx_s2t > dev_pplx_s2t:
                dec_s2t.models[0].save_params(model_folders[0], 1) # FIXME: add checkpoints
                best_dev_pplx_s2t = dev_pplx_s2t
            if best_dev_pplx_t2s > dev_pplx_t2s:
                dec_t2s.models[0].save_params(model_folders[1], 1) # FIXME: add checkpoints
                best_dev_pplx_t2s = dev_pplx_t2s

            r = 0 # reset the round

def _dual_learn(context: mx.context.Context, 
                vocab_source: Dict[str, int],
                vocab_target: Dict[str, int],
                all_data: Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', List[str], List[str]], 
                models: List[sockeye.dual_learning.TrainableInferenceModel], 
                opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                lmon: Tuple[int, int], # extra stuffs for learning monitor
                model_folders: Tuple[str, str],
                k: int):
    # set up decoders/translators
    logger.info("DEBUG - 8a")
    dec_s2t = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[0]], 
                                           vocab_source=vocab_source, 
                                           vocab_target=vocab_target)
    dec_t2s = sockeye.inference.Translator(context=context,
                                           ensemble_mode="linear", #unused
                                           set_bos=None, #unused
                                           models=[models[1]], 
                                           vocab_source=vocab_target,
                                           vocab_target=vocab_source)
    logger.info("Passed!")

    # set up monolingual data access/ids
    logger.info("DEBUG - 8b")
    orders_s = list(range(len(all_data[2])))
    orders_t = list(range(len(all_data[3])))
    np.random.shuffle(orders_s)
    np.random.shuffle(orders_t)
    logger.info("Passed!")

    # set up optimizers
    logger.info("DEBUG - 8c")
    dec_s2t.models[0].setup_optimizer(initial_learning_rate=grad_alphas[1], opt_configs=opt_configs)
    dec_t2s.models[0].setup_optimizer(initial_learning_rate=grad_alphas[2], opt_configs=opt_configs)
    logger.info("Passed!")

    # create eval metric
    metric_val = mx.metric.create([mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])]) # FIXME: use cross-entropy loss instead

    # print the perplexities over dev (for debugging only)
    best_dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[0], metric_val)
    best_dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[1], metric_val)
    logger.info("Perplexity over development set from source-to-target model:" + str(best_dev_pplx_s2t))
    logger.info("Perplexity over development set from target-to-source model:" + str(best_dev_pplx_t2s))

    # start the dual learning algorithm
    logger.info("DEBUG - 8d (learning loop)")
    id_s = 0
    id_t = 0
    r = 0 # learning round
    e_s = 0 # epoch over source mono data
    e_t = 0 # epoch over target mono data
    flag = True # role of source and target
    while e_s < lmon[0] or e_t < lmon[0]: 
        if id_s == len(orders_s): # source monolingual data
            # shuffle the data
            np.random.shuffle(orders_s)
            
            # update epochs
            e_s += 1
            
            # reset the data ids
            id_s = 0
        if id_t == len(orders_t): # target monoingual data
            # shuffle the data
            np.random.shuffle(orders_t)
                    
            # update epochs
            e_t += 1
            
            # reset the data ids
            id_t = 0

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_t respectively
        sent = all_data[2][orders_s[id_s]] if flag == True else all_data[3][orders_t[id_t]]
        logger.info("Sampled sentence: " + sent)

        def _process_sample(sent, tm_s2t, tm_t2s, lm):
            # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
            logger.info("DEBUG - 8d (learning loop) - K-best translation")
            trans_input = tm_s2t.make_input(0, sent) # 0: unused for now!
            trans_outputs = tm_s2t.translate_kbest(trans_input, k) # generate k-best translations
            mid_hyps = [list(sockeye.data_io.get_tokens(trans[1])) for trans in trans_outputs]
            mid_hyp_scores = [trans[4] for trans in trans_outputs]
            logger.info(str(k) +"-best translations: " + str(mid_hyps))#mid_hyps)
            logger.info("Scores: " + str(mid_hyp_scores))
            logger.info("Passed!")

            # create an input batch as input_iter
            agg_grads_s2t = None
            agg_grads_t2s = None
            for mid_hyp in mid_hyps:
                if len(mid_hyp) == 0: continue
                logger.info("DEBUG - 8d (learning loop) - create data batches")
                infer_input_s2t = _get_inputs(trans_input[2], tm_s2t.vocab_source, tm_s2t.buckets)
                infer_input_t2s = _get_inputs(mid_hyp, tm_t2s.vocab_source, tm_t2s.buckets) 
                input_batch_s2t = mx.io.DataBatch(data=[infer_input_s2t[0], infer_input_s2t[2], infer_input_t2s[0]], 
                                                  label=[infer_input_t2s[1]], # slice one position for label seq
                                                  bucket_key=(infer_input_s2t[3],infer_input_t2s[3]),
                                                  provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR),
                                                                mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                  provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                input_batch_t2s = mx.io.DataBatch(data=[infer_input_t2s[0], infer_input_t2s[2], infer_input_s2t[0]], 
                                                  label=[infer_input_s2t[1]], #  slice one position for label seq
                                                  bucket_key=(infer_input_t2s[3],infer_input_s2t[3]),
                                                  provide_data=[mx.io.DataDesc(name=C.SOURCE_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR),
                                                                mx.io.DataDesc(name=C.SOURCE_LENGTH_NAME, shape=(1,), layout=C.BATCH_MAJOR),
                                                                mx.io.DataDesc(name=C.TARGET_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)],
                                                  provide_label=[mx.io.DataDesc(name=C.TARGET_LABEL_NAME, shape=(1, infer_input_s2t[3]), layout=C.BATCH_MAJOR)])
                input_batch_mono = mx.io.DataBatch(data=[infer_input_t2s[0]], 
                                                   label=[infer_input_t2s[1]], #  slice one position for label seq
                                                   bucket_key=infer_input_t2s[3],
                                                   provide_data=[mx.io.DataDesc(name=C.MONO_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)],
                                                   provide_label=[mx.io.DataDesc(name=C.MONO_LABEL_NAME, shape=(1, infer_input_t2s[3]), layout=C.BATCH_MAJOR)])
                logger.info("Passed!")

                logger.info("DEBUG - 8e (learning loop) - computing rewards")
                # set the language-model reward for currently-sampled sentence from P(mid_hyp; mod_mlm_t)
                # 1) bind the data {(mid_hyp,mid_hyp)} into the model's module
                # 2) do forward step --> get the r1 = P(mid_hyp; mod_mlm_t)
                print("Computing reward_1")
                reward_1 = lm.compute_ll(input_batch_mono, metric_val)
                logger.info("reward_1=" + str(reward_1))
         
                # set the communication reward for currently-sampled sentence from P(sentA|mid_hype; mod_am_t2s)
                # do forward step --> get the r2 = P(sentA|mid_hyp; mod_am_t2s)
                print("Computing reward_2") 
                reward_2 = tm_t2s.models[0].compute_ll(input_batch_t2s, metric_val) # also includes forward step
                logger.info("reward_2=" + str(reward_2))

                # reward interpolation: r = alpha * r1 + (1 - alpha) * r2
                reward = grad_alphas[0] * reward_1 + (1.0 - grad_alphas[0]) * reward_2
                logger.info("total_reward=" + str(reward))
                logger.info("Passed!")
        
                # do forward step for s2t model - FIXME: this step is done twice (one for inference and one for computing loss). Better way? 
                tm_s2t.models[0].forward(input_batch_s2t)

                # do backward steps & collect gradients 
                logger.info("DEBUG - 8g (learning loop) - backward and collect gradients")
                agg_grads_s2t = tm_s2t.models[0].backward_and_collect_gradients(reward=-reward, # gradient ascent
                                                                                agg_grads=agg_grads_s2t)
                agg_grads_t2s = tm_t2s.models[0].backward_and_collect_gradients(reward=1.0 - grad_alphas[0], # gradient ascent
                                                                                agg_grads=agg_grads_t2s)
                logger.info("Passed!")
        
            if agg_grads_s2t != None and agg_grads_t2s != None:
                # update model parameters
                logger.info("DEBUG - 8h (learning loop) - update params for learning modules")
                tm_s2t.models[0].update_params(k=k, 
                                               agg_grads=agg_grads_s2t)
                tm_t2s.models[0].update_params(k=k, 
                                               agg_grads=agg_grads_t2s)
                logger.info("Passed!")
            
                # re-set the params for inference modules after each update of model parameters
                logger.info("DEBUG - 8i (learning loop) - update params for inference modules")
                tm_s2t.models[0].set_params_inference_modules()
                tm_t2s.models[0].set_params_inference_modules()
                logger.info("Passed!")

        if flag == False:   
            _process_sample(sent, 
                            dec_t2s, 
                            dec_s2t, 
                            models[3])
            r += 1 # next round       
            id_t += 1
        else:
            _process_sample(sent, 
                            dec_s2t, 
                            dec_t2s, 
                            models[2])
            id_s += 1
   
        # switch source and target roles
        flag = not flag
                                
        # testing over the development data (all_data[0] and all_data[1]) to check the improvements (after a desired number of rounds)
        if r == lmon[1]:
            # s2t model
            dev_pplx_s2t = dec_s2t.models[0].evaluate_dev(all_data[0], metric_val)

            # t2s model
            dev_pplx_t2s = dec_t2s.models[0].evaluate_dev(all_data[1], metric_val)

            # print the perplexities to the consoles
            logger.info("-------------------------------------------------------------------------")
            logger.info("Perplexity over development set from source-to-target model:" + str(dev_pplx_s2t))
            logger.info("Perplexity over development set from target-to-source model:" + str(dev_pplx_t2s))
            logger.info("-------------------------------------------------------------------------")

            # save the better model(s) if losses over dev are decreasing!
            if best_dev_pplx_s2t > dev_pplx_s2t:
                dec_s2t.models[0].save_params(model_folders[0], 1) # FIXME: add checkpoints
                best_dev_pplx_s2t = dev_pplx_s2t
            if best_dev_pplx_t2s > dev_pplx_t2s:
                dec_t2s.models[0].save_params(model_folders[1], 1) # FIXME: add checkpoints
                best_dev_pplx_t2s = dev_pplx_t2s

            r = 0 # reset the round

def main():
    # command line processing
    params = argparse.ArgumentParser(description='CLI for dual learning of sequence-to-sequence models.')
    arguments.add_device_args(params)
    arguments.add_dual_learning_args(params)
    args = params.parse_args()
    logger = setup_main_logger(__name__, file_logging=False, console=not args.quiet)

    # seed the RNGs
    logger.info("DEBUG - 1")
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)
    logger.info("Passed!")

    # checking status of output folder, resumption, etc.
    # create temporary logger to console only
    logger.info("DEBUG - 2")
    output_s2t_folder = os.path.abspath(args.output_s2t)
    _check_path(output_s2t_folder, logger, args.overwrite_output)
    output_t2s_folder = os.path.abspath(args.output_t2s)
    _check_path(output_t2s_folder, logger, args.overwrite_output)
    logger.info("Passed!")

    logger.info("DEBUG - 3")
    output_folder = os.path.abspath(args.output)
    _check_path(output_folder, logger, args.overwrite_output)
    logger = setup_main_logger(__name__, file_logging=True, console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)
    with open(os.path.join(output_s2t_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)
    with open(os.path.join(output_t2s_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)
    logger.info("Passed!")

    with ExitStack() as exit_stack:
        logger.info("DEBUG - 4")
        # get contexts (either in CPU or GPU)
        if args.use_cpu:
            logger.info("Device: CPU")
            #context = [mx.cpu()]
            context = mx.cpu()
        else:
            num_gpus = get_num_gpus()
            assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                             "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                             "binary isn't on the path)."
            assert len(args.device_ids) == 1, "cannot run on multiple devices for now"
            gpu_id = args.device_ids[0]
            if args.disable_device_locking:
                # without locking and a negative device id we just take the first device
                gpu_id = 0
            else:
                if gpu_id < 0:
                    # get a single (!) gpu id automatically:
                    gpu_ids = exit_stack.enter_context(acquire_gpus([-1], lock_dir=args.lock_dir))
                    gpu_id = gpu_ids[0]
            context = mx.gpu(gpu_id)
        logger.info("Passed!")

        # get model paths
        model_paths = args.models # [0]: s2t NMT; [1]: t2s NMT; [2]: s RNNLM; [3]: t RNNLM

        #--- load data
        # create vocabs on-the-fly
        logger.info("DEBUG - 5a")
        vocab_source = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_SRC_NAME))
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_s2t_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_t2s_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_target = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_TRG_NAME)) # assume all models use the same vocabularies!
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_s2t_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_t2s_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_source_size = len(vocab_source)
        vocab_target_size = len(vocab_target)
        logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)
        logger.info("Passed!")

        # parallel corpora
        logger.info("DEBUG - 5b")
        data_info = sockeye.data_io.DataInfo(os.path.abspath(args.source),
                                             os.path.abspath(args.target),
                                             os.path.abspath(args.validation_source),
                                             os.path.abspath(args.validation_target),
                                             '', #unused
                                             '')

        # create data iterators
        # important note: need to get buckets from train_iter and apply it to eval_iter and rev_eval_iter and other things within dual learning stuffs.
        train_iter, eval_iter = sockeye.data_io.get_training_data_iters(source=data_info.source,
                                                                        target=data_info.target,
                                                                        validation_source=data_info.validation_source,
                                                                        validation_target=data_info.validation_target,
                                                                        vocab_source=vocab_source,
                                                                        vocab_target=vocab_target,
                                                                        no_bos=False,
                                                                        batch_size=1, # FIXME: the following values are currently set manually!
                                                                        fill_up='replicate',
                                                                        max_seq_len_source=100,
                                                                        max_seq_len_target=100,
                                                                        bucketing=True,
                                                                        bucket_width=10)

        rev_train_iter, rev_eval_iter = sockeye.data_io.get_training_data_iters(source=data_info.target,
                                                                                target=data_info.source,
                                                                                validation_source=data_info.validation_target,
                                                                                validation_target=data_info.validation_source,
                                                                                vocab_source=vocab_target,
                                                                                vocab_target=vocab_source,
                                                                                no_bos=False,
                                                                                batch_size=1, # FIXME: the following values are currently set manually!
                                                                                fill_up='replicate',
                                                                                max_seq_len_source=100,
                                                                                max_seq_len_target=100,
                                                                                bucketing=True,
                                                                                bucket_width=10)
        
        # monolingual corpora
        # Note that monolingual source and target data may be different in sizes.
        # Assume that these monolingual corpora use the same vocabularies with parallel corpus,
        # otherwise, unknown words will be used in place of new words.
        src_mono_data = list(_read_lines(os.path.abspath(args.mono_source), 50)) # limit the sentence length to 50 
        trg_mono_data = list(_read_lines(os.path.abspath(args.mono_target), 50))
        logger.info("Passed!")

        # group all data
        # [0]: train data iter
        # [1]: validation data iter
        # [2]: reverse validation data iter
        # [3]: source mono data
        # [4]: target mono data
        all_data = (train_iter, rev_train_iter, 
                    eval_iter, rev_eval_iter, 
                    src_mono_data, trg_mono_data)
        
        #--- load models including:
        # [0]: source-to-target NMT model
        # [1]: target-to-source NMT model
        # [2]: source RNNLM model 
        # [3]: target RNNLM model
        logger.info("DEBUG - 6")
        models = []
        # NMT models
        logger.info("DEBUG - 6a")
        models.append(sockeye.dual_learning.TrainableInferenceModel(model_folder=model_paths[0],
                                                                    context=context, 
                                                                    fused=False,
                                                                    beam_size=args.beam_size,
                                                                    max_input_len=args.max_input_len)) # FIXME: can get bucketing info from train_iter
        models.append(sockeye.dual_learning.TrainableInferenceModel(model_folder=model_paths[1],
                                                                    context=context, 
                                                                    fused=False,
                                                                    beam_size=args.beam_size,
                                                                    max_input_len=args.max_input_len)) # FIXME: can get bucketing info from rev_train_iter
        
        # RNNLMs 
        logger.info("DEBUG - 6b")
        models.append(sockeye.dual_learning.InferenceLModel(model_folder=model_paths[2],
                                                            context=context))
        models.append(sockeye.dual_learning.InferenceLModel(model_folder=model_paths[3],
                                                            context=context))
        logger.info("Passed!")

        # learning rate scheduling
        logger.info("DEBUG - 7")
        learning_rate_half_life = None if args.learning_rate_half_life < 0 else args.learning_rate_half_life
        lr_scheduler = sockeye.lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                                             1000, # FIXME: checkpoint frequency, now manually set!
                                                             learning_rate_half_life,
                                                             args.learning_rate_reduce_factor,
                                                             args.learning_rate_reduce_num_not_improved)
        logger.info("Passed!")

        #--- execute dual-learning
        logger.info("DEBUG - 8 (_dual_learn)")
        '''
        _dual_learn(context=context,
                    vocab_source=vocab_source, vocab_target=vocab_target, 
                    all_data=all_data, 
                    models=models,
                    opt_configs=(args.optimizer, args.weight_decay, args.momentum, args.clip_gradient, lr_scheduler), # optimizer-related stuffs
                    grad_alphas=(args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s), # hyper-parameters for gradient updates
                    lmon=(args.epoch, args.dev_round), # extra stuffs for learning monitor
                    model_folders=(output_s2t_folder, output_t2s_folder), # output folders where the model files will live in!
                    k=args.k_best) # K in K-best translation
        '''
        '''
        _dual_learn_batch(context=context,
                    vocab_source=vocab_source, vocab_target=vocab_target, 
                    all_data=all_data, 
                    models=models,
                    opt_configs=(args.optimizer, args.weight_decay, args.momentum, args.clip_gradient, lr_scheduler), # optimizer-related stuffs
                    grad_alphas=(args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s), # hyper-parameters for gradient updates
                    lmon=(args.epoch, args.dev_round), # extra stuffs for learning monitor
                    model_folders=(output_s2t_folder, output_t2s_folder), # output folders where the model files will live in!
                    k=args.k_best,
                    minibatch_size=args.minibatch_size) # K in K-best translation
        '''
        _dual_learn_batch_soft_landing(context=context,
                    vocab_source=vocab_source, vocab_target=vocab_target, 
                    all_data=all_data, 
                    models=models,
                    opt_configs=(args.optimizer, args.weight_decay, args.momentum, args.clip_gradient, lr_scheduler), # optimizer-related stuffs
                    grad_alphas=(args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s), # hyper-parameters for gradient updates
                    lmon=(args.epoch, args.dev_round), # extra stuffs for learning monitor
                    model_folders=(output_s2t_folder, output_t2s_folder), # output folders where the model files will live in!
                    k=args.k_best,
                    minibatch_size=args.minibatch_size) # K in K-best translation
        logger.info("Passed!")

    #--- bye bye message
    print("Dual learning completed!")

if __name__ == "__main__":
    main()

