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
from typing import Optional, Dict, List, Tuple

import mxnet as mx
import numpy as np

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
import sockeye.dual_learning
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, expand_requested_device_ids

logger = logging.getLogger(__name__)

def _get_data_iters_from_lists(buckets: List[Tuple[int, int]],
                   source_sentences: List[List[int]], 
                   target_sentences: List[List[int]],
                   vocab_source: Dict[str, int], 
                   vocab_target: Dict[str, int],
                   batch_size: int,
                   fill_up: str,
                   max_seq_len: int,
                   bucketing: bool,
                   bucket_width: int) -> 'ParallelBucketSentenceIter':
    """
    Returns data iterators for data.
    
    :param source: list of source sentences.
    :param target: list of target sentences.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param batch_size: Batch size.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len: Maximum sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: data iterator.
    """
    data_iter = sockeye.data_io.ParallelBucketSentenceIter(source_sentences,
                                           target_sentences,
                                           buckets,
                                           batch_size,
                                           vocab_target[C.EOS_SYMBOL],
                                           C.PAD_ID,
                                           vocab_target[C.UNK_SYMBOL],
                                           fill_up=fill_up)

    return data_iter

def _get_training_data_iters(source: str, target: str,
                            validation_source: str, validation_target: str,
                            vocab_source: Dict[str, int], vocab_target: Dict[str, int],
                            batch_size: int,
                            fill_up: str,
                            max_seq_len: int,
                            bucketing: bool,
                            bucket_width: int) -> Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', List[Tuple[int, int]]]:
    """
    Returns data iterators for training and validation data.

    :param source: Path to source training data.
    :param target: Path to target training data.
    :param validation_source: Path to source validation data.
    :param validation_target: Path to target validation data.
    :param vocab_source: Source vocabulary.
    :param vocab_target: Target vocabulary.
    :param batch_size: Batch size.
    :param fill_up: Fill-up strategy for buckets.
    :param max_seq_len: Maximum sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :return: Tuple of (training data iterator, validation data iterator).
    """
    logger.info("Creating train data iterator")
    train_source_sentences, train_target_sentences = sockeye.data_io.read_parallel_corpus(source,
                                                                                          target,
                                                                                          vocab_source,
                                                                                          vocab_target)
    length_ratio = sum(len(t) / float(len(s)) for t, s in zip(train_source_sentences, train_target_sentences)) / len(
        train_target_sentences)
    logger.info("Average training target/source length ratio: %.2f", length_ratio)

    # define buckets
    buckets = sockeye.data_io.define_parallel_buckets(max_seq_len, bucket_width, length_ratio) if bucketing else [
        (max_seq_len, max_seq_len)]

    train_iter = sockeye.data_io.ParallelBucketSentenceIter(train_source_sentences,
                                                            train_target_sentences,
                                                            buckets,
                                                            batch_size,
                                                            vocab_target[C.EOS_SYMBOL],
                                                            C.PAD_ID,
                                                            vocab_target[C.UNK_SYMBOL],
                                                            fill_up=fill_up)

    logger.info("Creating validation data iterator")
    val_source_sentences, val_target_sentences = sockeye.data_io.read_parallel_corpus(validation_source,
                                                                                      validation_target,
                                                                                      vocab_source,
                                                                                      vocab_target)
    val_iter = sockeye.data_io.ParallelBucketSentenceIter(val_source_sentences,
                                                          val_target_sentences,
                                                          buckets,
                                                          batch_size,
                                                          vocab_target[C.EOS_SYMBOL],
                                                          C.PAD_ID,
                                                          vocab_target[C.UNK_SYMBOL],
                                                          fill_up=fill_up)
    return train_iter, val_iter, buckets

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

def _dual_learn(context: mx.context.Context, 
                vocab_source: Dict[str, int],
                vocab_target: Dict[str, int],
                all_data: Tuple['ParallelBucketSentenceIter', 'ParallelBucketSentenceIter', List[List[int]], List[List[int]]], 
                models: List[sockeye.dual_learning.TrainableInferenceModel], 
                opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                lmon: Tuple[int, int], # extra stuffs for learning monitor
                model_folders: Tuple[str, str],
                k: int,
                buckets: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]):
    # set up decoders/translators
    print("DEBUG 8a")
    dec_s2t = sockeye.inference.Translator(context,
                                           "linear", #unused
                                           [models[0]], vocab_source, vocab_target)
    dec_t2s = sockeye.inference.Translator(context,
                                           "linear", #unused
                                           [models[1]], vocab_target, vocab_source)
    dec_s = sockeye.inference.Translator(context,
                                         "linear", #unused
                                         [models[2]], vocab_source, vocab_source)
    dec_t = sockeye.inference.Translator(context,
                                         "linear", #unused
                                         [models[3]], vocab_target, vocab_target)
    print("Passed!")

    # set up monolingual data access/ids
    print("DEBUG 8b")
    orders_s = list(range(len(all_data[2])))
    orders_t = list(range(len(all_data[3])))
    np.random.shuffle(orders_s)
    np.random.shuffle(orders_t)
    print("Passed!")

    # set up optimizers
    print("DEBUG 8c")
    dec_s2t.models[0].setup_optimizer(initial_learning_rate=grad_alphas[1], opt_configs=opt_configs)
    dec_t2s.models[0].setup_optimizer(initial_learning_rate=grad_alphas[2], opt_configs=opt_configs)
    print("Passed!")

    # create eval metric
    metric_val = mx.metric.create([mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME])])

    # pointers for switching between the models 
    # including: p_dec_s2t, p_dec_t2s, p_dec (dynamically changed within the loop)
 
    # start the dual learning algorithm
    print("DEBUG 8d (learning loop)")
    best_dev_loss_s2t = 9e+99
    best_dev_loss_t2s = 9e+99
    id_s = 0
    id_t = 0
    r = 0 # learning round
    e_s = 0
    e_t = 0
    flag = True # role of source and target
    while e_s < lmon[0] or e_t < lmon[0]: 
        if id_s == len(orders_s): # source monolingual data
            # shuffle the data
            np.random.shuffle(orders_s)
            
            # update epochs
            e_s += 1
            
            id_s = 0
        if id_t == len(orders_t): # target monoingual data
            # shuffle the data
            np.random.shuffle(orders_t)
                    
            # update epochs
            e_t += 1
            
            id_t = 0

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_t respectively
        sent = ""
        if flag:
            sent = all_data[2][orders_s[id_s]]

            # switch the pointers
            p_dec_s2t = dec_s2t
            p_dec_t2s = dec_t2s
            p_dec = dec_t
        else:
            sent = all_data[3][orders_t[id_t]]

            # switch the pointers
            p_dec_s2t = dec_t2s
            p_dec_t2s = dec_s2t
            p_dec = dec_s

        print("Sampled sentence: ", sent)
        stokens = sent.split()
        s_sents = [sockeye.data_io.tokens2ids(stokens, vocab_source)] * k # FIXME: if sent is too long, reject it instead!

        # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
        print("DEBUG 8d (learning loop) - K-best translation")
        trans_input = p_dec_s2t.make_input(0, sent) # 0: unused!
        trans_outputs = p_dec_s2t.translate_kbest(trans_input, k) # generate k-best translations
        mid_hyps = [sockeye.data_io.tokens2ids(trans[2], vocab_target) for trans in trans_outputs] 
        print("Passed!")

        # create an input batch as input_iter
        print("DEBUG 8d (learning loop) - data iters")
        input_iter_m = _get_data_iters_from_lists(buckets[1], source_sentences=mid_hyps,
                                                  target_sentences=mid_hyps,
                                                  vocab_source=vocab_target,
                                                  vocab_target=vocab_target,
                                                  batch_size=1,
                                                  fill_up='replicate',
                                                  max_seq_len=100,
                                                  bucketing=True,
                                                  bucket_width=10)
        input_iter_s2t = _get_data_iters_from_lists(buckets[0], source_sentences=s_sents,
                                                    target_sentences=mid_hyps,
                                                    vocab_source=vocab_source,
                                                    vocab_target=vocab_target,
                                                    batch_size=1,
                                                    fill_up='replicate',
                                                    max_seq_len=100,
                                                    bucketing=True,
                                                    bucket_width=10) 
        input_iter_t2s = _get_data_iters_from_lists(buckets[1], source_sentences=mid_hyps,
                                                    target_sentences=s_sents,
                                                    vocab_source=vocab_target,
                                                    vocab_target=vocab_source,
                                                    batch_size=1,
                                                    fill_up='replicate',
                                                    max_seq_len=100,
                                                    bucketing=True,
                                                    bucket_width=10)
        print("Passed!")

        print("DEBUG 8e (learning loop) - computing rewards")
        # set the language-model reward for currently-sampled sentence from P(mid_hyp; mod_mlm_t)
        # 1) bind the data {(mid_hyp,mid_hyp)} into the model's module
        # 2) do forward step --> get the r1 = P(mid_hyp; mod_mlm_t)
        print("reward_1")
        reward_1 = p_dec.models[0].compute_ll(input_iter_m, metric_val)
        print("reward_1=", reward_1)

        # set the communication reward for currently-sampled sentence from P(sentA|mid_hype; mod_am_t2s)
        # 1) bind the data {(mid_hyp, sentA)} into the model's module
        # 2) do forward step --> get the r2 = P(sentA|mid_hyp; mod_am_t2s)
        print("reward_2")
        reward_2a = p_dec_s2t.models[0].compute_ll(input_iter_s2t, metric_val)
        print("reward_2a=", reward_2a)
        reward_2b = p_dec_t2s.models[0].compute_ll(input_iter_t2s, metric_val)
        print("reward_2b=", reward_2b)

        # reward interpolation: r = alpha * r1 + (1 - alpha) * r2
        reward = grad_alphas[0] * reward_1 + (1.0 - grad_alphas[0]) * reward_2b
        print("Passed!")
        
        # do backward steps and update model parameters
        print("DEBUG 8f (learning loop) - re-update model parameters")
        p_dec_s2t.models[0].update_params(reward)
        p_dec_t2s.models[0].update_params(1.0 - grad_alphas[0])
        print("Passed!")

        if flag == False: r += 1 # next round
        
        # switch source and target roles
        flag = -flag
        vocab_source, vocab_target = vocab_target, vocab_source # FIXME: is this way fast and correct?
        buckets[0], buckets[1] = buckets[1], buckets[0]
        
        # testing over the development data (all_data[0] and all_data[1]) to check the improvements (after a desired number of rounds)
        if r == opt_configs[1]:
            # s2t model
            dev_loss_s2t = p_dec_s2t.models[0].evaluate_dev(all_data[0], metric_val)

            # t2s model
            dev_loss_t2s = p_dec_t2s.models[0].evaluate_dev(all_data[1], metric_val)

            # print the losses to the consoles
            print("-------------------------------------------------------------------------")
            print("Loss over development set from source-to-target model: %f", dev_loss_s2t)
            print("Loss over development set from target-to-source model: %f", dev_loss_t2s)
            print("-------------------------------------------------------------------------")

            # save the better model(s) if losses over dev are decreasing!
            if best_dev_loss_s2t > dev_loss_s2t:
                p_dec_s2t.models[0].save_params(model_folders[0], 1) # FIXME: add checkpoints
                best_dev_loss_s2t = dev_loss_s2t
            if best_dev_loss_t2s > dev_loss_t2s:
                p_dec_t2s.models[0].save_params(model_folders[1], 1) # FIXME: add checkpoints
                best_dev_loss_t2s = dev_loss_t2s

            r = 0 # reset the round

def main():
    # command line processing
    print("DEBUG 1")
    params = argparse.ArgumentParser(description='CLI for dual learning of sequence-to-sequence models.')
    arguments.add_device_args(params)
    arguments.add_dual_learning_args(params)
    args = params.parse_args()
    print("Passed!")

    # seed the RNGs
    print("DEBUG 2")
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)
    print("Passed!")

    # checking status of output folder, resumption, etc.
    # create temporary logger to console only
    print("DEBUG 3")
    logger = setup_main_logger(__name__, file_logging=False, console=not args.quiet)
    output_s2t_folder = os.path.abspath(args.output_s2t)
    _check_path(output_s2t_folder, logger, args.overwrite_output)
    output_t2s_folder = os.path.abspath(args.output_t2s)
    _check_path(output_t2s_folder, logger, args.overwrite_output)
    print("Passed!")

    print("DEBUG 4")
    output_folder = os.path.abspath(args.output)
    _check_path(output_folder, logger, args.overwrite_output)
    logger = setup_main_logger(__name__, file_logging=True, console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)
    with open(os.path.join(output_s2t_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)
    with open(os.path.join(output_t2s_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)
    print("Passed!")

    with ExitStack() as exit_stack:
        print("DEBUG 5a")
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
        print("Passed!")

        # get model paths
        model_paths = args.models # [0]: s2t NMT; [1]: t2s NMT; [2]: s RNNLM; [3]: t RNNLM

        #--- load data
        # create vocabs on-the-fly
        print("DEBUG 5b")
        vocab_source = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_SRC_NAME))
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_s2t_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_t2s_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_target = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_TRG_NAME)) # assume all models use the same vocabularies!
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_s2t_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_t2s_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_source_size = len(vocab_source)
        vocab_target_size = len(vocab_target)
        logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)
        print("Passed!")

        # parallel corpora
        print("DEBUG 5c")
        data_info = sockeye.data_io.DataInfo(os.path.abspath(args.source),
                                             os.path.abspath(args.target),
                                             os.path.abspath(args.validation_source),
                                             os.path.abspath(args.validation_target),
                                             '', #unused
                                             '')

        # create data iterators
        # important note: need to get buckets from train_iter and apply it to eval_iter and rev_eval_iter and other things within dual learning stuffs.
        train_iter, eval_iter, buckets = _get_training_data_iters(source=data_info.source,
                                                                        target=data_info.target,
                                                                        validation_source=data_info.validation_source,
                                                                        validation_target=data_info.validation_target,
                                                                        vocab_source=vocab_source,
                                                                        vocab_target=vocab_target,
                                                                        batch_size=64, # FIXME: the following values are currently set manually!
                                                                        fill_up='replicate',
                                                                        max_seq_len=100,
                                                                        bucketing=True,
                                                                        bucket_width=10)

        rev_train_iter, rev_eval_iter, rbuckets = _get_training_data_iters(source=data_info.target,
                                                                           target=data_info.source,
                                                                           validation_source=data_info.validation_target,
                                                                           validation_target=data_info.validation_source,
                                                                           vocab_source=vocab_target,
                                                                           vocab_target=vocab_source,
                                                                           batch_size=64, # FIXME: the following values are currently set manually!
                                                                           fill_up='replicate',
                                                                           max_seq_len=100,
                                                                           bucketing=True,
                                                                           bucket_width=10)
        
        # monolingual corpora
        # Note that monolingual source and target data may be different in sizes.
        # Assume that these monolingual corpora use the same vocabularies with parallel corpus,
        # otherwise, unknown words will be used in place of new words.
        src_mono_data = sockeye.data_io.read_lines(os.path.abspath(args.mono_source)) # FIXME: how to do create a batch of these data?
        trg_mono_data = sockeye.data_io.read_lines(os.path.abspath(args.mono_target))
        print("Passed!")

        # group all data
        # [0]: train data iter
        # [1]: validation data iter
        # [2]: reverse validation data iter
        # [3]: source mono data
        # [4]: target mono data
        all_data = (eval_iter, rev_eval_iter, src_mono_data, trg_mono_data)
        
        #--- load models including:
        # [0]: source-to-target NMT model
        # [1]: target-to-source NMT model
        # [2]: source RNNLM model 
        # [3]: target RNNLM model
        # Vu's N.b. (as of 04 July 2017): I will use attentional auto-encoder in place of RNNLM model (which is not available for now in Sockeye).
        print("DEBUG 6")
        models = []
        for ip, model_path in enumerate(model_paths):
            models.append(sockeye.dual_learning.TrainableInferenceModel(model_folder=model_path,
                                                                        context=context,
                                                                        train_iter=train_iter if ip % 2 == 0 else rev_train_iter, # FIXME: how about the monolingual RNNLM?
                                                                        fused=False,
                                                                        beam_size=args.beam_size,
                                                                        max_input_len=args.max_input_len))
        print("Passed!")

        # learning rate scheduling
        print("DEBUG 7")
        learning_rate_half_life = None if args.learning_rate_half_life < 0 else args.learning_rate_half_life
        lr_scheduler = sockeye.lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                                             1000, # FIXME: checkpoint frequency, now manually set!
                                                             learning_rate_half_life,
                                                             args.learning_rate_reduce_factor,
                                                             args.learning_rate_reduce_num_not_improved)
        print("Passed!")

        #--- execute dual-learning
        print("DEBUG 8 (_dual_learn)")
        _dual_learn(context,
                    vocab_source, vocab_target, 
                    all_data, 
                    models, 
                    (args.optimizer, args.weight_decay, args.momentum, args.clip_gradient, lr_scheduler), # optimizer-related stuffs
                    (args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s), # hyper-parameters for gradient updates
                    (args.epoch, args.dev_round), # extra stuffs for learning monitor
                    (output_s2t_folder, output_t2s_folder), # output folders where the model files will live in!
                    args.k_best, # K in K-best translation
                    (buckets, rbuckets))
        print("Passed!")

    #--- bye bye message
    print("Dual learning completed!")

if __name__ == "__main__":
    main()
