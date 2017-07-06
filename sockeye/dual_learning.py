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
from contextlib import ExitStack
from typing import Optional, Dict

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
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, expand_requested_device_ids

def _check_path(opath, logger):
    training_state_dir = os.path.join(opath, C.TRAINING_STATE_DIRNAME)
    if os.path.exists(opath):
        if args.overwrite_output:
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
                all_data: Tuple[sockeye.data_io.ParallelBucketSentenceIter, sockeye.data_io.ParallelBucketSentenceIter, sockeye.data_io.ParallelBucketSentenceIter, List[List[int]], List[List[int]]], 
                models: List[TrainableInferenceModel], 
                opt_configs: Tuple[str, float, float, float, sockeye.lr_scheduler.LearningRateScheduler], # optimizer-related stuffs
                grad_alphas: Tuple[float, float, float], # hyper-parameters for gradient updates
                lmon: Tuple[int, int], # extra stuffs for learning monitor
                model_folders: Tuple[str, str],
                k: int):
    # set up decoders/translators
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

    # set up monolingual data access/ids
    orders_s = range(len(all_data[3]))
    orders_t = range(len(all_data[4]))
    random.shuffle(orders_s)
    random.shuffle(orders_t)

    # set up optimizers
    models[0].setup_optimizer(opt_configs)
    models[1].setup_optimizer(opt_configs)

    # pointers for switching between the models 
    # including: p_dec_s2t, p_dec_t2s, p_dec (dynamically changed within the loop)
 
    # start the dual learning algorithm
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
            random.shuffle(orders_s)
            
            # update epochs
            e_s += 1
            
            id_s = 0
        if id_t == len(orders_t): # target monoingual data
            # shuffle the data
            random.shuffle(orders_t)
                    
            # update epochs
            e_t += 1
            
            id_t = 0

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_t respectively
        sent = ""
        if flag:
            sent = all_data[3][orders_s[id_s]]

            # switch the pointers
            p_dec_s2t = dec_s2t
            p_dec_t2s = dec_t2s
            p_dec = dec_t
        else:
            sent = all_data[4][orders_t[id_t]]

            # switch the pointers
            p_dec_s2t = dec_t2s
            p_dec_t2s = dec_s2t
            p_dec = dec_s

        # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
        trans_input = p_dec_s2t.make_input(0, sent) # 0: unused!
        mid_hyps = p_dec_s2t.translate_kbest(trans_input, k) # generate k-best translation
        s_sents = [sent] * k
        # create an input batch as input_iter
        input_iter_m = sockeye.data_io.get_data_iters(source=mid_hyps,
                                                        target=mid_hyps,
                                                        vocab_source=vocab_target,
                                                        vocab_target=vocab_target,
                                                        batch_size=1,
                                                        fill_up='replicate',
                                                        max_seq_len=100,
                                                        bucketing=False,
                                                        bucket_width=1) # unused
        input_iter_s2t = sockeye.data_io.get_data_iters(source=s_sents,
                                                        target=mid_hyps,
                                                        vocab_source=vocab_source,
                                                        vocab_target=vocab_target,
                                                        batch_size=1,
                                                        fill_up='replicate',
                                                        max_seq_len=100,
                                                        bucketing=False,
                                                        bucket_width=1) # unused
        input_iter_t2s = sockeye.data_io.get_data_iters(source=mid_hyps,
                                                        target=s_sents,
                                                        vocab_source=vocab_target,
                                                        vocab_target=vocab_source,
                                                        batch_size=1,
                                                        fill_up='replicate',
                                                        max_seq_len=100,
                                                        bucketing=False,
                                                        bucket_width=1) # unused

        # set the language-model reward for currently-sampled sentence from P(mid_hyp; mod_mlm_t)
        # 1) bind the data {(mid_hyp,mid_hyp)} into the model's module
        # 2) do forward step --> get the r1 = P(mid_hyp; mod_mlm_t)
        reward_1 = p_dec.models[0].compute_ll(input_iter_m)

        # set the communication reward for currently-sampled sentence from P(sentA|mid_hype; mod_am_t2s)
        # 1) bind the data {(mid_hyp, sentA)} into the model's module
        # 2) do forward step --> get the r2 = P(sentA|mid_hyp; mod_am_t2s)
        reward_2 = p_dec_t2s.models[0].compute_ll(input_iter_t2s)

        # reward interpolation: r = alpha * r1 + (1 - alpha) * r2
        reward = grad_alphas[0] * reward_1 + (1.0 - grad_alphas[0]) * reward_2
        
        # update model parameters
        p_dec_s2t.models[0].compute_ll(input_iter_s2t)
        p_dec_s2t.models[0].update_model(reward)
        p_dec_t2s.models[0].update_model(1.0 - grad_alphas[0])

        if flag == False: r += 1 # next round
        
        # switch source and target roles
        flag = !flag
        
        # testing over the development data (all_data[1] and all_data[2] to check the improvements (after a desired number of rounds))
        if r == opt_configs[1]:
            # s2t model
            dev_loss_s2t = p_dec_s2t.models[0].evaluate_dev(all_data[1])

            # t2s model
            dev_loss_t2s = p_dec_t2s.models[0].evaluate_dev(all_data[2])

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
    params = argparse.ArgumentParser(description='CLI for dual learning of sequence-to-sequence models.')
    arguments.add_dual_learning_args(params)
    arguments.add_device_args(params)
    args = params.parse_args()

    # seed the RNGs
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # checking status of output folder, resumption, etc.
    # create temporary logger to console only
    logger = setup_main_logger(__name__, file_logging=False, console=not args.quiet)
    output_s2t_folder = os.path.abspath(args.output_s2t)
    _check_path(output_s2t_folder, logger)
    output_t2s_folder = os.path.abspath(args.output_t2s)
    _check_path(output_t2s_folder, logger)

    logger = setup_main_logger(__name__, file_logging=True, console=not args.quiet, path=os.path.join(os.path.abspath(args.output), C.LOG_NAME))
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)
    with open(os.path.join(output_s2t_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)
    with open(os.path.join(output_t2s_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)

    with ExitStack() as exit_stack:
        # get contexts (either in CPU or GPU)
        if args.use_cpu:
            logger.info("Device: CPU")
            context = [mx.cpu()]
        else:
            num_gpus = get_num_gpus()
            assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                                 "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                                 "binary isn't on the path)."
            if args.disable_device_locking:
                context = expand_requested_device_ids(args.device_ids)
            else:
                context = exit_stack.enter_context(acquire_gpus(args.device_ids, lock_dir=args.lock_dir))
            logger.info("Device(s): GPU %s", context)
            context = [mx.gpu(gpu_id) for gpu_id in context]

        # get model paths
        model_paths = args.models # [0]: s2t NMT; [1]: t2s NMT; [2]: s RNNLM; [3]: t RNNLM

        #--- load data
        # create vocabs on-the-fly
        vocab_source = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_SRC_NAME))
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_s2t_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_source, os.path.join(output_t2s_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_target = sockeye.vocab.vocab_from_json_or_pickle(os.path.join(model_paths[0], C.VOCAB_TRG_NAME)) # assume all models use the same vocabularies!
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_s2t_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder
        sockeye.vocab.vocab_to_json(vocab_target, os.path.join(output_t2s_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX) # dump vocab json file into new output folder

        vocab_source_size = len(vocab_source)
        vocab_target_size = len(vocab_target)
        logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)

        # parallel corpora
        data_info = sockeye.data_io.DataInfo(os.path.abspath(args.source),
                                             os.path.abspath(args.target),
                                             os.path.abspath(args.validation_source),
                                             os.path.abspath(args.validation_target),
                                             args.source_vocab,
                                             args.target_vocab)

        # create data iterators
        train_iter, eval_iter = sockeye.data_io.get_training_data_iters(source=data_info.source,
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

        rev_eval_iter = sockeye.data_io.get_data_iters(source=data_info.validation_target,
                                                       target=data_info.validation_source,
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
        src_mono_data = sockeye.data_io.read_sentences(os.path.abspath(args.mono_source), args.source_vocab) # FIXME: how to do create a batch of these data?
        trg_mono_data = sockeye.data_io.read_sentences(os.path.abspath(args.mono_target), args.target_vocab)

        # group all data
        all_data = (train_iter, eval_iter, rev_eval_iter, src_mono_data, trg_mono_data)
        
        #--- load models including:
        # [0]: source-to-target NMT model
        # [1]: target-to-source NMT model
        # [2]: source RNNLM model 
        # [3]: target RNNLM model
        # Vu's N.b. (as of 04 July 2017): I will use attentional auto-encoder in place of RNNLM model (which is not available for now in Sockeye).
        models = []
        for model_path in model_paths:
            models.append(TrainableInferenceModel(model_folder=model_path,
                          context=context,
                          train_iter=train_iter,
                          fused=False,
                          beam_size=args.beam_size))

        # learning rate scheduling
        learning_rate_half_life = none_if_negative(args.learning_rate_half_life)
        lr_scheduler = sockeye.lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type,
                                                             args.checkpoint_frequency,
                                                             learning_rate_half_life,
                                                             args.learning_rate_reduce_factor,
                                                             args.learning_rate_reduce_num_not_improved)

        #--- execute dual-learning
        _dual_learn(context,
                    vocab_source, vocab_target, 
                    all_data, 
                    models, 
                    (args.optimizer, args.weight_decay, args.momentum, args.clip_gradient, lr_scheduler), # optimizer-related stuffs
                    (args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s) # hyper-parameters for gradient updates
                    (args.epochs, args.dev_round), # extra stuffs for learning monitor
                    (output_s2t_folder, output_t2s_folder) # output folders where the model files will live in!
                    args.K # K in K-best translation
                    )

    #--- bye bye message
    print("Dual learning completed!")

if __name__ == "__main__":
    main()

