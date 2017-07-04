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

def _dual_learn(all_data, 
                models, 
                opt_configs, # optimizer-related stuffs
                grad_alphas # hyper-parameters for gradient updates
                lmon, # extra stuffs for learning monitor
                k):
    # set up decoders/translators
    # FIXME

    # set up monolingual data
    # FIXME
    
    # set up optimizers
    # FIXME

    # create pointers for switching between the models
    # FIXME

    # start the dual learning algorithm
    best_loss_s2t = 9e+99
    best_loss_t2s = 9e+99
    id_s = 0
    id_t = 0
    r = 0 #round
    flag = True # role of source and target
    while True: # stopping criterion will be imposed within the loop
        if id_s == len(orders_s): # source monolingual data
            # shuffle the data
            # FIXME: update epochs?
            id_s = 0
        if id_t == len(orders_t): # target monoingual data
            #shuffle the data
            # FIXME: update epochs?
            id_t = 0

        # sample sentence sentA and sentB from mono_cor_s and mono_cor_s respectively
        sent = ""
        if flag:
            sent = all_data[4][orders_s[id_s]]
            # switch the pointers
            # FIXME
        else:
            sent = all_data[5][orders_t[id_t]]
            # switch the pointers
            # FIXME

        # generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t)
        # FIXME
        mid_hyps = [] # generate k-best translation
        for mid_hyp in mid_hyps:
            # set the language-model reward for current sampled sentence from p_mlm_t
            # FIXME

            # set the communication reward for current sampled sentence from p_am_t2s
            # FIXME

            # reward interpolation
            # FIXME
        
        # make total customized losses
        # FIXME

        # execute forward step
        # FIXME

        # execute backward step 
        # FIXME

        # update parameters
        # FIXME

        # switch source and target roles
        flag = !flag

        if id_s == id_t: r += 1 # FIXME: perhaps a bug due to different sizes of monolingual data? Idea: check flag instead?

        # testing over the development data to check the improvements (after a desired number of rounds)
        if r == opt_configs[1]:
            # s2t model
            # FIXME

            # t2s model
            # FIXME

            r = 0

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
        src_train_data, trg_train_data = sockeye.data_io.read_parallel_corpus(os.path.abspath(args.source),
                                                                              os.path.abspath(args.target),
                                                                              args.source_vocab,
                                                                              args.target_vocab)
        src_val_data, trg_val_data = sockeye.data_io.read_parallel_corpus(os.path.abspath(args.validation_source),
                                                                          os.path.abspath(args.validation_target),
                                                                          args.source_vocab,
                                                                          args.target_vocab)
        
        # monolingual corpora
        # Note that monolingual source and target data may be different in sizes.
        # Assume that these monolingual corpora use the same vocabularies with parallel corpus,
        # otherwise, unknown words will be used in place of new words.
        src_mono_data = sockeye.data_io.read_sentences(os.path.abspath(args.mono_source), args.source_vocab) 
        trg_mono_data = sockeye.data_io.read_sentences(os.path.abspath(args.mono_target), args.target_vocab)

        # group all data
        all_data = [src_train_data, trg_train_data, src_val_data, trg_val_data, src_mono_data, trg_mono_data]
        
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
                          fused=False,
                          beam_size=args.beam_size)

        #--- execute dual-learning
        _dual_learn(all_data, 
                    models, 
                    (args.optimizer, args.weight_decay, args.momentum, args.clip_gradient), # optimizer-related stuffs
                    (args.alpha, args.initial_lr_gamma_s2t, args.initial_lr_gamma_t2s) # hyper-parameters for gradient updates
                    (args.epochs, args.dev_round), # extra stuffs for learning monitor
                    args.K # K in K-best translation
                    )

    #--- bye bye message
    print("Dual learning completed!")

if __name__ == "__main__":
    main()

