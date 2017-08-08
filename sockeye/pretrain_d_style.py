import os
import sys
import json
import pickle
import random
import shutil
import argparse
from contextlib import ExitStack
from typing import Dict

import mxnet as mx
import numpy as np

import sockeye.data_io
import sockeye.style_training
import sockeye.style_pretraining_G
import sockeye.inference
import sockeye.encoder
import sockeye.constants as C
import sockeye.arguments as arguments
from sockeye.log import setup_main_logger

from sockeye.train import _build_or_load_vocab
from sockeye.utils import acquire_gpus, get_num_gpus, expand_requested_device_ids


def none_if_negative(val):
    return None if val < 0 else val


def _dict_difference(dict1: Dict, dict2: Dict):
    diffs = set()
    for k, v in dict1.items():
        if k not in dict2 or dict2[k] != v:
            diffs.add(k)
    return diffs


def make_iters_same_length(iter1: sockeye.data_io.ParallelBucketSentenceIter,
                           iter2: sockeye.data_io.ParallelBucketSentenceIter,
                           logger):
    """
    Takes two iters and ensures that they have the same number of batches (per epoch)
    This is necessary to make ParallelFetchingIter work since its behavior is
    weird when the two iters are not the same size.
    Equivalent to down-sampling
    TODO: Implement up-sampling

    :param iter1: A data iterator
    :param iter2: Another data iterator
    """
    max_idx_len = min(len(iter1.idx), len(iter2.idx))
    iter1.downsampling = True
    iter1.downsampling_threshold = max_idx_len
    iter2.downsampling = True
    iter2.downsampling_threshold = max_idx_len
    logger.info("Set downsampling threshold to %d for iters.", max_idx_len)

def main():
    params = argparse.ArgumentParser(description='CLI to train sockeye style transfer models.')
    arguments.add_io_args(params)
    arguments.add_model_parameters(params)
    arguments.add_training_args(params)
    arguments.add_device_args(params)
    args = params.parse_args()

    # seed the RNGs
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    # Temporary logger
    logger = setup_main_logger(__name__, file_logging=False, console=not args.quiet)
    output_folder = os.path.abspath(args.output)

    resume_training = False

    output_folder_orig = output_folder
    output_folder = os.path.join(output_folder, "d_pretrain")

    training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)

    if os.path.exists(output_folder):
        if args.overwrite_output:
            logger.info("Removing existing output folder %s.", output_folder)
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        elif os.path.exists(training_state_dir):
            with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "r") as fp:
                old_args = json.load(fp)
            arg_diffs = _dict_difference(vars(args), old_args) | _dict_difference(old_args, vars(args))
            # Remove args that may differ without affecting the training.
            arg_diffs -= set(C.ARGS_MAY_DIFFER)
            if not arg_diffs:
                resume_training = True
            else:
                # We do not have the logger yet
                logger.error("Mismatch in arguments for training continuation.")
                logger.error("Differing arguments: %s.", ", ".join(arg_diffs))
                sys.exit(1)
        else:
            logger.error("Refusing to overwrite existing output folder %s.", output_folder)
            sys.exit(1)
    else:
        os.makedirs(output_folder)

    logger = setup_main_logger(__name__,
                               file_logging=True,
                               console=not args.quiet, path=os.path.join(output_folder, C.LOG_NAME))
    logger.info("Sockeye version %s", sockeye.__version__)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Arguments: %s", args)
    with open(os.path.join(output_folder, C.ARGS_STATE_NAME), "w") as fp:
        json.dump(vars(args), fp)

    e_corpus = args.source
    f_corpus = args.target
    e_mono_corpus = args.mono_source
    f_mono_corpus = args.mono_target
    e_val = args.validation_source
    f_val = args.validation_target
    external_vocab = args.joint_vocab

    num_embed = args.num_embed
    attention_num_hidden = args.rnn_num_hidden if not args.attention_num_hidden else args.attention_num_hidden
    attention_type = args.attention_type
    rnn_cell_type = args.rnn_cell_type
    rnn_num_layers = args.rnn_num_layers
    rnn_num_hidden = args.rnn_num_hidden
    dropout = args.dropout
    num_words = args.num_words
    word_min_count = args.word_min_count
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    disc_num_hidden = args.disc_num_hidden
    disc_num_layers = args.disc_num_layers
    disc_dropout = args.disc_dropout
    disc_act = args.disc_act

    with ExitStack() as exit_stack:
        # context
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

        # Build vocab
        # These vocabs are built on the training data.
        # Joint vocab for e and f
        vocab = _build_or_load_vocab(external_vocab, [e_corpus, f_corpus], num_words, word_min_count)
        sockeye.vocab.vocab_to_json(vocab, os.path.join(output_folder, C.VOCAB_SRC_NAME) + C.JSON_SUFFIX)
        sockeye.vocab.vocab_to_json(vocab, os.path.join(output_folder, C.VOCAB_TRG_NAME) + C.JSON_SUFFIX)

        vocab_size = len(vocab)
        logger.info("Vocabulary size (merged e, f): %d", vocab_size)

        # NamedTuple which will keep track of stuff
        data_info = sockeye.data_io.StyleDataInfo(os.path.abspath(e_corpus),
                                                  os.path.abspath(f_corpus),
                                                  os.path.abspath(e_mono_corpus),
                                                  os.path.abspath(f_mono_corpus),
                                                  os.path.abspath(e_val),
                                                  os.path.abspath(f_val),
                                                  vocab)

        # This will return a ParallelBucketIterator
        # For these, target is always = source (autenc target output)
        # Vocabularies are shared across e and f
        # e->e
        e_train_iter = sockeye.data_io.get_style_training_data_iters(
                                source=data_info.e_mono,
                                vocab=vocab,
                                batch_size=batch_size,
                                fill_up=args.fill_up,
                                max_seq_len=max_seq_len,
                                bucketing=not args.no_bucketing,
                                bucket_width=args.bucket_width,
                                target_bos_symbol=C.E_BOS_SYMBOL,
                                suffix='_e'
                            )

        # Similar iter for f->f
        f_train_iter = sockeye.data_io.get_style_training_data_iters(
                                source=data_info.f_mono,
                                vocab=vocab,
                                batch_size=batch_size,
                                fill_up=args.fill_up,
                                max_seq_len=max_seq_len,
                                bucketing=not args.no_bucketing,
                                bucket_width=args.bucket_width,
                                target_bos_symbol=C.F_BOS_SYMBOL,
                                suffix='_f'
                            )

        # Validation iters
        e_val_iter = sockeye.data_io.get_style_training_data_iters(
            source=data_info.e_val,
            vocab=vocab,
            batch_size=batch_size,
            fill_up='replicate_first',
            max_seq_len=max_seq_len,
            bucketing=not args.no_bucketing,
            bucket_width=args.bucket_width,
            target_bos_symbol=C.E_BOS_SYMBOL,
            suffix='_val_e',
            do_not_shuffle=True
        )

        f_val_iter = sockeye.data_io.get_style_training_data_iters(
            source=data_info.f_val,
            vocab=vocab,
            batch_size=batch_size,
            fill_up='replicate_first',
            max_seq_len=max_seq_len,
            bucketing=not args.no_bucketing,
            bucket_width=args.bucket_width,
            target_bos_symbol=C.F_BOS_SYMBOL,
            suffix='_val_f',
            do_not_shuffle=True
        )

        # Merge the two iterators to get one.
        make_iters_same_length(e_train_iter, f_train_iter, logger)
        train_iter = mx.io.PrefetchingIter([e_train_iter, f_train_iter])

        make_iters_same_length(e_val_iter, f_val_iter, logger)
        val_iter = mx.io.PrefetchingIter([e_val_iter, f_val_iter])

        model_config = sockeye.model.ModelConfig(max_seq_len=max_seq_len,
                                                 vocab_source_size=vocab_size,
                                                 vocab_target_size=vocab_size,
                                                 num_embed_source=num_embed,
                                                 num_embed_target=num_embed,
                                                 attention_type=attention_type,
                                                 attention_num_hidden=attention_num_hidden,
                                                 attention_coverage_type="count",
                                                 attention_coverage_num_hidden=1,
                                                 attention_use_prev_word=args.attention_use_prev_word,
                                                 dropout=dropout,
                                                 rnn_cell_type=rnn_cell_type,
                                                 rnn_num_layers=rnn_num_layers,
                                                 rnn_num_hidden=rnn_num_hidden,
                                                 rnn_residual_connections=args.rnn_residual_connections,
                                                 weight_tying=args.weight_tying,
                                                 context_gating=args.context_gating,
                                                 lexical_bias=args.lexical_bias,
                                                 learn_lexical_bias=args.learn_lexical_bias,
                                                 data_info=data_info,
                                                 loss=args.loss,
                                                 valid_loss=args.valid_loss,
                                                 normalize_loss=args.normalize_loss,
                                                 smoothed_cross_entropy_alpha=args.smoothed_cross_entropy_alpha,
                                                 disc_act=disc_act,
                                                 disc_num_hidden=disc_num_hidden,
                                                 disc_num_layers=disc_num_layers,
                                                 disc_dropout=disc_dropout,
                                                 loss_lambda=args.disc_loss_lambda,
                                                 g_loss_weight=args.g_loss_weight,
                                                 disc_batch_norm=args.disc_batch_norm)

        # For lexical bias, set to None
        lexicon = None

        initializer = sockeye.initializer.get_initializer(args.rnn_h2h_init, lexicon=lexicon)

        optimizer = args.optimizer
        optimizer_params = {'wd': args.weight_decay,
                            "learning_rate": args.initial_learning_rate}
        clip_gradient = none_if_negative(args.clip_gradient)
        if clip_gradient is not None:
            optimizer_params["clip_gradient"] = clip_gradient
        if args.momentum is not None:
            optimizer_params["momentum"] = args.momentum
        if args.normalize_loss:
            # When normalize_loss is turned on we normalize by the number of non-PAD symbols in a batch which implicitly
            # already contains the number of sentences and therefore we need to disable rescale_grad.
            optimizer_params["rescale_grad"] = 1.0
        else:
            # Making MXNet module API's default scaling factor explicit
            optimizer_params["rescale_grad"] = 1.0 / args.batch_size
        logger.info("Optimizer: %s", optimizer)
        logger.info("Optimizer Parameters: %s", optimizer_params)

        ####################### PRE_TRAINING D ################################
        # Get the pre-trained params from G
        pretrain_param_path = os.path.join(output_folder_orig, "g_pretrain", C.PARAMS_BEST_NAME)
        assert os.path.exists(pretrain_param_path)
        G_params = sockeye.utils.load_params(pretrain_param_path)[0]
        # The params to keep fixed during the training of D.

        logger.info("Copying pre-trained params (G) and starting pre-training of D")

        # Learning rate scheduler for pre-training G
        learning_rate_half_life = none_if_negative(args.learning_rate_half_life)

        if not resume_training:
            lr_scheduler = sockeye.lr_scheduler.get_lr_scheduler(args.learning_rate_scheduler_type_pre_d,
                                                                 args.checkpoint_frequency,
                                                                 learning_rate_half_life,
                                                                 args.learning_rate_reduce_factor,
                                                                 args.learning_rate_reduce_num_not_improved)
        else:
            with open(os.path.join(training_state_dir, C.SCHEDULER_STATE_NAME), "rb") as fp:
                lr_scheduler = pickle.load(fp)

        optimizer_params["lr_scheduler"] = lr_scheduler

        # Initialize the pre-training model for D
        D_pretrain_model = sockeye.style_training.StyleTrainingModel(model_config=model_config,
                                                          context=context,
                                                          train_iter=train_iter,
                                                          valid_iter=val_iter,
                                                          fused=args.use_fused_rnn,
                                                          bucketing=not args.no_bucketing,
                                                          lr_scheduler=lr_scheduler,
                                                          rnn_forget_bias=args.rnn_forget_bias,
                                                          vocab=vocab)

        # Pack the RNN cells from the G-params
        for cell in D_pretrain_model.rnn_cells:
            G_params = cell.pack_weights(G_params)

        # Keys to keep fixed
        D_fixed_params = list(G_params.keys())
        D_pretrain_model.module._fixed_param_names = D_fixed_params

        if resume_training:
            logger.info("Found partial training in directory %s. Resuming from saved state.", training_state_dir)
            D_pretrain_model.load_params_from_file(os.path.join(training_state_dir, C.TRAINING_STATE_PARAMS_NAME))

        # initialize memory for params
        D_pretrain_model.module.bind(data_shapes=train_iter.provide_data,
                                     label_shapes=train_iter.provide_label,
                                     for_training=True,
                                     force_rebind=True,
                                     grad_req='write')
        # Initialize params
        D_pretrain_model.module.init_params(initializer=initializer,
                                            arg_params=D_pretrain_model.params,
                                            aux_params=None,
                                            allow_missing=False,
                                            force_init=False)

        if not resume_training:
            # Copy params from G to D
            D_pretrain_model.load_params_from_file(pretrain_param_path)
            D_pretrain_model.module.set_params(G_params, None, allow_missing=True)

        # Pre-train D
        D_pretrain_model.fit(train_iter=train_iter,
                             val_iter=None,
                             output_folder=output_folder,
                             metrics=args.metrics,
                             initializer=initializer,
                             max_updates=args.max_updates,
                             checkpoint_frequency=args.checkpoint_frequency,
                             optimizer=optimizer, optimizer_params=optimizer_params,
                             optimized_metric=C.CROSS_ENTROPY,
                             metric_threshold=0.1,
                             max_num_not_improved=args.max_num_checkpoint_not_improved,
                             min_num_epochs=args.min_num_epochs,
                             monitor_bleu=args.monitor_bleu,
                             use_tensorboard=args.use_tensorboard)


if __name__ == "__main__":
    main()
