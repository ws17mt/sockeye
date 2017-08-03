import os
import argparse
from contextlib import ExitStack
import mxnet as mx

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

logger = setup_main_logger(__name__, file_logging=False, console=True)

params = argparse.ArgumentParser(description='CLI to train sockeye style transfer models.')
arguments.add_io_args(params)
arguments.add_model_parameters(params)
arguments.add_training_args(params)
arguments.add_device_args(params)
args = params.parse_args()

e_corpus = args.source
f_corpus = args.target
e_mono_corpus = args.mono_source
f_mono_corpus = args.mono_target
e_val = args.validation_source
f_val = args.validation_target
external_vocab = args.joint_vocab

output_folder = args.output

lr_scheduler = None
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

    # Similar iter for e->f
    ef_train_iter = sockeye.data_io.get_style_training_data_iters(
                            source=data_info.e,
                            target=data_info.f,
                            vocab=vocab,
                            batch_size=batch_size,
                            fill_up=args.fill_up,
                            max_seq_len=max_seq_len,
                            bucketing=not args.no_bucketing,
                            bucket_width=args.bucket_width,
                            target_bos_symbol=C.F_BOS_SYMBOL,
                            suffix='_e',
                            target_suffix='_f'
                        )

    # Similar iter for f->e
    fe_train_iter = sockeye.data_io.get_style_training_data_iters(
                            source=data_info.f,
                            target=data_info.e,
                            vocab=vocab,
                            batch_size=batch_size,
                            fill_up=args.fill_up,
                            max_seq_len=max_seq_len,
                            bucketing=not args.no_bucketing,
                            bucket_width=args.bucket_width,
                            target_bos_symbol=C.E_BOS_SYMBOL,
                            suffix='_f',
                            target_suffix='_e'
                        )

    # Validation iter
    ef_val_iter = sockeye.data_io.get_style_training_data_iters(
                            source=data_info.e_val,
                            target=data_info.f_val,
                            vocab=vocab,
                            batch_size=batch_size,
                            fill_up=args.fill_up,
                            max_seq_len=max_seq_len,
                            bucketing=not args.no_bucketing,
                            bucket_width=args.bucket_width,
                            target_bos_symbol=C.F_BOS_SYMBOL,
                            suffix='_val_e',
                            target_suffix='_val_f'
                        )

    fe_val_iter = sockeye.data_io.get_style_training_data_iters(
                            source=data_info.f_val,
                            target=data_info.e_val,
                            vocab=vocab,
                            batch_size=batch_size,
                            fill_up=args.fill_up,
                            max_seq_len=max_seq_len,
                            bucketing=not args.no_bucketing,
                            bucket_width=args.bucket_width,
                            target_bos_symbol=C.E_BOS_SYMBOL,
                            suffix='_val_f',
                            target_suffix='_val_e'
                        )

    # Merge the two iterators to get one.
    train_iter = mx.io.PrefetchingIter([e_train_iter, f_train_iter])
    pretrain_iter = mx.io.PrefetchingIter([ef_train_iter, fe_train_iter])
    val_iter = mx.io.PrefetchingIter([ef_val_iter, fe_val_iter])

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
                                             loss_lambda=args.disc_loss_lambda)

    # For lexical bias, set to None
    lexicon = None

    initializer = sockeye.initializer.get_initializer(args.rnn_h2h_init, lexicon=lexicon)

    optimizer = args.optimizer
    optimizer_params = {'wd': args.weight_decay,
                        "learning_rate": args.initial_learning_rate}
    if lr_scheduler is not None:
        optimizer_params["lr_scheduler"] = lr_scheduler
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

    # This is a hack to get the pre-training and joint models to write
    # resumption files in different directories
    C.TRAINING_STATE_DIRNAME = "g_pretraining_state"
    C.TRAINING_STATE_TEMP_DIRNAME = "tmp.g_pretraining_state"
    C.TRAINING_STATE_TEMP_DELETENAME = "delete.g_pretraining_state"

    G_pretrain_model = sockeye.style_pretraining_G.StylePreTrainingModel_G(model_config=model_config,
                                                            context=context,
                                                            train_iter=pretrain_iter,
                                                            fused=args.use_fused_rnn,
                                                            bucketing=not args.no_bucketing,
                                                            lr_scheduler=lr_scheduler,
                                                            rnn_forget_bias=args.rnn_forget_bias,
                                                            vocab=vocab)

    logger.info("Starting pre-training of G")

    # Pre-train G
    G_pretrain_model.fit(train_iter=pretrain_iter,
              val_iter=val_iter,
              output_folder=output_folder,
              metrics=args.metrics,
              initializer=initializer,
              max_updates=args.max_updates,
              checkpoint_frequency=args.checkpoint_frequency,
              optimizer=optimizer, optimizer_params=optimizer_params,
              optimized_metric=args.optimized_metric,
              max_num_not_improved=args.max_num_checkpoint_not_improved,
              min_num_epochs=args.min_num_epochs,
              monitor_bleu=args.monitor_bleu,
              use_tensorboard=args.use_tensorboard)



    # Get the pre-trained params from G
    G_params = G_pretrain_model.module.get_params()
    D_fixed_params = list(G_params[0].keys())

    logger.info("Copying pre-trained params (G) and starting pre-training of D")

    C.TRAINING_STATE_DIRNAME = "d_pretraining_state"
    C.TRAINING_STATE_TEMP_DIRNAME = "tmp.d_pretraining_state"
    C.TRAINING_STATE_TEMP_DELETENAME = "delete.d_pretraining_state"

    # Initialize the pre-training model for D
    D_pretrain_model = sockeye.style_training.StyleTrainingModel(model_config=model_config,
                                                      context=context,
                                                      train_iter=train_iter,
                                                      valid_iter=val_iter,
                                                      fused=args.use_fused_rnn,
                                                      bucketing=not args.no_bucketing,
                                                      lr_scheduler=lr_scheduler,
                                                      rnn_forget_bias=args.rnn_forget_bias,
                                                      vocab=vocab,
                                                      fixed_param_names=D_fixed_params)

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

    # Copy params from G to D
    D_pretrain_model.module.set_params(G_params[0], G_params[1], allow_missing=True)

    # Pre-train D
    D_pretrain_model.fit(train_iter=train_iter,
                         val_iter=val_iter,
                         output_folder=output_folder,
                         metrics=args.metrics,
                         initializer=initializer,
                         max_updates=args.max_updates,
                         checkpoint_frequency=args.checkpoint_frequency,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         optimized_metric=args.optimized_metric,
                         max_num_not_improved=args.max_num_checkpoint_not_improved,
                         min_num_epochs=args.min_num_epochs,
                         monitor_bleu=args.monitor_bleu,
                         use_tensorboard=args.use_tensorboard)

    C.TRAINING_STATE_DIRNAME = "training_state"
    C.TRAINING_STATE_TEMP_DIRNAME = "tmp.training_state"
    C.TRAINING_STATE_TEMP_DELETENAME = "delete.training_state"

    model = sockeye.style_training.StyleTrainingModel(model_config=model_config,
                                                      context=context,
                                                      train_iter=train_iter,
                                                      valid_iter=val_iter,
                                                      fused=args.use_fused_rnn,
                                                      bucketing=not args.no_bucketing,
                                                      lr_scheduler=lr_scheduler,
                                                      rnn_forget_bias=args.rnn_forget_bias,
                                                      vocab=vocab)

    # Get the pre-trained params from G
    D_params = D_pretrain_model.module.get_params()

    # initialize memory for params
    model.module.bind(data_shapes=train_iter.provide_data,
                      label_shapes=train_iter.provide_label,
                      for_training=True,
                      force_rebind=True,
                      grad_req='write')
    # Initialize params
    model.module.init_params(initializer=initializer,
                             arg_params=model.params,
                             aux_params=None,
                             allow_missing=False,
                             force_init=False)

    # Copy params from D to joint model
    model.module.set_params(D_params[0], D_params[1], allow_missing=False)

    model.fit(train_iter=train_iter,
              val_iter=val_iter,
              output_folder=output_folder,
              metrics=args.metrics,
              initializer=initializer,
              max_updates=args.max_updates,
              checkpoint_frequency=args.checkpoint_frequency,
              optimizer=optimizer, optimizer_params=optimizer_params,
              optimized_metric=args.optimized_metric,
              max_num_not_improved=args.max_num_checkpoint_not_improved,
              min_num_epochs=args.min_num_epochs,
              monitor_bleu=args.monitor_bleu,
              use_tensorboard=args.use_tensorboard)
