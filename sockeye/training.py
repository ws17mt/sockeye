# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for training
"""
import logging
import os
import pickle
import random
import shutil
import time
from typing import AnyStr, List, Optional

import mxnet as mx
import numpy as np

import sockeye.callback
import sockeye.checkpoint_decoder
import sockeye.constants as C
import sockeye.data_io
import sockeye.inference
import sockeye.loss
import sockeye.lr_scheduler
import sockeye.model
import sockeye.utils

logger = logging.getLogger(__name__)


class _TrainingState:
    """
    Stores the state of the training process. These are the variables that will
    be stored to disk when resuming training.
    """
    def __init__(self,
                 num_not_improved,
                 epoch,
                 checkpoint,
                 updates,
                 samples,
                 mono_epoch):
        self.num_not_improved = num_not_improved
        self.epoch = epoch
        self.checkpoint = checkpoint
        self.updates = updates
        self.samples = samples
        self.mono_epoch = mono_epoch


class TrainingModel(sockeye.model.SockeyeModel):
    """
    Defines an Encoder/Decoder model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.

    :param model_config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param fused: If True fused RNN cells will be used (should be slightly more efficient, but is only available
            on GPUs).
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    :param rnn_forget_bias: Initial value of the RNN forget biases.
    :param mono_source_iter: Optional iterator for monolingual source data.
    :param mono_source_iter: Optional iterator for monolingual target data.
    """

    def __init__(self,
                 model_config: sockeye.model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 rnn_forget_bias: float,
                 mono_source_iter: sockeye.data_io.MonoBucketSentenceIter=None,
                 mono_target_iter: sockeye.data_io.MonoBucketSentenceIter=None) -> None:
        super().__init__(model_config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self.bucketing = bucketing
        self._build_model_components(self.config.max_seq_len, fused, rnn_forget_bias)
        self.module = self._build_module(train_iter, self.config.max_seq_len)
        self.lm_source_module = None
        self.lm_target_module = None
        if mono_source_iter is not None:
            self.lm_source_module = self.build_lm_module(mono_source_iter, self.config.max_seq_len)
        if mono_target_iter is not None:
            self.lm_target_module = self.build_lm_module(mono_target_iter, self.config.max_seq_len)
        self.training_monitor = None

    def _build_lm_module(self,
                         mono_iter: sockeye.data_io.MonoBucketSentenceIter,
                         max_seq_len: int):
        """
        Build a sister module for training an LM
        """
        mono = mx.sym.Variable(C.MONO_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.MONO_LABEL_NAME))
        loss = sockeye.loss.get_loss(self.config)

        data_names = [x[0] for x in mono_iter.provide_data]
        label_names = [x[0] for x in mono_iter.provide_label]

        def sym_gen(seq_len):
            """
            Returns a (grouped) loss symbol given mono input length.
            Also returns data and label names for the BucketingModule
            """
            logits = self.lm.encode(mono, None, seq_len=seq_len)
            outputs = loss.get_loss(logits, labels)
            return mx.sym.Group(outputs), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", mono_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=mono_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to max_seq_len=%s", max_seq_len)
            symbol, _, __ = sym_gen(mono_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

    def _build_module(self,
                      train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                      max_seq_len: int):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME)
        target = mx.sym.Variable(C.TARGET_NAME)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        loss = sockeye.loss.get_loss(self.config)

        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

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

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to max_seq_len=%s", max_seq_len)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

    @staticmethod
    def _create_eval_metric(metric_names: List[AnyStr]) -> mx.metric.CompositeEvalMetric:
        """
        Creates a composite EvalMetric given a list of metric names.
        """
        metrics = []
        # output_names refers to the list of outputs this metric should use to update itself, e.g. the softmax output
        for metric_name in metric_names:
            if metric_name == C.ACCURACY:
                metrics.append(sockeye.utils.Accuracy(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            elif metric_name == C.PERPLEXITY:
                metrics.append(mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.SOFTMAX_OUTPUT_NAME]))
            else:
                raise ValueError("unknown metric name")
        return mx.metric.create(metrics)

    def _prime_module(self, module, train_iter, output_folder, symbol_name, initializer, optimizer, optimizer_params):
        if module is not None:
            module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label,
                        for_training=True, force_rebind=True, grad_req='write')

            module.symbol.save(os.path.join(output_folder, symbol_name))

            module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                               allow_missing=False, force_init=False)

            module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

    def fit(self,
            train_iter: sockeye.data_io.ParallelBucketSentenceIter,
            val_iter: sockeye.data_io.ParallelBucketSentenceIter,
            output_folder: str,
            metrics: List[AnyStr],
            initializer: mx.initializer.Initializer,
            max_updates: int,
            checkpoint_frequency: int,
            optimizer: str,
            optimizer_params: dict,
            optimized_metric: str = "perplexity",
            max_num_not_improved: int = 3,
            min_num_epochs: Optional[int] = None,
            monitor_bleu: int = 0,
            use_tensorboard: bool = False,
            mono_source_iter: sockeye.data_io.MonoBucketSentenceIter=None,
            mono_target_iter: sockeye.data_io.MonoBucketSentenceIter=None):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator.
        :param val_iter: The validation data iterator.
        :param output_folder: The folder in which all model artifacts will be stored in (parameters, checkpoints, etc.).
        :param metrics: The metrics that will be evaluated during training.
        :param initializer: The parameter initializer.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing in number of updates.
        :param optimizer: The MXNet optimizer that will update the parameters.
        :param optimizer_params: The parameters for the optimizer.
        :param optimized_metric: The metric that is tracked for early stopping.
        :param max_num_not_improved: Stop training if the optimized_metric does not improve for this many checkpoints.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        :param monitor_bleu: Monitor BLEU during training (0: off, >=0: the number of sentences to decode for BLEU
               evaluation, -1: decode the full validation set.).
        :param use_tensorboard: If True write tensorboard compatible logs for monitoring training and
               validation metrics.
        :return: Best score on validation data observed during training.
        """
        self.save_config(output_folder)

        self._prime_module(self.module, train_iter, output_folder,
                           C.SYMBOL_NAME, initializer, optimizer, optimizer_params)
        self._prime_module(self.lm_source_module, mono_source_iter, output_folder,
                           C.L2R_SOURCE_SYMBOL_NAME, initializer, optimizer, optimizer_params)
        self._prime_module(self.lm_target_module, mono_target_iter, output_folder,
                           C.TARGET_SYMBOL_NAME, initializer, optimizer, optimizer_params)

        checkpoint_decoder = sockeye.checkpoint_decoder.CheckpointDecoder(self.context[-1],
                                                                          self.config.data_info.validation_source,
                                                                          self.config.data_info.validation_target,
                                                                          output_folder, self.config.max_seq_len,
                                                                          limit=monitor_bleu) \
            if monitor_bleu else None

        logger.info("Training started.")
        self.training_monitor = sockeye.callback.TrainingMonitor(train_iter.batch_size, output_folder,
                                                                 optimized_metric=optimized_metric,
                                                                 use_tensorboard=use_tensorboard,
                                                                 checkpoint_decoder=checkpoint_decoder)
        self._fit(train_iter, val_iter, output_folder,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved,
                  min_num_epochs=min_num_epochs,
                  mono_source_iter=mono_source_iter,
                  mono_target_iter=mono_target_iter)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.training_monitor.get_best_checkpoint(),
                    self.training_monitor.optimized_metric,
                    self.training_monitor.get_best_validation_score())
        return self.training_monitor.get_best_validation_score()

    def _update_mono(self,
                     mono_iter,
                     module,
                     next_data_batch,
                     train_state,
                     metric):
        if mono_iter is not None:
            if not mono_iter.iter_next():
                train_state.epoch[mono_iter.name] += 1
                mono_iter.reset()

            # process batch
            batch = next_data_batch
            module.forward_backward(batch)
            module.update()

            if mono_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = mono_iter.next()
                module.prepare(next_data_batch)

            if metric is not None:
                module.update_metric(metric, batch.label)

            return next_data_batch

    def _fit(self,
             train_iter: sockeye.data_io.ParallelBucketSentenceIter,
             val_iter: sockeye.data_io.ParallelBucketSentenceIter,
             output_folder: str,
             metrics: List[AnyStr],
             max_updates: int,
             checkpoint_frequency: int,
             max_num_not_improved: int,
             min_num_epochs: Optional[int] = None,
             mono_source_iter: sockeye.data_io.MonoBucketSentenceIter=None,
             mono_target_iter: sockeye.data_io.MonoBucketSentenceIter=None):
        """
        Internal fit method. Runtime determined by early stopping.

        :param train_iter: Training data iterator.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        """
        metric_train = self._create_eval_metric(metrics)
        metric_val = self._create_eval_metric(metrics)
        tic = time.time()

        training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(training_state_dir):
            train_state = self.load_checkpoint(training_state_dir, train_iter, mono_source_iter, mono_target_iter)
        else:
            # TODO: Consider having one _TrainingState per task (remove mono_epoch dict)
            train_state = _TrainingState(
                num_not_improved=0,
                epoch=0,
                checkpoint=0,
                updates=0,
                samples=0,
                mono_epoch={}
            )

        next_data_batch = train_iter.next()
        next_source_batch = None
        next_target_batch = None
        if mono_source_iter is not None:
            next_source_batch = mono_source_iter.next()
            train_state.mono_epoch[mono_source_iter.name] = 0
        if mono_target_iter is not None:
            next_target_batch = mono_target_iter.next()
            train_state.mono_epoch[mono_target_iter.name] = 0

        while max_updates == -1 or train_state.updates < max_updates:
            if not train_iter.iter_next():
                train_state.epoch += 1
                train_iter.reset()

            # process batch
            batch = next_data_batch
            self.module.forward_backward(batch)
            self.module.update()

            if train_iter.iter_next():
                # pre-fetch next batch
                next_data_batch = train_iter.next()
                self.module.prepare(next_data_batch)

            self.module.update_metric(metric_train, batch.label)

            # TODO: Add real metrics to the LM objectives
            next_source_batch = self._update_mono(mono_source_iter, self.lm_source_module,
                                                  next_source_batch, train_state, None)
            next_target_batch = self._update_mono(mono_target_iter, self.lm_target_module,
                                                  next_target_batch, train_state, None)

            # TODO: Move into _update_mono
            self.training_monitor.batch_end_callback(train_state.epoch, train_state.updates, metric_train)
            train_state.updates += 1
            train_state.samples += train_iter.batch_size
            # End TODO

            if train_state.updates > 0 and train_state.updates % checkpoint_frequency == 0:
                train_state.checkpoint += 1
                self._save_params(output_folder, train_state.checkpoint)
                self.training_monitor.checkpoint_callback(train_state.checkpoint, metric_train)

                toc = time.time()
                logger.info("Checkpoint [%d]\tUpdates=%d Epoch=%d Samples=%d Time-cost=%.3f",
                            train_state.checkpoint, train_state.updates, train_state.epoch,
                            train_state.samples, (toc - tic))
                tic = time.time()

                for name, val in metric_train.get_name_value():
                    logger.info('Checkpoint [%d]\tTrain-%s=%f', train_state.checkpoint, name, val)
                metric_train.reset()

                # evaluation on validation set
                has_improved, best_checkpoint = self._evaluate(train_state, val_iter, metric_val)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.new_evaluation_result(has_improved)

                if has_improved:
                    best_path = os.path.join(output_folder, C.PARAMS_BEST_NAME)
                    if os.path.lexists(best_path):
                        os.remove(best_path)
                    actual_best_fname = C.PARAMS_NAME % best_checkpoint
                    os.symlink(actual_best_fname, best_path)
                    train_state.num_not_improved = 0
                else:
                    train_state.num_not_improved += 1
                    logger.info("Model has not improved for %d checkpoints", train_state.num_not_improved)

                if train_state.num_not_improved >= max_num_not_improved:
                    logger.info("Maximum number of not improved checkpoints (%d) reached: %d",
                                max_num_not_improved, train_state.num_not_improved)
                    stop_fit = True

                    if min_num_epochs is not None and train_state.epoch < min_num_epochs:
                        logger.info("Minimum number of epochs (%d) not reached yet: %d",
                                    min_num_epochs,
                                    train_state.epoch)
                        stop_fit = False

                    if stop_fit:
                        logger.info("Stopping fit")
                        self.training_monitor.stop_fit_callback()
                        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
                        if os.path.exists(final_training_state_dirname):
                            shutil.rmtree(final_training_state_dirname)
                        break

                self._checkpoint(train_state, output_folder, train_iter, mono_source_iter, mono_target_iter)

    def _save_params(self, output_folder: str, checkpoint: int):
        """
        Saves the parameters to disk.
        """
        arg_params, aux_params = self.module.get_params()  # sync aux params across devices
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        params_base_fname = C.PARAMS_NAME % checkpoint
        self.save_params_to_file(os.path.join(output_folder, params_base_fname))

    def _evaluate(self, training_state, val_iter, val_metric):
        """
        Computes val_metric on val_iter. Returns whether model improved or not.
        """
        val_iter.reset()
        val_metric.reset()

        for nbatch, eval_batch in enumerate(val_iter):
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(val_metric, eval_batch.label)

        for name, val in val_metric.get_name_value():
            logger.info('Checkpoint [%d]\tValidation-%s=%f', training_state.checkpoint, name, val)

        return self.training_monitor.eval_end_callback(training_state.checkpoint, val_metric)

    def _module_checkpoint(self, training_state_dirname,
                           train_iter, iter_file,
                           module, module_file):
        if train_iter is not None:
            # Optimizer state (from mxnet)
            opt_state_fname = os.path.join(training_state_dirname, module_file)
            if self.bucketing:
                # This is a bit hacky, as BucketingModule does not provide a
                # save_optimizer_states call. We take the current active module and
                # save its state. This should work, as the individual modules in
                # BucketingModule share the same optimizer through
                # borrow_optimizer.
                self.module._curr_module.save_optimizer_states(opt_state_fname)
            else:
                self.module.save_optimizer_states(opt_state_fname)

            # State of the bucket iterator
            train_iter.save_state(os.path.join(training_state_dirname, iter_file))

    def _checkpoint(self, training_state: _TrainingState, output_folder: str,
                    train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                    mono_source_iter: sockeye.data_io.MonoBucketSentenceIter,
                    mono_target_iter: sockeye.data_io.MonoBucketSentenceIter):
        """
        Saves checkpoint. Note that the parameters are saved in _save_params.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)
        # Link current parameter file
        params_base_fname = C.PARAMS_NAME % training_state.checkpoint
        os.symlink(os.path.join("..", params_base_fname),
                   os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        self._module_checkpoint(training_state_dirname,
                                train_iter, C.BUCKET_ITER_STATE_NAME,
                                self.module, C.MODULE_OPT_STATE_NAME)

        self._module_checkpoint(training_state_dirname,
                                mono_source_iter, C.SOURCE_BUCKET_ITER_STATE_NAME,
                                self.lm_source_module, C.SOURCE_MODULE_OPT_STATE_NAME)

        self._module_checkpoint(training_state_dirname,
                                mono_target_iter, C.TARGET_BUCKET_ITER_STATE_NAME,
                                self.lm_target_module, C.TARGET_MODULE_OPT_STATE_NAME)

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp)  # Yes, one uses _, the other does not

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.save_state(os.path.join(training_state_dirname, C.MONITOR_STATE_NAME))

        # Our own state
        self.save_state(training_state, os.path.join(training_state_dirname, C.TRAINING_STATE_NAME))

        # The lr scheduler
        with open(os.path.join(training_state_dirname, C.SCHEDULER_STATE_NAME), "wb") as fp:
            pickle.dump(self.lr_scheduler, fp)

        # We are now finished with writing. Rename the temporary directory to
        # the actual directory
        final_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)

        # First we rename the existing directory to minimize the risk of state
        # loss if the process is aborted during deletion (which will be slower
        # than directory renaming)
        delete_training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DELETENAME)
        if os.path.exists(final_training_state_dirname):
            os.rename(final_training_state_dirname, delete_training_state_dirname)
        os.rename(training_state_dirname, final_training_state_dirname)
        if os.path.exists(delete_training_state_dirname):
            shutil.rmtree(delete_training_state_dirname)

    def save_state(self, training_state: _TrainingState, fname: str):
        """
        Saves the state (of the TrainingModel class) to disk.

        :param fname: file name to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(training_state, fp)

    def load_state(self, fname: str) -> _TrainingState:
        """
        Loads the training state (of the TrainingModel class) from disk.

        :param fname: file name to load the state from.
        """
        training_state = None
        with open(fname, "rb") as fp:
            training_state = pickle.load(fp)
        return training_state

    def _load_checkpoint(self,
                         directory: str,
                         train_iter,
                         iter_file: str,
                         module: mx.mod.Module,
                         module_file: str):
        if train_iter is not None:
            # Source optimzer state (from mxnet)
            opt_state_fname = os.path.join(directory, module_file)
            if self.bucketing:
                # Same hacky solution as for saving the state
                module._curr_module.load_optimizer_states(opt_state_fname)
            else:
                module.load_optimizer_states(opt_state_fname)

            # State of the bucket iterator
            train_iter.load_state(os.path.join(directory, iter_file))

    def load_checkpoint(self, directory: str,
                        train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                        mono_source_iter: sockeye.data_io.MonoBucketSentenceIter,
                        mono_target_iter: sockeye.data_io.MonoBucketSentenceIter) -> _TrainingState:
        """
        Loads the full training state from disk. This includes optimizer,
        random number generators and everything needed.  Note that params
        should have been loaded already by the initializer.

        :param directory: directory where the state has been saved.
        :param train_iter: training data iterator.
        """
        self._load_checkpoint(directory,
                              train_iter, C.BUCKET_ITER_STATE_NAME,
                              self.module, C.MODULE_OPT_STATE_NAME)
        self._load_checkpoint(directory,
                              mono_source_iter, C.SOURCE_BUCKET_ITER_STATE_NAME,
                              self.lm_source_module, C.SOURCE_MODULE_OPT_STATE_NAME)
        self._load_checkpoint(directory,
                              mono_target_iter, C.TARGET_BUCKET_ITER_STATE_NAME,
                              self.lm_target_module, C.TARGET_MODULE_OPT_STATE_NAME)

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(directory, C.RNG_STATE_NAME), "rb") as fp:
            random.setstate(pickle.load(fp))
            np.random.set_state(pickle.load(fp))

        # Monitor state, in order to get the full information about the metrics
        self.training_monitor.load_state(os.path.join(directory, C.MONITOR_STATE_NAME))

        # And our own state
        return self.load_state(os.path.join(directory, C.TRAINING_STATE_NAME))
