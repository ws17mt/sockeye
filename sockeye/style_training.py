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
import sockeye.discriminator
import sockeye.inference
import sockeye.loss
import sockeye.lr_scheduler
import sockeye.model
import sockeye.utils
import sockeye.decoder

logger = logging.getLogger(__name__)

def mask_labels_after_EOS(logits: mx.sym.Symbol,
                          batch_size: int,
                          target_seq_len: int) -> mx.sym.Symbol:
    """
    Calculate sentence lengths for translated sentences.
    
    :param logits: Input data. Shape: (batch_size * target_seq_len, target_vocab_size).
    :param batch_size: Batch size for the input data.
    :param target_seq_len: Maximum sequence length for target sentences.
    :return: Actual sequence lengths for each target sequence. Shape: (batch_size,).
    """
    # (batch_size, ) : index of the highest-prob word at each timestep
    best_tokens = mx.sym.argmax(logits, axis=1)
    # NOTE rows are word1,sent1 - word2,sent1 - ...
    # (batch_size, target_seq_len): as above, but reshaped
    best_tokens = best_tokens.reshape((-1, target_seq_len))
    eos_index = C.VOCAB_SYMBOLS.index(C.EOS_SYMBOL)
    eos_sym = mx.sym.ones((1,))
    # (1, ) : symbol representing the index of EOS
    eos_sym = eos_sym * eos_index
    # (batch_size, target_seq_len) : 1 if EOS, 0 else
    eos_indices = mx.sym.broadcast_equal(lhs=best_tokens, rhs=eos_sym)
    # add a 1 in the last position of every line -- if no EOS, the last word is EOS
    # (batch_size, target_len) : zeros in all positions except ones in last column
    last_word = mx.sym.concat(mx.sym.zeros(shape=(batch_size, target_seq_len-1)),
                              mx.sym.ones(shape=(batch_size, 1)), dim=1)
    # if there is a zero in the last column, replace it with a one
    eos_indices = mx.sym.maximum(eos_indices, last_word)
    # (batch_size, ) : position of the first EOS in each sentence
    eos_position = mx.sym.argmax(eos_indices, axis=1)
    # in fact, we want length, not position
    sentence_length = mx.sym.broadcast_plus(eos_position, mx.sym.ones(1,))
    return sentence_length

class _StyleTrainingState:
    """
    Stores the state of the training process. These are the variables that will
    be stored to disk when resuming training.
    """
    def __init__(self,
                 num_not_improved,
                 epoch,
                 checkpoint,
                 updates,
                 samples):
        self.num_not_improved = num_not_improved
        self.epoch = epoch
        self.checkpoint = checkpoint
        self.updates = updates
        self.samples = samples


# encoder-generator-discriminator(s)
class StyleTrainingModel(sockeye.model.SockeyeModel):
    """
    Defines an Encoder/Decoder/Discriminators model (with attention).
    RNN configuration (number of hidden units, number of layers, cell type)
    is shared between encoder & decoder.
    Discriminator configuration (number of hidden units, number of layers, activation)
    is shared between discriminators.

    :param model_config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param train_iter: The iterator over the training data.
    :param fused: If True fused RNN cells will be used (should be slightly more efficient, but is only available
            on GPUs).
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param lr_scheduler: The scheduler that lowers the learning rate during training.
    :param rnn_forget_bias: Initial value of the RNN forget biases.
    """

    def __init__(self,
                 model_config: sockeye.model.ModelConfig,
                 context: List[mx.context.Context],
                 train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                 fused: bool,
                 bucketing: bool,
                 lr_scheduler,
                 rnn_forget_bias: float,
                 vocab) -> None:
        super().__init__(model_config)
        self.context = context
        self.lr_scheduler = lr_scheduler
        self.bucketing = bucketing
        self.embedding = sockeye.encoder.Embedding(num_embed=self.config.num_embed_source,
                                                   vocab_size=len(vocab),
                                                   prefix=C.EMBEDDING_PREFIX,
                                                   dropout=self.config.dropout)
        self.vocab = vocab
        self._build_model_components(self.config.max_seq_len, fused, rnn_forget_bias, initialize_embedding=False)
        self._build_discriminators(self.config.disc_act, self.config.disc_num_hidden, self.config.disc_num_layers,
                                   self.config.disc_dropout, self.config.loss_lambda)
        self.module = self._build_module(train_iter, self.config.max_seq_len, self.config.max_seq_len)
        self.training_monitor = None

    def _build_discriminators(self, act: str, num_hidden: int, num_layers: int, dropout: float, loss_lambda: float):
        """
        Builds and sets discriminators for style transfer.

        :param act: Activation function for the discriminators.
        :param num_hidden: Number of hidden units for the discriminators.
        :param num_layers: Number of layers for the discriminators.
        :param dropout: Dropout probability for the discriminators.
        :param loss_lambda: Weight for the discriminator losses.
        """
        self.discriminator_e = sockeye.discriminator.get_discriminator(act, num_hidden,
                                                                       num_layers, dropout,
                                                                       loss_lambda,
                                                                       prefix=C.DISCRIMINATOR_E_PREFIX)
        self.discriminator_f = sockeye.discriminator.get_discriminator(act, num_hidden,
                                                                       num_layers, dropout,
                                                                       loss_lambda,
                                                                       prefix=C.DISCRIMINATOR_F_PREFIX)

    def _build_module(self,
                      train_iter: sockeye.data_io.ParallelBucketSentenceIter,
                      e_max_seq_len: int,
                      f_max_seq_len: int):
        """
        Initializes model components, creates training symbol and module, and binds it.

        :param train_iter: The training iterator; this contains the data for e and f
        :param e_max_seq_len: The max length of the e encoder/decoders
        :param f_max_seq_len: The max length of the f encoder/decoders
        """

        # The symbol for the source/target and their lengths
        # source_e: (bs, seq_len)
        e_source = mx.sym.Variable(C.SOURCE_NAME + '_e')
        # source_length_e: (bs,)
        e_source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME + '_e')
        # target_e: (bs, seq_len)
        e_target = mx.sym.Variable(C.TARGET_NAME + '_e')
        # target_label_e: (bs, seq_len) -> (bs*seq_len,)
        e_labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME + '_e'), shape=(-1,))

        # source_f: (bs, seq_len)
        f_source = mx.sym.Variable(C.SOURCE_NAME + '_f')
        # source_length_f: (bs,)
        f_source_length = mx.sym.Variable(C.SOURCE_LENGTH_NAME + '_f')
        # target_e: (bs, seq_len)
        f_target = mx.sym.Variable(C.TARGET_NAME + '_f')
        # target_label_e: (bs, seq_len) -> (bs*seq_len,)
        f_labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME + '_f'), shape=(-1,))

        # Returns loss_G, loss_D
        # loss_G ;
        loss = sockeye.loss.get_loss(self.config)

        # ['source_e', 'source_length_e', 'target_e', 'source_f', 'source_length_f', 'target_f']
        data_names = [x[0] for x in train_iter.provide_data]
        # ['target_label_e', 'target_label_f']
        label_names = [x[0] for x in train_iter.provide_label]

        def sym_gen(e_seq_len, f_seq_len):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            The computation graph contains two instances of the encoder (input = {e,f})
            and four instances of the decoder (output={e,f} x input)

            :param e_seq_len: The maximum length of the e encoder/decoder
            :param f_seq_len: The maximum length of the f encoder/decoder
            """

            # (int)
            vocab_size = len(self.vocab)

            ##### ENCODER FOR E and F #####
            # Add the embedding "encoder" to the list of encoders.
            self.encoder.encoders = [self.embedding] + self.encoder.encoders

            # Symbols for encoding e and f source input respectively.
            # e_encoded, f_encoded: (seq_len, bs, num_hidden)
            e_encoded = self.encoder.encode(data=e_source, data_length=e_source_length, seq_len=e_seq_len)
            f_encoded = self.encoder.encode(data=f_source, data_length=f_source_length, seq_len=f_seq_len)

            # The autoencoders for e and f
            # e_logits_autoencoder, f_logits_autoencoder: (bs * seq_len, vocab_size)
            e_logits_autoencoder = self.decoder.decode(source_encoded=e_encoded,
                                                       source_seq_len=e_seq_len,
                                                       source_length=e_source_length,
                                                       target=e_target,
                                                       target_seq_len=e_seq_len,
                                                       embedding=self.embedding)

            f_logits_autoencoder = self.decoder.decode(source_encoded=f_encoded,
                                                       source_seq_len=f_seq_len,
                                                       source_length=f_source_length,
                                                       target=f_target,
                                                       target_seq_len=f_seq_len,
                                                       embedding=self.embedding)

            #####################
            def sym_gen_transfer(source_encoded, source_length, seq_len, word_id_initial):
                """
                Creates the computation graph for the transfer generators

                :param source_encoded: The symbol for the output of the encoders
                :param source_length: The length of the source
                :param seq_len: The maximum length to unroll the transfer generators to
                :param word_id_initial: The id of the word to be fed as input at time=0
                :return: The logits of the transfer RNN.
                """
                source_encoded_batch_major = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1,
                                                             name='source_encoded_batch_major')

                # Initialize attention
                attention_func = self.attention.on(source_encoded_batch_major, source_length, seq_len)
                attention_state = self.attention.get_initial_state(source_length, seq_len)

                # Embedding for first lang specific word
                # word_id_prev: (bs,)
                word_id_prev = word_id_initial

                # source_encoded: (seq_len, bs, num_hidden)
                # Decoder state is a tuple of hidden and layer states
                decoder_state = self.decoder.compute_init_states(source_encoded, source_length)

                # We'll accumulate the pre-softmax logits for each timestep here
                transfer_logits_list = []

                # Used to bypass embedding the input word when it is an
                # expected word (and embedding of)
                input_is_embedding = False

                for seq_idx in range(seq_len):
                    # softmax_out : (bs, vocab_size)
                    # logits : (bs, vocab_size)
                    (softmax_out, decoder_state, attention_state, local_logits) \
                        = self.decoder.predict(
                                                word_id_prev=word_id_prev,
                                                state_prev=decoder_state,
                                                attention_func=attention_func,
                                                attention_state_prev=attention_state,
                                                embedding=self.embedding,
                                                word_is_embedding=input_is_embedding
                                              )
                    # Expand logits to : (bs, 1, vocab_size) : to concat later along axis 1
                    transfer_logits_list.append(mx.sym.expand_dims(local_logits, axis=1))

                    # The input to the next time step will be the expected word
                    # (bs, vocab_size) * (vocab_size, num_embed)
                    # This will be an 'embedded' word and not a one hot vector
                    # Its embedding will have to be bypassed by the decoder.
                    word_id_prev = mx.sym.dot(softmax_out, self.embedding.embed_weight)
                    input_is_embedding = True

                # (bs * seq_len) x vocab_size
                transfer_logits = mx.sym.reshape(mx.sym.concat(*transfer_logits_list, dim=1),
                                                 shape=(-1, vocab_size))
                return transfer_logits
            ################

            # Initialize the first words for the two generators
            # (bs,)
            # TODO: Not sure if dtype is required
            f_first_word = mx.sym.BlockGrad(mx.sym.Variable(name='f_bos_transfer_input',
                                           init=mx.init.Constant(value=self.vocab[C.F_BOS_SYMBOL]),
                                           shape=(train_iter.batch_size,),
                                           dtype=np.float32))
            e_first_word = mx.sym.BlockGrad(mx.sym.Variable(name='e_bos_transfer_input',
                                           init=mx.init.Constant(value=self.vocab[C.E_BOS_SYMBOL]),
                                           shape=(train_iter.batch_size,),
                                           dtype=np.float32))

            # f_logits_transfer: (bs * seq_len, vocab_size)
            #TODO: Only send the last input to e_encoded
            f_logits_transfer = sym_gen_transfer(e_encoded, e_source_length, e_seq_len, f_first_word)
            e_logits_transfer = sym_gen_transfer(f_encoded, f_source_length, f_seq_len, e_first_word)
            #f_logits_transfer = e_logits_autoencoder
            #e_logits_transfer = f_logits_autoencoder
            
            # feed the autoencoder output to the discriminator
            # f_D_autoencoder, e_D_autoencoder: (bs, 2) (2 = binary decisions)
            f_D_autoencoder = self.discriminator_f.discriminate(f_logits_autoencoder, f_seq_len,
                                                                vocab_size, f_source_length)

            e_D_autoencoder = self.discriminator_e.discriminate(e_logits_autoencoder, e_seq_len,
                                                                vocab_size, e_source_length)

            # Logits_transfer keeps generating to max_seq_len since there's no stopping condition
            # We post-process to determine EOS and pad the output after an EOS appears
            # This happens for the entire batch
            # The following operation will determine where the EOS occurs (greedy) and pad
            # (Deepak's work)
            # Note that the batch sizes are switched because we care about the input batch size.
            # f_transfer_length, e_transfer_length: (bs,)
            f_transfer_length = mask_labels_after_EOS(f_logits_transfer, train_iter.batch_size, e_seq_len)
            e_transfer_length = mask_labels_after_EOS(e_logits_transfer, train_iter.batch_size, f_seq_len)

            # e_logits_transfer come from f->e, so we use f_batch_size, etc. for e_D_transfer
            # f_D_transfer, e_D_transfer: (bs, 2) (2 = binary decisions)
            e_D_transfer = self.discriminator_e.discriminate(e_logits_transfer, f_seq_len, vocab_size,
                                                             e_transfer_length)
            f_D_transfer = self.discriminator_f.discriminate(f_logits_transfer, e_seq_len, vocab_size,
                                                             f_transfer_length)

            # get labels to train the discriminators
            # for autoencoders (all ones) and transfers (all zeros)
            e_disc_labels_autoencoder = mx.symbol.ones(shape=(train_iter.batch_size,), name='e_disc_labels_ae')
            f_disc_labels_autoencoder = mx.symbol.ones(shape=(train_iter.batch_size,), name='f_disc_labels_ae')
            e_disc_labels_transfer = mx.symbol.zeros(shape=(train_iter.batch_size,), name='e_disc_labels_tr')
            f_disc_labels_transfer = mx.symbol.zeros(shape=(train_iter.batch_size,), name='f_disc_labels_tr')

            # logits_ae_e and logits_tr_f are originally in e
            loss_G, loss_D = loss.get_loss(e_logits_autoencoder, f_logits_autoencoder, e_labels, f_labels,
                                    e_D_autoencoder, e_D_transfer, e_disc_labels_autoencoder, e_disc_labels_transfer,
                                    f_D_autoencoder, f_D_transfer, f_disc_labels_autoencoder, f_disc_labels_transfer)
            print(loss_D.list_arguments())
            return mx.sym.Group([loss_G, loss_D]), data_names, label_names

        # TODO: Add bucketing later

        logger.info("No bucketing. Unrolled to max_seq_len=%s or %s :(", e_max_seq_len, f_max_seq_len)
        # TODO don't know why the e_train_iter.buckets[0] didn't work..
        symbol, _, _ = sym_gen(e_max_seq_len, f_max_seq_len)
        #symbol, _, __ = sym_gen(e_train_iter.buckets[0], f_train_iter.buckets[0])
        return mx.mod.Module(symbol=symbol,
                             data_names=data_names,
                             label_names=label_names,
                             logger=logger,
                             context=self.context)

    # TODO redo this so it makes more sense for our case (ex. we will never use accuracy)
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
                # TODO for now we will just use the autoencoder loss for style transfer..
                # TODO change this because it makes no sense -- want equilibrium
                metrics.append(mx.metric.Perplexity(ignore_label=C.PAD_ID, output_names=[C.GAN_LOSS + '_g_output']))
            else:
                raise ValueError("unknown metric name")
        return mx.metric.create(metrics)

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
            use_tensorboard: bool = False):
        """
        Fits model to data given by train_iter using early-stopping w.r.t data given by val_iter.
        Saves all intermediate and final output to output_folder

        :param train_iter: The training data iterator for language e and f.
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

        # TODO this needs to be fixed..
        data_shapes = train_iter.provide_data
        label_shapes = train_iter.provide_label

        self.module.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=True,
                         force_rebind=True, grad_req='write')

        self.module.symbol.save(os.path.join(output_folder, C.SYMBOL_NAME))

        self.module.init_params(initializer=initializer, arg_params=self.params, aux_params=None,
                                allow_missing=False, force_init=False)

        self.module.init_optimizer(kvstore='device', optimizer=optimizer, optimizer_params=optimizer_params)

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
        val_iter = None
        self._fit(train_iter, output_folder,
                  metrics=metrics,
                  max_updates=max_updates,
                  checkpoint_frequency=checkpoint_frequency,
                  max_num_not_improved=max_num_not_improved,
                  min_num_epochs=min_num_epochs)

        logger.info("Training finished. Best checkpoint: %d. Best validation %s: %.6f",
                    self.training_monitor.get_best_checkpoint(),
                    self.training_monitor.optimized_metric,
                    self.training_monitor.get_best_validation_score())
        return self.training_monitor.get_best_validation_score()

    # TODO: There's no val iter here.
    def _fit(self,
             train_iter: sockeye.data_io.ParallelBucketSentenceIter,
             output_folder: str,
             metrics: List[AnyStr],
             max_updates: int,
             checkpoint_frequency: int,
             max_num_not_improved: int,
             min_num_epochs: Optional[int] = None):
        """
        Internal fit method. Runtime determined by early stopping.

        :param e_train_iter: Training data iterator for language e.
        :param f_train_iter: Training data iterator for language f.
        :param val_iter: Validation data iterator.
        :param output_folder: Model output folder.
        :param metrics: List of metric names to track on training and validation data.
        :param max_updates: Maximum number of batches to process.
        :param checkpoint_frequency: Frequency of checkpointing.
        :param max_num_not_improved: Maximum number of checkpoints until fitting is stopped if model does not improve.
        :param min_num_epochs: Minimum number of epochs to train, even if validation scores did not improve.
        """
        # cross-entropy metric (and labels) for the discriminators TODO move this with the other metric..
        ce_D = mx.metric.create('ce')
        # labels are (4*batch_size,) and are 111...000...111...000...
        # TODO not sure if there is a better place to put this (esp. to guarantee that order is correct)
        loss_D_labels = mx.nd.concat(mx.nd.ones((train_iter.batch_size,)), mx.nd.zeros((train_iter.batch_size,)),
                                     mx.nd.ones((train_iter.batch_size,)), mx.nd.zeros((train_iter.batch_size,)),
                                     dim=0)

        metric_train = self._create_eval_metric(metrics)
        tic = time.time()

        training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
        if os.path.exists(training_state_dir):
            train_state = self.load_checkpoint(training_state_dir, train_iter)
        else:
            train_state = _StyleTrainingState(
                num_not_improved=0,
                epoch=0,
                checkpoint=0,
                updates=0,
                samples=0
            )

        # batch details
        # bs = 20, max_seq_len=50
        # source_e, (20, 50), source_length_e, (20, ), target_e, (20, 50)
        # source_f, (20, 50), source_length_f, (20, ), target_f, (20, 50)
        # target_label_e, (20, 50), target_label_f, (20, 50)
        next_data_batch = train_iter.next()

        while max_updates == -1 or train_state.updates < max_updates:
            # If any of these iterators runs out of things to produce
            # we call it an event and reset both iterators
            if not train_iter.iter_next():
                train_state.epoch += 1
                train_iter.reset()

            # process batch
            # This is a batch with the training data for e and f
            batch = next_data_batch
            self.module.forward_backward(batch)
            _, loss_D = self.module.get_outputs()
#            ce_D.update(loss_D_labels, [loss_D]) # TODO: print this
            self.module.update()

            # NOTE: batch.label is [label_e, label_f] so we concatenate them
            # TODO is there a better way of doing this so that we can ensure order is the same?
            self.module.update_metric(metric_train, [mx.nd.concat(*batch.label)])
            self.training_monitor.batch_end_callback(train_state.epoch, train_state.updates, metric_train)
            train_state.updates += 1
            train_state.samples += train_iter.batch_size

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

                # TODO:evaluation on validation set
                has_improved = True
                best_checkpoint = True
                # TODO will need to fix the lr_scheduler instead of always saying has_improved=True
                # has_improved, best_checkpoint = self._evaluate(train_state, val_iter, metric_val)
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

                self._checkpoint(train_state, output_folder, train_iter)

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

    def _checkpoint(self, training_state: _StyleTrainingState,
                    output_folder: str,
                    train_iter: sockeye.data_io.ParallelBucketSentenceIter):
        """
        Saves checkpoint. Note that the parameters are saved in _save_params.
        """
        # Create temporary directory for storing the state of the optimization process
        training_state_dirname = os.path.join(output_folder, C.TRAINING_STATE_TEMP_DIRNAME)
        if not os.path.exists(training_state_dirname):
            os.mkdir(training_state_dirname)
        # Link current parameter file
        params_base_fname = C.PARAMS_NAME % training_state.checkpoint
        os.symlink(os.path.join("..", params_base_fname), os.path.join(training_state_dirname, C.TRAINING_STATE_PARAMS_NAME))

        # Optimizer state (from mxnet)
        opt_state_fname = os.path.join(training_state_dirname, C.MODULE_OPT_STATE_NAME)
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
        train_iter.save_state(os.path.join(training_state_dirname, C.BUCKET_ITER_STATE_NAME))

        # RNG states: python's random and np.random provide functions for
        # storing the state, mxnet does not, but inside our code mxnet's RNG is
        # not used AFAIK
        with open(os.path.join(training_state_dirname, C.RNG_STATE_NAME), "wb") as fp:
            pickle.dump(random.getstate(), fp)
            pickle.dump(np.random.get_state(), fp) # Yes, one uses _, the other does not

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

    def save_state(self, training_state: _StyleTrainingState, fname: str):
        """
        Saves the state (of the TrainingModel class) to disk.

        :param fname: file name to save the state to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(training_state, fp)

    def load_state(self, fname: str) -> _StyleTrainingState:
        """
        Loads the training state (of the TrainingModel class) from disk.

        :param fname: file name to load the state from.
        """
        training_state = None
        with open(fname, "rb") as fp:
            training_state = pickle.load(fp)
        return training_state

    def load_checkpoint(self, directory: str,
                        train_iter: sockeye.data_io.ParallelBucketSentenceIter) -> _StyleTrainingState:
        """
        Loads the full training state from disk. This includes optimizer,
        random number generators and everything needed.  Note that params
        should have been loaded already by the initializer.

        :param directory: directory where the state has been saved.
        :param train_iter: training data iterator.
        """

        # Optimzer state (from mxnet)
        opt_state_fname = os.path.join(directory, C.MODULE_OPT_STATE_NAME)
        if self.bucketing:
            # Same hacky solution as for saving the state
            self.module._curr_module.load_optimizer_states(opt_state_fname)
        else:
            self.module.load_optimizer_states(opt_state_fname)

        # State of the bucket iterator
        train_iter.load_state(os.path.join(directory, C.BUCKET_ITER_STATE_NAME))

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
