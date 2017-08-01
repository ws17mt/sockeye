import os

import mxnet as mx

import sockeye.data_io
import sockeye.style_training
import sockeye.inference
import sockeye.encoder
import sockeye.constants as C
from sockeye.log import setup_main_logger

from sockeye.train import _build_or_load_vocab

logger = setup_main_logger(__name__, file_logging=False, console=True)

#### ANNA ####
# e_corpus = "/home/acurrey/labs/nmt/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok"
# f_corpus = "/home/acurrey/labs/nmt/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok"
# Convention: Source (e), Target (f)
#### GAURAV #####
#e_corpus = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok"
#f_corpus = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok"
#e_val = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/val.de.atok"
#f_val = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/val.en.atok"
#### PSC (de-en) ####
#e_corpus = "/pylon2/ci560op/gkumar6/code/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok"
#f_corpus = "/pylon2/ci560op/gkumar6/code/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok"
#e_val = "/pylon2/ci560op/gkumar6/code/JSALT17-NMT-Lab/data/multi30k/val.de.atok"
#f_val = "/pylon2/ci560op/gkumar6/code/JSALT17-NMT-Lab/data/multi30k/val.en.atok"
#### PSC (fr-en) ####
e_corpus = "/pylon2/ci560op/fosterg/data/fr/processed/mono.100k.bpe.fr"
f_corpus = "/pylon2/ci560op/fosterg/data/fr/processed/mono.100k.bpe.en"
e_val = "/pylon2/ci560op/fosterg/data/fr/processed/dev1.bpe.fr"
f_val = "/pylon2/ci560op/fosterg/data/fr/processed/dev1.bpe.en"

external_vocab = "/pylon2/ci560op/fosterg/data/fr/processed/vocab.enfr.json"

output_folder="tmp"

# TODO: hard-coded stuff; remove when user args are back.
lr_scheduler = None
num_embed = 500
attention_type="fixed" # TODO:Fix
attention_num_hidden = 1024
dropout=0.3
rnn_cell_type=C.GRU_TYPE
rnn_num_layers=2
rnn_num_hidden=500
num_words = 10000
word_min_count = 1
batch_size = 64
max_seq_len=50
disc_num_hidden=500
disc_num_layers=1
disc_dropout=0.
disc_act='softrelu'

#lr_scheduler = None
#num_embed = 4
#attention_type="fixed" # TODO:Fix
#attention_num_hidden = 5
#dropout=0.1
#rnn_cell_type=C.GRU_TYPE
#rnn_num_layers=1
#rnn_num_hidden=6
#num_words = 10000
#word_min_count = 1
#batch_size = 20
#max_seq_len=50
#disc_num_hidden=7
#disc_num_layers=1
#disc_dropout=0.
#disc_act='relu'

# TODO: Device selection hardcoded to use CPU
context = [mx.gpu()]

# Build vocab
# These vocabs are built on the training data.
# Joint vocab for e and f
# TODO: Is there a way to reload vocab from somewhere? (E.g., BPE dict)
vocab = _build_or_load_vocab(external_vocab, [e_corpus, f_corpus], num_words, word_min_count)

vocab_size = len(vocab)
logger.info("Vocabulary size (merged e, f): %d", vocab_size)

# NamedTuple which will keep track of stuff
data_info = sockeye.data_io.StyleDataInfo(os.path.abspath(e_corpus),
                                          os.path.abspath(f_corpus),
                                          os.path.abspath(e_val),
                                          os.path.abspath(f_val),
                                          vocab)

# This will return a ParallelBucketIterator
# For these, target is always = source (autenc target output)
# Vocabularies are shared across e and f
e_train_iter = sockeye.data_io.get_style_training_data_iters(
                        source=data_info.e,
                        vocab=vocab,
                        batch_size=batch_size,
                        fill_up=True,
                        max_seq_len=max_seq_len,
                        bucketing=False,
                        bucket_width=100,
                        target_bos_symbol=C.E_BOS_SYMBOL,
                        suffix='_e'
                    )

# Similar iter for f
f_train_iter = sockeye.data_io.get_style_training_data_iters(
                        source=data_info.f,
                        vocab=vocab,
                        batch_size=batch_size,
                        fill_up=True,
                        max_seq_len=max_seq_len,
                        bucketing=False,
                        bucket_width=100,
                        target_bos_symbol=C.F_BOS_SYMBOL,
                        suffix='_f'
                    )

# Validation iter
val_iter = sockeye.data_io.get_style_training_data_iters(
                        source=data_info.e_val,
                        target=data_info.f_val,
                        vocab=vocab,
                        batch_size=batch_size,
                        fill_up=True,
                        max_seq_len=max_seq_len,
                        bucketing=False,
                        bucket_width=100,
                        target_bos_symbol=C.F_BOS_SYMBOL,
                        suffix='_val_e',
                        target_suffix='_val_f'
                    )

# Merge the two iterators to get one.
train_iter = mx.io.PrefetchingIter([e_train_iter, f_train_iter])

# TODO: Look at the model config in train.py
# This has several "simple" options to make things work
# Also, the redundancy is args is to be compatible with the sockeye model config
# Effectively, these args are the same for "source" and "target".
# Source and target do not mean the same thing with the style model as they do with the
# traditional NMT models
model_config = sockeye.model.ModelConfig(max_seq_len=max_seq_len,
                                         vocab_source_size=vocab_size,
                                         vocab_target_size=vocab_size,
                                         num_embed_source=num_embed,
                                         num_embed_target=num_embed,
                                         attention_type=attention_type,
                                         attention_num_hidden=attention_num_hidden,
                                         attention_coverage_type="count",
                                         attention_coverage_num_hidden=1,
                                         attention_use_prev_word=False,
                                         dropout=dropout,
                                         rnn_cell_type=rnn_cell_type,
                                         rnn_num_layers=rnn_num_layers,
                                         rnn_num_hidden=rnn_num_hidden,
                                         rnn_residual_connections=False,
                                         weight_tying=False,
                                         context_gating=False,
                                         lexical_bias=False,
                                         learn_lexical_bias=False,
                                         data_info=data_info,
                                         loss=C.GAN_LOSS,
                                         normalize_loss=False,
                                         smoothed_cross_entropy_alpha=0.3,
                                         disc_act=disc_act,
                                         disc_num_hidden=disc_num_hidden,
                                         disc_num_layers=disc_num_layers,
                                         disc_dropout=disc_dropout,
                                         loss_lambda=50000.0)

model = sockeye.style_training.StyleTrainingModel(model_config=model_config,
                                                  context=context,
                                                  train_iter=train_iter,
                                                  fused=False,
                                                  bucketing=False,
                                                  lr_scheduler=lr_scheduler,
                                                  rnn_forget_bias=0.0,
                                                  vocab=vocab)

# For lexical bias, set to None
lexicon = None

initializer = sockeye.initializer.get_initializer(C.RNN_INIT_ORTHOGONAL, lexicon=lexicon)

optimizer = 'adam'
optimizer_params = {'wd': 0.0,
                    "learning_rate": 0.0003}

clip_gradient = None
# Making MXNet module API's default scaling factor explicit
optimizer_params["rescale_grad"] = 1.0 / batch_size
optimizer_params["clip_gradient"] = 1.0

logger.info("Optimizer: %s", optimizer)
logger.info("Optimizer Parameters: %s", optimizer_params)

model.fit(train_iter=train_iter,
          val_iter=val_iter,
          output_folder=output_folder,
          metrics=[C.PERPLEXITY],
          initializer=initializer,
          max_updates=-1,
          checkpoint_frequency=100,
          optimizer=optimizer, optimizer_params=optimizer_params,
          optimized_metric=C.PERPLEXITY,
          max_num_not_improved=8,
          min_num_epochs=0,
          monitor_bleu=0,
          use_tensorboard=False)
