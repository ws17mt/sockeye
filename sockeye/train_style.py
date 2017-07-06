import os

import sockeye.data_io
from sockeye.log import setup_main_logger

from train import _build_or_load_vocab

logger = setup_main_logger(__name__, file_logging=False, console=True)

source = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok"
target = "/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok"

num_words = 50000
word_min_count = 1
batch_size = 50
max_seq_len=200

# Build vocab
vocab_source = _build_or_load_vocab(None, source, num_words, word_min_count)
vocab_target = _build_or_load_vocab(None, target, num_words, word_min_count)

vocab_source_size = len(vocab_source)
vocab_target_size = len(vocab_target)
logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)

data_info = sockeye.data_io.StyleDataInfo(os.path.abspath(source),
                                          os.path.abspath(target),
                                          vocab_source,
                                          vocab_target)

source_train_iter = sockeye.data_io.get_style_training_data_iters(
                        source=data_info.source,
                        vocab=vocab_source,
                        batch_size=batch_size,
                        fill_up=True,
                        max_seq_len=max_seq_len,
                        bucketing=False,
                        bucket_width=100
                    )

target_train_iter = sockeye.data_io.get_style_training_data_iters(
                        source=data_info.target,
                        vocab=vocab_target,
                        batch_size=batch_size,
                        fill_up=True,
                        max_seq_len=max_seq_len,
                        bucketing=False,
                        bucket_width=100
                    )
