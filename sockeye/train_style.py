import os

import sockeye.data_io
from sockeye.log import setup_main_logger

from train import _build_or_load_vocab

logger = setup_main_logger(__name__, file_logging=False, console=True)

source = "/export/b09/ws15gkumar/experiments/JSALT17/code/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok"
target = "/export/b09/ws15gkumar/experiments/JSALT17/code/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok"

num_words = 50000
word_min_count = 1

# Build vocab
vocab_source = _build_or_load_vocab(None, source, num_words, word_min_count)
vocab_target = _build_or_load_vocab(None, target, num_words, word_min_count)

vocab_source_size = len(vocab_source)
vocab_target_size = len(target_source)
logger.info("Vocabulary sizes: source=%d target=%d", vocab_source_size, vocab_target_size)

data_info = sockeye.data_io.StyleDataInfo(os.path.abspath(source),
                                          os.path.abspath(target),
                                          vocab_source,
                                          vocab_target)


