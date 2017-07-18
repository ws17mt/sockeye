python -m sockeye.train --source data/toy/train.en.tok --target data/toy/train.de.tok --source-metadata data/toy/train.en.deps --validation-source data/toy/val.en.tok --validation-target data/toy/val.de.tok --val-source-metadata data/toy/val.en.deps --use-cpu --output toy_model --batch-size 2 --no-bucketing --rnn-num-hidden 32 --num-embed 32
rm -rf toy_model
