python -m sockeye.train --source data/toy2/train.en.tok --target data/toy2/train.de.tok --validation-source data/toy2/val.en.tok --validation-target data/toy2/val.de.tok --use-cpu --output toy_model --batch-size 2 --rnn-num-hidden 32 --num-embed 32 --checkpoint-frequency 50 --edge-vocab data/toy2/edge_vocab.json
rm -rf toy_model
