#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

rm -rf tmp/*

python train_style.py \
    -s /pylon2/ci560op/fosterg/data/fr/processed/mono.100k.bpe.fr \
    -t /pylon2/ci560op/fosterg/data/fr/processed/mono.100k.bpe.en \
    -vs /pylon2/ci560op/fosterg/data/fr/processed/dev1.bpe.fr \
    -vt /pylon2/ci560op/fosterg/data/fr/processed/dev1.bpe.en \
    --joint-vocab /pylon2/ci560op/fosterg/data/fr/processed/vocab.enfr.json
    -o tmp \
    -b 64 \
    --num-embed 500 \
    --attention-num-hidden 1024 \
    --attention-type mlp \
    --rnn-cell-type gru \
    --rnn-num-layers 2 \
    --rnn-num-hidden 500 \
    --dropout 0.3 \
    --num-words 10000 \
    --word-min-count 1 \
    --max-seq-len 50 \
    --disc-num-hidden 500 \
    --disc-num-layers 2 \
    --disc-dropout 0.0 \
    --disc-act softrelu \
    --no-bucketing \
    --bucket-width 100 \
    --loss gan-loss \
    --valid-loss cross-entropy \
    --disc-loss-lambda 50000.0 \
    --max-updates -1 \
    --checkpoint-frequency 1000

#--use-cpu \