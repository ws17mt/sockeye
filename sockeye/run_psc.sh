#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

DATA_HOME=/pylon2/ci560op/fosterg/data/fr/processed
OUT_FOLDER=tmp

python train_style.py \
    -s ${DATA_HOME}/train.10k.bpe.fr \
    -t ${DATA_HOME}/train.10k.bpe.en \
    -ms ${DATA_HOME}/train.10k.bpe.fr \
    -mt ${DATA_HOME}/train.10k.bpe.en \
    -vs ${DATA_HOME}/dev1.bpe.fr \
    -vt ${DATA_HOME}/dev1.bpe.en \
    --joint-vocab ${DATA_HOME}/vocab.enfr.json \
    -o $OUT_FOLDER \
    --rnn-cell-type gru \
    --rnn-num-hidden 512 \
    --dropout 0.5 \
    --max-seq-len 80 \
    --disc-num-hidden 500 \
    --disc-num-layers 2 \
    --disc-dropout 0.0 \
    --disc-act softrelu \
    --no-bucketing \
    --loss gan-loss \
    --valid-loss cross-entropy \
    --disc-loss-lambda 10.0 \
    --max-updates -1 \
    --checkpoint-frequency 400 \
    --no-bucketing \
    --weight-tying \
    --seed 1
    --normalize-loss \
    --disc-batch-norm \
    --g-loss-weight 100.0

