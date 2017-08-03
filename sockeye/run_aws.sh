#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

DATA_HOME=/efs/data/fr/processed
OUT_FOLDER=tmp

mkdir -p ${OUT_FOLDER}
rm -rf ${OUT_FOLDER}/*

python train_style.py \
    -s ${DATA_HOME}/train.100k.bpe.fr \
    -t ${DATA_HOME}/train.100k.bpe.en \
    -vs ${DATA_HOME}/dev1.bpe.fr \
    -vt ${DATA_HOME}/dev1.bpe.en \
    --joint-vocab ${DATA_HOME}/vocab.enfr.json \
    -o $OUT_FOLDER \
    -b 128 \
    --num-embed 512 \
    --attention-num-hidden 1024 \
    --attention-type mlp \
    --rnn-cell-type gru \
    --rnn-num-layers 2 \
    --rnn-num-hidden 512 \
    --dropout 0.3 \
    --initial-learning-rate 0.0002 \
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
    --checkpoint-frequency 1000 \
    --no-bucketing

    #--weight-tying \
