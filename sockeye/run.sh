#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

DATA_HOME=/Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k
OUT_FOLDER=tmp

mkdir -p ${OUT_FOLDER}
rm -rf ${OUT_FOLDER}/*

python train_style.py \
    -s ${DATA_HOME}/train-toy.de.atok \
    -t ${DATA_HOME}/train-toy.en.atok \
    -ms ${DATA_HOME}/train-toy.de.atok \
    -mt ${DATA_HOME}/train-toy.en.atok \
    -vs ${DATA_HOME}/val.de.atok \
    -vt ${DATA_HOME}/val.en.atok \
    -o  $OUT_FOLDER \
    -b 20 \
    --num-embed 4 \
    --attention-num-hidden 5 \
    --attention-type mlp \
    --rnn-cell-type gru \
    --rnn-num-layers 1 \
    --rnn-num-hidden 6 \
    --dropout 0.2 \
    --initial-learning-rate 0.0002 \
    --num-words 10000 \
    --word-min-count 1 \
    --max-seq-len 50 \
    --disc-num-hidden 7 \
    --disc-num-layers 1 \
    --disc-dropout 0.0 \
    --disc-act softrelu \
    --use-cpu \
    --no-bucketing \
    --bucket-width 100 \
    --loss gan-loss \
    --valid-loss cross-entropy \
    --disc-loss-lambda 50000.0 \
    --max-updates 100 \
    --checkpoint-frequency 100


#    --joint_vocab None
#    --weight-tying \
