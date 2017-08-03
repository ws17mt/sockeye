#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

rm -rf tmp/*

python train_style.py \
    -s /Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.de.atok \
    -t /Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/train-toy.en.atok \
    -vs /Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/val.de.atok \
    -vt /Users/gaurav/Dropbox/Projects/JSALT17-NMT-Lab/data/multi30k/val.en.atok \
    -o tmp \
    -b 20 \
    --num-embed 4 \
    --attention-num-hidden 5 \
    --attention-type mlp \
    --rnn-cell-type gru \
    --rnn-num-layers 1 \
    --rnn-num-hidden 6 \
    --dropout 0.1 \
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
