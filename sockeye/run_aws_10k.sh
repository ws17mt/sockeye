#!/usr/bin/env bash

set -e

SOCKEYE=/home/ubuntu/sockeye/sockeye

DATA_HOME=/efs/data/fr/processed
OUT_FOLDER=/efs/gkumar/exp/par_10k_mono_100k

ARGS="
    -s ${DATA_HOME}/train.10k.bpe.fr \
    -t ${DATA_HOME}/train.10k.bpe.en \
    -ms ${DATA_HOME}/train.100k.bpe.fr \
    -mt ${DATA_HOME}/train.100k.bpe.en \
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
    --seed 1 \
    --normalize-loss \
    --disc-batch-norm \
    --g-loss-weight 100.0
"

if [ ! -d ${OUT_FOLDER}/g_pretrain/training_state ] && [ -f ${OUT_FOLDER}/g_pretrain/params.best ]; then
  echo "[RUN:] Skipping pre-training of G. Complete model exists at $OUT_FOLDER/g_pretrain/params.best"
else
  echo "[RUN:] Starting pre-training of G "
  python $SOCKEYE/pretrain_g_style.py $ARGS
  echo "[RUN:] Finished pre-training of G "
fi

if [ ! -d ${OUT_FOLDER}/d_pretrain/training_state ] && [ -f ${OUT_FOLDER}/d_pretrain/params.best ]; then
  echo "[RUN:] Skipping pre-training of D. Complete model exists at $OUT_FOLDER/d_pretrain/params.best"
else
  echo "[RUN:] Starting pre-training of D "
  python $SOCKEYE/pretrain_d_style.py $ARGS
  echo "[RUN:] Starting pre-training of D "
fi

echo "[RUN:] Starting joint training "
python $SOCKEYE/train_style.py $ARGS
echo "[RUN:] Finished joint training "
