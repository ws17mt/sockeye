#!/usr/bin/env bash

cd ..
python setup.py install --user
cd sockeye

rm -rf tmp/*

python train_style.py
