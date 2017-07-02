# This is an implementation of the following work:
# Dual Learning for Machine Translation
# Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
# https: //arxiv.org/abs/1611.00179 (accepted at NIPS 2016)
# Developed by Vu Cong Duy Hoang (vhoang2@student.unimelb.edu.au)
# Date: July 2017
# ---------------------------------------------------------------------------------

"""
Simple Training CLI.
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
from contextlib import ExitStack
from typing import Optional, Dict

import mxnet as mx
import numpy as np

import sockeye.arguments as arguments
import sockeye.attention
import sockeye.constants as C
import sockeye.data_io
import sockeye.decoder
import sockeye.encoder
import sockeye.initializer
import sockeye.lexicon
import sockeye.lr_scheduler
import sockeye.model
import sockeye.training
import sockeye.utils
import sockeye.vocab
from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, expand_requested_device_ids

def main():
	#FIXME

if __name__ == "__main__":
    main()

