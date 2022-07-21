import argparse
import copy
import datetime
import imp
import math
import os
import pickle
import random
import re

import numpy as np
import torch

RANDOM_SEED = 42


def init_random(RANDOM_SEED):
    np.random.seed(RANDOM_SEED)  # numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)  # cpu
    torch.cuda.manual_seed(RANDOM_SEED)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)
