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


def init_random(seed):
    np.random.seed(seed)  # numpy
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)
