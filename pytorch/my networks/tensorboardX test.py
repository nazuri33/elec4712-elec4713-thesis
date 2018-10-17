# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:33:46 2018

@author: owenm
"""
from tensorboardX import SummaryWriter
import random
import numpy as np

summary_writer = SummaryWriter(log_dir = f"test_log/exp_{random.randint(0,100)}")

for i in range(10):
    summary_writer.add_scalar("training/loss", np.random.rand(), i)
    summary_writer.add_scalar("validation/loss", np.random.rand() + .1, i)
    
