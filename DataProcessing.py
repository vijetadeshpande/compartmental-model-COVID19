#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:51:51 2020

@author: vijetadeshpande
"""

import pandas as pd
import numpy as np
import glob

path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/Data/India' # use your path
all_files = glob.glob(path + "/*.csv")

data_dict = {}
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data_dict[filename] = df
