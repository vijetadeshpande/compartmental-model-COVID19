#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:51:51 2020

@author: vijetadeshpande
"""

import pandas as pd
import numpy as np
import glob
import re
import os
from copy import deepcopy

path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/Data/India'
all_files = glob.glob(path + "/*.csv")

data_dict = {}
for filename in all_files:
    basename = os.path.basename(filename)
    basename = re.sub('.csv', '', basename)
    if basename == 'ICMRTestingLabs':
        continue
    df = pd.read_csv(filename, index_col=None, header=0)
    data_dict[basename] = df
    del df
    
# cases in punjab
dia_df = deepcopy(data_dict['covid_19_india'])
dia_df['Date'] = pd.to_datetime(dia_df['Date'], format = '%d/%m/%y')
dia_df['day'] = 0
dia_df['National confirmed cumulative'] = 0
dia_df['National confirmed new'] = 0
dia_df['National recovered cumulative'] = 0
dia_df['National recovered new'] = 0
dia_df['National deceased cumulative'] = 0
dia_df['National deceased new'] = 0


# create a sequence for all cases in india
state_total = {}
state_rec = {}
state_dec = {}
date_prev = dia_df['Date'].iloc[0]
day = 0
#state_prev = dia_df['State/UnionTerritory'].loc[0]
for row in dia_df.index:
    day = (dia_df['Date'].loc[row] - date_prev).days
    dia_df['day'].loc[row] = day
    
    # get current state and update cumulative number
    state_cur = dia_df['State/UnionTerritory'].loc[row]
    total = dia_df['Confirmed'].loc[row]
    
    # check if cases have incread and by how much
    if state_cur in state_total.keys():
        if state_total[state_cur] < total:
            new_cases = total - state_total[state_cur]
        else:
            new_cases = 0
        state_total[state_cur] = total
    else:
        new_cases = total
        state_total[state_cur] = total
    
    # update confirmed cumulative and new
    if row > 0:
        dia_df['National confirmed cumulative'].loc[row] = dia_df['National confirmed cumulative'].loc[row-1] + new_cases
    else:
        dia_df['National confirmed cumulative'].loc[row] = new_cases
    dia_df['National confirmed new'].loc[row] = new_cases
        
# save csv
dia_df.to_csv(os.path.join(path, 'calibration_data.csv'))