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
state = 'Maharashtra'

data_dict = {}
for filename in all_files:
    basename = os.path.basename(filename)
    basename = re.sub('.csv', '', basename)
    if basename == 'ICMRTestingLabs':
        continue
    df = pd.read_csv(filename, index_col=None, header=0)
    data_dict[basename] = df
    del df
    
# cases
def filter_df(df, state_extract = 'All'):
    if state_extract == 'All':
        df = df
    else:
        df = df.loc[df.loc[:, 'State/UnionTerritory'] == state, :]
    
    # reset index
    df = df.reset_index()
    
    return df

dia_df = filter_df(deepcopy(data_dict['covid_19_india']), state)
dia_df['Date'] = pd.to_datetime(dia_df['Date'], format = '%d/%m/%y')
dia_df['day'] = 0
dia_df['National confirmed cumulative'] = 0
dia_df['National confirmed new'] = 0
dia_df['National recovered cumulative'] = 0
dia_df['National recovered new'] = 0
dia_df['National deceased cumulative'] = 0
dia_df['National deceased new'] = 0

# aux function
def update_national_cases(row, day, state, var):
    
    # get cumulative number of 'var'
    total = dia_df[var].loc[row]
    
    # check if cases have incread and by how much
    if state_cur in cases[var].keys():
        if cases[var][state_cur] < total:
            new_cases = total - cases[var][state_cur]
        else:
            new_cases = 0
        cases[var][state_cur] = total
    else:
        new_cases = total
        cases[var][state_cur] = total
    
    # new name dictionary
    cumu = new_name[var][0]
    inci = new_name[var][1]
    
    # update confirmed cumulative and new
    if row > 0:
        dia_df[cumu].loc[row] = dia_df[cumu].loc[row-1] + new_cases
    else:
        dia_df[cumu].loc[row] = new_cases
    dia_df[inci].loc[row] = new_cases
    
    return
    

# create a sequence for all cases in india
cases = {'Confirmed': {},
         'Deaths': {},
         'Cured': {}}
new_name = {'Confirmed': ['National confirmed cumulative', 'National confirmed new'],
            'Deaths': ['National deceased cumulative', 'National deceased new'],
            'Cured': ['National recovered cumulative', 'National recovered new']}
date_prev = dia_df['Date'].iloc[0]
day = 0
#state_prev = dia_df['State/UnionTerritory'].loc[0]
for row in dia_df.index:
    day = (dia_df['Date'].loc[row] - date_prev).days
    dia_df['day'].loc[row] = day
    
    # get current state and update cumulative number
    state_cur = dia_df['State/UnionTerritory'].loc[row]
    
    # update confirmed, deceases and recovered cases
    for var in cases.keys():
        update_national_cases(row, day, state_cur, var)
        
# save csv
dia_df.to_csv(os.path.join(path, ('calibration_data_for_%s_state.csv')%(state)))