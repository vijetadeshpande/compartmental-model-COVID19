#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:02:11 2020

@author: vijetadeshpande
"""
import EIRModel
import pandas as pd
import numpy as np

# import data for calibration
path_data = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/Data/India/calibration_data.csv'
data_cali = pd.read_csv(path_data)
new_infections = np.array(data_cali['National confirmed new'].values)
diagnosed_cases = np.array(data_cali['National confirmed cumulative'].values)

# rate matrix file path
file_path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/rate_matrix_old.xlsx'

# initialize model
covid = EIRModel.EIRModel(initial_pop = np.array([0, 0, 4, 0, 0, 0, 0, 0]), 
                          rate_matrix_file_path = file_path)
covid.calibration_data = new_infections

# loop over some horizon
pop_prev = covid.pop_day_start
social_dist = 'no'

for day in range(0, 60):
    # update population at start of day and day 
    covid.day_count = day
    covid.pop_day_start = pop_prev
    
    # fit diagnosis rate for this day
    covid.backward(day)
    
    # forward pass to get population for next day
    pop_cur = covid.forward(pop_prev, social_dist)
    covid.pop_day_end = pop_prev
    
    # update previous pop for next day
    pop_prev = pop_cur
    
    
    
    
    

