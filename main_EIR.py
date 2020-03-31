#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:02:11 2020

@author: vijetadeshpande
"""
import EIRModel
import pandas as pd
import numpy as np

# rate matrix file path
file_path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/rate_matrix.xlsx'

# initialize model
covid = EIRModel.EIRModel(initial_pop = np.array([40, 10, 0, 0, 0, 0, 0]), 
                          rate_matrix_file_path = file_path)

# loop over some horizon
pop_prev = covid.init_pop
social_dist = 'no'

for day in range(0, 40):
    if (day == 21 or day == 35):
        social_dist = ('yes' * int(day == 14)) + ('q' * int(day == 30))
    for t in range(0, int(1/covid.unit_time)):
        pop_cur = covid.forward(pop_prev, social_dist)
        pop_prev = pop_cur
        pop_cur = 0
