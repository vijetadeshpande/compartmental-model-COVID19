#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:02:11 2020

@author: vijetadeshpande
"""
import EIRModel
import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

# import data for calibration
path_data = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/Data/India/calibration_data.csv'
data_cali = pd.read_csv(path_data)

# extract data for calibration and visualization 
new_infections = deepcopy(data_cali.loc[:, ['day', 'National confirmed new']])
cumulative_infections = deepcopy(data_cali.loc[:, ['day', 'National confirmed cumulative']]).groupby(by = 'day').max()
cumulative_deaths = deepcopy(data_cali.loc[:, ['day', 'National deceased cumulative']]).groupby(by = 'day').max()
cumulative_rec = deepcopy(data_cali.loc[:, ['day', 'National recovered cumulative']]).groupby(by = 'day').max()

# rate matrix file path
file_path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/rate_matrix.xlsx'

# initialize model
covid = EIRModel.EIRModel(initial_pop = np.array([0, 0, 0, 4, 0, 0, 0, 0, 0]), 
                          rate_matrix_file_path = file_path)
covid.calibration_data = new_infections.groupby(by = 'day').sum().loc[:, 'National confirmed new'].values

# loop over some horizon
pop_prev = covid.pop_day_start
social_dist = 'no'

for day in range(0, 70):
    
    if day == 40:
        print('are thamb ki')
    
    # update population at start of day and day 
    covid.day_count = day
    covid.pop_day_start = pop_prev
    
    # fit diagnosis rate for this day
    covid.backward(day)
    
    # forward pass to get population for next day
    pop_cur = covid.forward(pop_prev, social_dist)
    covid.pop_day_end = pop_cur
    
    # store population
    covid.epi_markers['population'].append(pop_cur)
    
    # update previous pop for next day
    pop_prev = pop_cur

# collect population at each time step
pop_t = covid.epi_markers['population']

# visualization
deaths = []
infections = []
rec = []
testing = []
test_rate = covid.epi_markers['testing rate']
index = 0
for pop in pop_t:
    deaths.append(pop[len(pop)-1])
    infections.append(sum(pop[4:7]))
    rec.append(pop[6])
    testing.append(test_rate[index][0])
    index += 1
    
# death plot
death_df = pd.DataFrame(0, index = np.arange(0,70*2), columns = ['day', 'Cumulative Deaths', 'Source'])
death_df['day'].iloc[0:70] = np.arange(0, 70)
death_df['Cumulative Deaths'].iloc[0:70] = deaths
death_df['Source'].iloc[0:70] = 'Model estimates'
death_df['day'].iloc[70:] = np.arange(0, 70)
death_df['Cumulative Deaths'].iloc[70:] = cumulative_deaths.loc[:, 'National deceased cumulative'].values
death_df['Source'].iloc[70:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = death_df, x = 'day', y = 'Cumulative Deaths', hue = 'Source')
plt.savefig('Deaths_comparison')    
    
# infections plot
inf_df = pd.DataFrame(0, index = np.arange(0,70*2), columns = ['day', 'Cumulative Infections', 'Source'])
inf_df['day'].iloc[0:70] = np.arange(0, 70)
inf_df['Cumulative Infections'].iloc[0:70] = infections
inf_df['Source'].iloc[0:70] = 'Model estimates'
inf_df['day'].iloc[70:] = np.arange(0, 70)
inf_df['Cumulative Infections'].iloc[70:] = cumulative_infections.loc[:, 'National confirmed cumulative'].values
inf_df['Source'].iloc[70:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = inf_df, x = 'day', y = 'Cumulative Infections', hue = 'Source')
plt.savefig('Infections_comparison')  

# recovered plot
rec_df = pd.DataFrame(0, index = np.arange(0,70*2), columns = ['day', 'Cumulative Recoveries', 'Source'])
rec_df['day'].iloc[0:70] = np.arange(0, 70)
rec_df['Cumulative Recoveries'].iloc[0:70] = rec
rec_df['Source'].iloc[0:70] = 'Model estimates'
rec_df['day'].iloc[70:] = np.arange(0, 70)
rec_df['Cumulative Recoveries'].iloc[70:] = cumulative_rec.loc[:, 'National recovered cumulative'].values
rec_df['Source'].iloc[70:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = rec_df, x = 'day', y = 'Cumulative Recoveries', hue = 'Source')
plt.savefig('Recovered_comparison')  

# testing rate
test_df = pd.DataFrame(0, index = np.arange(0,70), columns = ['day', 'Diagnosis rate'])
test_df['day'] = np.arange(0, 70)
test_df['Diagnosis rate'] = testing
plt.figure()
sns.lineplot(data = test_df, x = 'day', y = 'Diagnosis rate')
plt.savefig('opti_out_diagnosis')  

#
test_df = pd.DataFrame(0, index = np.arange(0,50), columns = ['day', 'Diagnosis rate'])
test_df['day'] = np.arange(20, 70)
test_df['Diagnosis rate'] = testing[20:]
plt.figure()
sns.lineplot(data = test_df, x = 'day', y = 'Diagnosis rate')
plt.savefig('opti_out_diagnosis2')  
