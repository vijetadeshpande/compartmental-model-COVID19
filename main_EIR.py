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


# test positivity
def test_pos_fraction(x):
    
    if x <= 0:
        test_p = 0.001
    else:
        test_p = 0.0117*np.log(x) + 0.0025
    
    return test_p

if True:
    # import data for calibration
    path_data = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/Data/India/calibration_data_for_Punjab_state.csv'
    data_cali = pd.read_csv(path_data)
    
    # extract data for calibration and visualization 
    new_infections = deepcopy(data_cali.loc[:, ['day', 'National confirmed new']])
    cumulative_infections = deepcopy(data_cali.loc[:, ['day', 'National confirmed cumulative']]).groupby(by = 'day').max()
    cumulative_deaths = deepcopy(data_cali.loc[:, ['day', 'National deceased cumulative']]).groupby(by = 'day').max()
    cumulative_rec = deepcopy(data_cali.loc[:, ['day', 'National recovered cumulative']]).groupby(by = 'day').max()
    active_cases = cumulative_infections.values - cumulative_rec.values
    
    # rate matrix file path
    file_path = r'/Users/vijetadeshpande/Documents/GitHub/compartmental-model-COVID19/rate_matrix_expanded3.xlsx'
    
    # initialize model
    covid = EIRModel.EIRModel(rate_matrix_file_path = file_path)
    covid.calibration_data = active_cases
    #new_infections.groupby(by = 'day').sum().loc[:, 'National confirmed new'].values
    #cumulative_infections.groupby(by = 'day').sum().loc[:, 'National confirmed cumulative'].values
    
    # loop over some horizon
    pop_prev = covid.pop_day_start
    social_dist = 'no'
    max_day = data_cali.day.max() - data_cali.day.min()
    for day in range(0, max_day+1):
        
        # set test positive fraction
        covid.test_pos = test_pos_fraction(day)
        
        if day == 53:
            rate_mat = covid.q
            new_r0 = 1.2/10
            rate_mat[[2,3,4,5], 0] = new_r0
            covid.q = rate_mat
        
        # update population at start of day and day 
        covid.day_count = day
        covid.pop_day_start = pop_prev
        
        # fit diagnosis rate for this day
        covid.backward(day)
        
        # forward pass to get population for next day
        pop_cur = covid.forward(pop_prev, social_dist)
        covid.epi_markers['recoveries'].append(covid.recoveries)
        covid.epi_markers['diagnosed new'].append(covid.diagnosed + np.multiply(covid.diagnosed, np.multiply(covid.test_c_trace, covid.test_pos)))
        covid.epi_markers['deaths'].append(covid.deaths)
        covid.pop_day_end = pop_cur
        
        # store population
        covid.epi_markers['population'].append(pop_cur)
        
        # update previous pop for next day
        pop_prev = pop_cur
    
    # collect population at each time step
    pop_t = covid.epi_markers['population']
    
# visualization
model_deaths = []
model_deaths = np.cumsum(covid.epi_markers['deaths'])
death_idx = covid.compartment_index['D']
inf_idx = [covid.compartment_index['I_d_moderate'], covid.compartment_index['I_d_severe'], covid.compartment_index['H_moderate'], covid.compartment_index['H_severe']]
model_rec = np.cumsum(covid.epi_markers['recoveries'])
model_dia = covid.epi_markers['diagnosed new']
model_active = []
model_cumulative = np.cumsum(model_dia)
testing = []
test_rate = covid.epi_markers['testing rate']
index = 0
for pop in pop_t:
    model_active.append(sum(pop[6:-2]))
    testing.append(test_rate[index][0])
    index += 1 
        

# death plot
death_df = pd.DataFrame(0, index = np.arange(0,(max_day+1)*2), columns = ['day', 'Cumulative Deaths', 'Source'])
death_df['day'].iloc[0:max_day+1] = np.arange(0, max_day+1)
death_df['Cumulative Deaths'].iloc[0:max_day+1] = model_deaths
death_df['Source'].iloc[0:max_day+1] = 'Model estimates'
death_df['day'].iloc[max_day+1:] = np.arange(0, max_day+1)
death_df['Cumulative Deaths'].iloc[max_day+1:] = cumulative_deaths.loc[:, 'National deceased cumulative'].values
death_df['Source'].iloc[max_day+1:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = death_df, x = 'day', y = 'Cumulative Deaths', hue = 'Source')
plt.savefig('Deaths_comparison')    
    
# new infections plot
inf_df = pd.DataFrame(0, index = np.arange(0,(max_day+1)*2), columns = ['day', 'New Infections', 'Source'])
inf_df['day'].iloc[0:max_day+1] = np.arange(0, max_day+1)
inf_df['New Infections'].iloc[0:max_day+1] = model_dia
inf_df['Source'].iloc[0:max_day+1] = 'Model estimates'
inf_df['day'].iloc[max_day+1:] = np.arange(0, max_day+1)
inf_df['New Infections'].iloc[max_day+1:] = new_infections.groupby(by = 'day').sum().loc[:, 'National confirmed new'].values #cumulative_infections.loc[:, 'National confirmed cumulative'].values
inf_df['Source'].iloc[max_day+1:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = inf_df, x = 'day', y = 'New Infections', hue = 'Source')
plt.savefig('New_infections_comparison')

# active cases
inf_df = pd.DataFrame(0, index = np.arange(0,(max_day+1)*2), columns = ['day', 'Active cases', 'Source'])
inf_df['day'].iloc[0:max_day+1] = np.arange(0, max_day+1)
inf_df['Active cases'].iloc[0:max_day+1] = model_active
inf_df['Source'].iloc[0:max_day+1] = 'Model estimates'
inf_df['day'].iloc[max_day+1:] = np.arange(0, max_day+1)
inf_df['Active cases'].iloc[max_day+1:] = np.reshape(active_cases, len(active_cases))
inf_df['Source'].iloc[max_day+1:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = inf_df, x = 'day', y = 'Active cases', hue = 'Source')
plt.savefig('Active_cases_comparison')

# recovered plot
rec_df = pd.DataFrame(0, index = np.arange(0,(max_day+1)*2), columns = ['day', 'Cumulative Recoveries', 'Source'])
rec_df['day'].iloc[0:max_day+1] = np.arange(0, max_day+1)
rec_df['Cumulative Recoveries'].iloc[0:max_day+1] = model_rec
rec_df['Source'].iloc[0:max_day+1] = 'Model estimates'
rec_df['day'].iloc[max_day+1:] = np.arange(0, max_day+1)
rec_df['Cumulative Recoveries'].iloc[max_day+1:] = cumulative_rec.loc[:, 'National recovered cumulative'].values
rec_df['Source'].iloc[max_day+1:] = 'ICMR published data'
plt.figure()
sns.lineplot(data = rec_df, x = 'day', y = 'Cumulative Recoveries', hue = 'Source')
plt.savefig('Recovered_comparison')  

# testing rate
test_df = pd.DataFrame(0, index = np.arange(0,max_day+1), columns = ['day', 'Diagnosis rate'])
test_df['day'] = np.arange(0, max_day+1)
test_df['Diagnosis rate'] = testing
plt.figure()
sns.lineplot(data = test_df, x = 'day', y = 'Diagnosis rate')
plt.savefig('opti_out_diagnosis')  

#
test_df = pd.DataFrame(0, index = np.arange(0,max_day+1-20), columns = ['day', 'Diagnosis rate'])
test_df['day'] = np.arange(20, max_day+1)
test_df['Diagnosis rate'] = testing[20:]
plt.figure()
sns.lineplot(data = test_df, x = 'day', y = 'Diagnosis rate')
plt.savefig('opti_out_diagnosis2')  
