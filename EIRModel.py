#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:29:16 2020

@author: vijetadeshpande
"""
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error


class EIRModel:
    def __init__(self, initial_pop, rate_matrix_file_path, divide_time_step = 3600):
        
        # set main attributes
        self.day_count = 0
        self.pop_init = initial_pop
        self.pop_day_start = initial_pop
        rate_matrix = pd.read_excel(rate_matrix_file_path, header = 0, index_col = 0).fillna(0)
        self.q = rate_matrix.values
        self.compartments = rate_matrix.index.values
        self.compartment_index = {}
        for com in self.compartments:
            self.compartment_index[com] = np.where(self.compartments == com)[0][0]
        
        # few hyper-par
        self.unit_time = np.divide(1, divide_time_step)
        self.reproduction_rate = {'no social distancing': 3.88/11.5, 
                                  'social distancing': 1.26/11.5,
                                  'quarantining': 0.32/11.5}
        
        # few important estimates which might be required for each time step
        self.epi_markers = {'diagnosed new': [],
                            'diagnosed cumulative': initial_pop[3],
                            'births': [],
                            'recoveries': [],
                            'deaths': [],
                            'population instantaneous': None,
                            'hospitalizations': [],
                            'testing rate': []
                }
        self.pop_day_end = 0
        self.delta_population = 0
        
        # what are we calibrating
        decision_mat = np.zeros((len(self.compartments), len(self.compartments)))
        asym, sym, bir = self.compartment_index['exposed'], self.compartment_index['symptomatic'], self.compartment_index['births']
        decision_mat[asym][bir], decision_mat[sym][bir] = 1, 1
        self.decision_var = decision_mat
        self.calibration_data = [1, 3, 4, 8, 10, 15, 22, 34, 50]
    
    def extract_val(self, com):
        delta_pop = self.delta_population
        idx = self.compartment_index[com]
        
        return delta_pop[idx]
    
    def update_epi_markers(self):
        
        # extract delta_pop
        delta_pop = self.delta_population
        
        # update diagnosed cases
        x = self.extract_val('diagnosed')
        self.epi_markers['dignosed new'].append(x)
        self.epi_markers['diagnosed cimulative'] += x
        
        # update births, i.e. new transmissions
        x = self.extract_val('births')
        self.epi_markers['births'].append(x)
        
        # update recoveries
        x = self.extract_val('recovery')
        self.epi_markers['recoveries'].append(x)
        
        # update deaths
        x = self.extract_val('death')
        self.epi_markers['deaths'].append(x)
        
        # hospitalizations
        x = self.extract_val('severe cases') + self.extract_val('non-severe cases')
        self.epi_markers['hospitalizations'].append(x)
        
        return
    
    def calculate_pop_change(self, pop_cur):
        delta_pop = np.multiply(np.multiply(np.reshape(pop_cur, (len(pop_cur),1)), self.q), self.unit_time)
        #print(('Following number should be close to zero: %f')%(sum(sum(delta_pop))*1000000000))
        
        # set attribute
        self.delta_population = delta_pop
        
        return delta_pop
    
    def forward(self, pop_prev, social_distancing = 'no'):
        
        for t in range(0, int(1/self.unit_time)):
            # store population
            #self.instantaneous_pop = pop_prev
            
            # calculate change in pop size
            delta_pop = self.calculate_pop_change(pop_prev)
             
            # update population
            pop_cur = pop_prev + np.sum(delta_pop, axis = 0)
            
            # cupdate epi markers
            #self.update_epi_markers()
            
            # update
            pop_prev = pop_cur
             
        return pop_cur
    
    def calculate_error(self, desc_var):
        # extract values
        day = self.day_count
        
        # first do forward pass
        pop_prev = self.pop_day_start
        pop_est = self.forward(pop_prev)
        
        # extract data to match to
        pop_act = self.calibration_data[day]
        
        # calculate squared error
        err = mean_squared_error([pop_act], [pop_est[self.compartment_index['diagnosed']]])
        
        return err
    
    def update_rate_par(self, rate_mat, s1, s2, val):
        
        # get index of states
        s1_idx, s2_idx = self.compartment_index[s1], self.compartment_index[s2]
        
        # update rate parameter
        rate_mat[s1_idx][s2_idx] = val
        
        # update diagonal element
        rate_mat[s1_idx][s1_idx] = -1 * np.sum(rate_mat[s1_idx][s2_idx+1:])
        
        return rate_mat
        
    def forward_fit(self, x_data, d_rate):
        
        # here x_data is day
        # y_datai is diagnosed cases
        # p0 is diagnosis rate (testing rate)
        
        # pop at day start (this will be used as initial population for this day)
        pop_start = self.pop_day_start
        
        # to have population at day end, we need a forward pass with par p0 in the rate matrix
        rate_mat = self.q
        
        # update value of diagnosis/testinf rate
        rate_mat = self.update_rate_par(rate_mat, 'symptomatic', 'diagnosed', d_rate)
        self.q = rate_mat
        
        # forward pass
        pop_end = self.forward(pop_start)
        self.pop_day_end = pop_end
        
        # predicted value of y_hat: predicted value of diagnoded cases
        y_hat = np.array([pop_end[self.compartment_index['diagnosed']]])
        
        return y_hat
        
    
    def backward(self, day):
        
        # what's decision variable?, initialize it
        desc_var_init = []
        #desc_var_init.append(self.q[self.compartment_index['exposed']][self.compartment_index['births']])
        desc_var_init.append(self.q[self.compartment_index['symptomatic']][self.compartment_index['diagnosed']])
        #desc_var_init.append(day)
        
        # scipy curve fit
        d_rate = opt.curve_fit(f = self.forward_fit, 
                               xdata = np.array(day),
                               ydata = np.array(self.calibration_data[day]),
                               p0 = desc_var_init,
                               method = 'lm')
        
        # store testing rate for the day
        self.epi_markers['testing rate'].append(d_rate[0])
        
        # print
        print(('Testing rate on day %f is: %f')%(day, d_rate[0][0]))
        
        # change rate matrix
        rate_mat = self.q
        rate_mat = self.update_rate_par(rate_mat, 'symptomatic', 'diagnosed', d_rate[0][0])
        self.q = rate_mat
        
        return
        
        

