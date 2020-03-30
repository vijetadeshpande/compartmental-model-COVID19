#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:29:16 2020

@author: vijetadeshpande
"""
import pandas as pd
import numpy as np


class EIRModel:
    def __init__(self, initial_pop, rate_matrix_file_path):
        rate_matrix = pd.read_excel(rate_matrix_file_path, header = 0, index_col = 0).fillna(0)
        self.q = rate_matrix.values
        self.compartments = rate_matrix.index.values
        if len(initial_pop) == len(self.compartments):
            self.init_pop = initial_pop
        else:
            print('Length of initial population should be equal to number of compartments')
        self.unit_time = np.divide(1, 1)
        self.entering_population = None  
        self.incidence = None
        self.new_hospitalization = None
    
    def calculate_pop_change(self, pop_cur):
        delta_pop = np.multiply(np.multiply(np.reshape(pop_cur, (len(pop_cur),1)), self.q), self.unit_time)
        
        print(('Following number should be close to zero: %f')%(sum(sum(delta_pop))*1000000000))
        
        self.entering_pop = delta_pop
        
        return delta_pop
    
    def calculate_new_infections(self):
        # take new cases from exposed, symptomatic and diagnosed
        self.incidence = np.multiply(self.entering_population, np.array([1,1,1,0,0,0,0]))
        
        return
    
    def calculate_new_hospitalization(self):
        # take new hospitalizations from severe and non-severe compartments
        self.new_hospitalization = np.multiply(self.entering_population, np.array([0,0,0,1,1,0,0]))
        
        return
    
    def forward(self, pop_prev):
        
         # calculate change in pop size
         delta_pop = self.calculate_pop_change(pop_prev)
         
         # update population
         pop_cur = pop_prev + np.sum(delta_pop, axis = 0)
         
         # compute new indections and new hospitalizations
         #self.calculate_new_infections()
         #self.calculate_new_hospitalization()
         
         return pop_cur
     

    

