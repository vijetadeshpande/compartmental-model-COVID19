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
        
        # simple check
        if len(initial_pop) == len(self.compartments):
            self.init_pop = initial_pop
        else:
            print('Length of initial population should be equal to number of compartments')
        
        # set main attributes
        rate_matrix = pd.read_excel(rate_matrix_file_path, header = 0, index_col = 0).fillna(0)
        self.q = rate_matrix.values
        self.compartments = rate_matrix.index.values
        self.compartment_index = {}
        for com in self.compartments:
            self.compartment_index[com] = np.where(self.compartments == com)
        
        # few hyper-par
        self.unit_time = np.divide(1, 1)
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
                            'hospitalizations': []
                }
        self.insta_population = 0
        self.delta_population = 0
        
        # what are we calibrating
        decision_mat = np.zeros((len(self.compartments), len(self.compartments)))
        decision_mat[1:3][0] = 1
        self.decision_var = decision_mat
    
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
        
        # store population
        self.insta_population = pop_prev
        
        # calculate change in pop size
        delta_pop = self.calculate_pop_change(pop_prev)
         
        # update population
        pop_cur = pop_prev + np.sum(delta_pop, axis = 0)
        
        # cupdate epi markers
        self.update_epi_markers()
         
        return pop_cur
    
    def backward(self):
        
        # extract the model output to which we are fitting the model
        # number of diagnosed cases
        
        return
        
        

