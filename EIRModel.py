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
from copy import deepcopy


class EIRModel:
    def __init__(self, rate_matrix_file_path, initial_pop = [], divide_time_step = 3600):
        
        # useless attrubutes
        self.day_count = 0
        
        # read rate matrix and set related attributes
        rate_matrix = pd.read_excel(rate_matrix_file_path, header = 0, index_col = 0).fillna(0)
        self.q = rate_matrix.values
        self.compartments = rate_matrix.index.values
        self.compartment_index = {}
        for com in self.compartments:
            self.compartment_index[com] = np.where(self.compartments == com)[0][0]
        
        # set initial population
        initial_pop = np.zeros(len(self.compartments))
        initial_pop[self.compartment_index['I_shed']] = 48
        initial_pop[self.compartment_index['I_s_mild']] = 4
        initial_pop[self.compartment_index['I_s_moderate']] = 4
        initial_pop[self.compartment_index['I_s_severe']] = 4
        #initial_pop[self.compartment_index['I_d_moderate']:] = 0
        self.pop_init = initial_pop
        self.pop_day_start = initial_pop
        
        # attributes for floating population
        self.pop_day_end = 0
        self.delta_population = 0
        self.diagnosed = 0
        self.recoveries = 0
        self.deaths = 0
        
        # unit time step for simulation 
        self.unit_time = np.divide(1, divide_time_step)
        
        # some data for basic reproduction number
        self.reproduction_rate = {'no social distancing': 3.88/11.5, 
                                  'social distancing': 1.26/11.5,
                                  'quarantining': 0.32/11.5}
        
        # some more assumptions or data
        self.test_pos = 0.01
        self.test_c_trace = 10 #100
        
        # few important estimates which might be required for each time step
        self.epi_markers = {'diagnosed new': [],
                            'diagnosed cumulative': initial_pop[3],
                            'births': [],
                            'recoveries': [],
                            'deaths': [],
                            'population': [],
                            'hospitalizations': [],
                            'testing rate': []
                }
        
        # data for calibration (y_actual)
        self.calibration_data = None
        
        
    def pop_shift(self, pop_end, add_pos):
        
        # Assumption
        # all positive cases from contact tracing are asymptomatic and are considered mild cases after diagnosis
        
        # get index of compartements of interest
        shed, d_mild = self.compartment_index['I_shed'], self.compartment_index['I_d_mild']
        
        # update population at day end
        pop_end[shed] -= add_pos
        pop_end[d_mild] += add_pos
        
        return pop_end
    
    def contact_tracing(self, diagnosed, pop):
        
        # assumptions for contact tracing:
        
        # 1. on top of these diagnosed cases from all d_rates, we trace 100 contacts for each diagnosed case.
        #   e.g. 2 cases are diagnosed = 200 contacts are traced, 
        #       200 traced will get a test and 2% will be positive (this prevalence of 2% is estimate from ICMR data)
        
        # TODO: following line does not correctly calculate new diagnosed cases (little less than actual number)
        # how many cases are diagnosed in severe and moderate state
        #pop_diff = pop_end - pop_start
        
        # index of compartments of interest
        #mild, mod, sev = self.compartment_index['I_d_mild'], self.compartment_index['I_d_moderate'], self.compartment_index['I_d_severe']
        
        # total new cases
        new_dia = diagnosed
        
        # positive cases from traced contacts
        additional_pos = np.multiply(new_dia, np.multiply(self.test_c_trace, self.test_pos))
        
        # shift newly diagnosed from I_shed to I_d
        pop = self.pop_shift(pop, additional_pos)
        
        
        return pop
    
    def extract_val(self, com):
        delta_pop = self.delta_population
        idx = self.compartment_index[com]
        
        return delta_pop[idx]
    
    def update_epi_markers(self, incident_pop):
        
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
        h_moderate, h_severe, rec = self.compartment_index['H_moderate'], self.compartment_index['H_severe'], self.compartment_index['R']
        rec_t = incident_pop[h_moderate][rec] + incident_pop[h_severe][rec]
        self.epi_markers['recoveries'].append(rec_t)
        
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
        
        # set newly diagnosed cases
        s_mild, s_mod, s_sev = self.compartment_index['I_s_mild'], self.compartment_index['I_s_moderate'], self.compartment_index['I_s_severe']
        d_mild, d_mod, d_sev = self.compartment_index['I_d_mild'], self.compartment_index['I_d_moderate'], self.compartment_index['I_d_severe']
        self.diagnosed += delta_pop[s_mild][d_mild] + delta_pop[s_mod][d_mod] + delta_pop[s_sev][d_sev]
        
        # set new recoveries
        h_moderate, h_severe, rec = self.compartment_index['H_moderate'], self.compartment_index['H_severe'], self.compartment_index['R']
        self.recoveries += delta_pop[h_moderate][rec] + delta_pop[h_severe][rec]
        
        # set new deaths
        h_severe, death = self.compartment_index['H_severe'], self.compartment_index['D']
        self.deaths += delta_pop[h_severe][death]
        
        
        return delta_pop
    
    def forward(self, pop_prev, social_distancing = 'no'):
        
        # initialize
        self.diagnosed = 0
        self.recoveries = 0
        self.deaths = 0
        
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
        
        # contact tracing
        pop_cur = self.contact_tracing(self.diagnosed, pop_cur)
             
        return pop_cur
    
    def update_rate_par(self, rate_mat, s1, s2, val):
        
        # get index of states
        s1_idx, s2_idx = self.compartment_index[s1], self.compartment_index[s2]
        
        # update rate parameter
        rate_mat[s1_idx][s2_idx] = val
        
        # update diagonal element
        rate_mat[s1_idx][s1_idx] = -1 * np.sum(rate_mat[s1_idx][s2_idx:])
        
        return rate_mat
    
    def diagnosis_rate_assumption(self, rate_mat, d_rate):
        
        rate_mat = deepcopy(rate_mat)
        
        # our diagnosis assumption is as follows
        # 1. decision variable foe the optimization model is only the diagnosis rate in severe disease state
        #   THIS IS 'd_rate' INPUT 
        # 2. diagnosis rate of moderate disease state is same as severe disease state
        # 3. diagnosis rate of mild disease state is 4 time d_rate of severe 
        #   (for now I have assumed d_rate value for mild to be 0, because we only have data hospitalized cases and mild cases are not hospitalized)

        # Note: tests are assumed to be perfect and immediate
        
        
        # change d_rate of severe
        rate_mat = self.update_rate_par(rate_mat, 'I_s_severe', 'I_d_severe', d_rate)
        
        # assumption #2
        rate_mat = self.update_rate_par(rate_mat, 'I_s_moderate', 'I_d_moderate', d_rate)
        
        # assumption #3
        rate_mat = self.update_rate_par(rate_mat, 'I_s_mild', 'I_d_mild', 0)
        
        
        return rate_mat
    
    
    
    def calculate_y_hat(self, diag_start, diag_end, diag_c_tracing):
        
        # index of compartments of interest
        #mild, mod, sev = self.compartment_index['I_d_mild'], self.compartment_index['I_d_moderate'], self.compartment_index['I_d_severe']
        
        #
        y_hat = (diag_end - diag_start) + diag_c_tracing#sum((pop_end - pop_start)[[mild, mod, sev]])
        
        return y_hat
        
    def forward_fit(self, x_data, d_rate):
        
        # here x_data is day
        # y_datai is diagnosed cases
        # p0 is diagnosis rate (testing rate)
        
        # pop at day start (this will be used as initial population for this day)
        pop_start = self.pop_day_start
        diag_start = self.diagnosed
        
        # to have population at day end, we need a forward pass with par p0 in the rate matrix
        rate_mat = self.q
        
        # update value of diagnosis/testinf rate
        rate_mat = self.diagnosis_rate_assumption(rate_mat, d_rate)
        self.q = rate_mat
        
        # forward pass
        pop_end = self.forward(pop_start)
        diag_end = self.diagnosed
        
        # predicted value of y_hat: predicted value of diagnoded cases
        #y_hat = np.array([pop_end[self.compartment_index['diagnosed']]])
        #y_hat = np.array([self.calculate_y_hat(diag_start, diag_end, add_pos)])
        y_hat = np.array([sum(pop_end[6:-2])])
        
        
        return y_hat
        
    
    def backward(self, day):
        
        # what's decision variable?, initialize it
        desc_var_init = []
        #desc_var_init.append(self.q[self.compartment_index['exposed']][self.compartment_index['births']])
        desc_var_init.append(self.q[self.compartment_index['I_s_severe']][self.compartment_index['I_d_severe']])
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
        print(('Testing rate on day %i is: %f')%(day, d_rate[0][0]))
        
        # change rate matrix
        rate_mat = self.q
        rate_mat = self.diagnosis_rate_assumption(rate_mat, d_rate[0][0])
        self.q = rate_mat
        
        return
        
        

