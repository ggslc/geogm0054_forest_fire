#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 7 12:44:26 2023

@author: Killaspy_S
"""
from ca_fire import evolve_forest, write_netcdf


# defaults
GRID_N_X = 1000  # sub-divisions along X axis
# PROB_GROWTH = 1.0e-4  # probability of new growth per cell per unit time
PROB_GROWTH_LIST = [1e+00, 1e-01, 1e-02, 1e-03, 1e-04, 1e-05]
PROB_NEW_FIRE = 1.0e-6  # probability of new fire per cell per unit time
N_TIME_STEP = 2000  # number of time-steps
VERBOSE = True # set to False to suppress printing

# run the model
# result = evolve_forest(
#    (GRID_N_X, GRID_N_X), N_TIME_STEP, PROB_GROWTH, PROB_NEW_FIRE, verbose=VERBOSE)



for prob_growth in PROB_GROWTH_LIST:
    # result = evolve_forest(
        # (GRID_N_X, GRID_N_X), N_TIME_STEP, PROB_GROWTH_LIST, PROB_NEW_FIRE, verbose=VERBOSE)
    
    result = evolve_forest((GRID_N_X, GRID_N_X), N_TIME_STEP, prob_growth, PROB_NEW_FIRE, verbose=VERBOSE)
    
    #output
    file_name = f'forest_pg_{prob_growth:1.0e}_{N_TIME_STEP:06d}.nc'

    if VERBOSE:
        print(f'writing final output to {file_name}')
    
    write_netcdf(file_name , *result)




