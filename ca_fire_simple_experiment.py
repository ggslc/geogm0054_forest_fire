#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:35:51 2022

@author: steph
"""
from ca_fire import evolve_forest, write_netcdf


# defaults
GRID_N_X = 100  # sub-divisions along X axis
PROB_GROWTH = 1.0e-4  # probability of new growth per cell per unit time
PROB_NEW_FIRE = 1.0e-6  # probability of new fire per cell per unit time
N_TIME_STEP = 20000  # number of time-steps
VERBOSE = True # set to False to suppress printing

# run the model
result = evolve_forest(
    (GRID_N_X, GRID_N_X), N_TIME_STEP, PROB_GROWTH, PROB_NEW_FIRE, verbose=VERBOSE)

#output
file_name = f'forest_grid_{N_TIME_STEP:06d}.nc'

if VERBOSE:
    print(f'writing final output to {file_name}')

write_netcdf(file_name , *result)
