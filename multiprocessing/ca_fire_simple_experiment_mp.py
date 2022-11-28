#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:35:51 2022

@author: steph

"""
from multiprocessing import Pool
from itertools import product
import sys
#import modules from the parent dir
sys.path.append('../') 

from ca_fire import evolve_forest, write_netcdf

# defaults
GRID_N_X = 100 # sub-divisions along X axis
PROB_GROWTH_LIST = [1.0e-2, 1.0e-3, 1.0e-4] # probability of new growth per cell per unit time
PROB_IGNITE_LIST = [1.0e-4, 1.0e-5]  # probability of new fire per cell per unit time
N_TIME_STEP = 4000  # number of time-steps
PARALLEL = False

def run_simulation(args):
    
    # run the model
    prob_growth, prob_ignite = args
    
    print (f'----\nrunning with prob_growth = {prob_growth:1.0e}, prob_ignite = {prob_ignite:1.0e}')
    sys.stdout.flush()
    result = evolve_forest(
        (GRID_N_X, GRID_N_X), N_TIME_STEP, prob_growth, prob_ignite, verbose=False)

    #output
    file_name = f'forest_grid_pg_{prob_growth:1.0e}_pi_{prob_ignite:1.0e}_{N_TIME_STEP:06d}.nc'
    print (f'writing to {file_name} \n----')
    sys.stdout.flush()
    write_netcdf(file_name , *result)


if __name__ == '__main__':
   
    probs_list = list(product(PROB_GROWTH_LIST, PROB_IGNITE_LIST))
    print (f'running {len(probs_list)} simulations')
    
    if PARALLEL:    
        #parallel version
        with Pool(4) as pool:
            pool.map(run_simulation, probs_list)

    else:
        #serial version
        for probs in probs_list:
            run_simulation(probs)

    print (f'completed {len(probs_list)} simulations')
