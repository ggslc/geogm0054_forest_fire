#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:35:51 2022

@author: steph
"""


import datetime
from ca_fire_mpi import evolve_forest, write_netcdf, write_netcdf_mpi


# defaults
GRID_N_X = 480 # sub-divisions along X axis
PROB_GROWTH = 1.0e-3 # probability of new growth per cell per unit time
PROB_NEW_FIRE = 1.0e-5 # probability of new fire per cell per unit time
TEST_CELL = False # if True, lighting strikes every time step in the test cell
TEST_CELL_COORDS = (8, 8)
N_TIME_STEP = 4000  # number of time-steps


start_time = datetime.datetime.now()

# run the model
forest_grid, area_trees, area_fire = evolve_forest(
    (GRID_N_X, GRID_N_X), N_TIME_STEP, PROB_GROWTH, PROB_NEW_FIRE, 
    test_cell=TEST_CELL, test_cell_coords=TEST_CELL_COORDS)


#output
write_netcdf(f'forest_grid_nx_{GRID_N_X}_{N_TIME_STEP:06d}.nc', 
             forest_grid, area_trees, area_fire)


# TODO switch to write_netcdf_mpi for the correct result in parallel 
# write_netcdf_mpi(f'forest_grid_nx_{GRID_N_X}_{N_TIME_STEP:06d}.nc', forest_grid, area_trees, area_fire )
end_time = datetime.datetime.now()
print(f'On rank {forest_grid._rank} : elapsed time = {(end_time - start_time).seconds} seconds')