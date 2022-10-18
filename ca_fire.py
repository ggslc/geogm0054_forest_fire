#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 4 2022

@author: stephen cornford

CA fire model based on Charbonneau 2017
'Natural Complexity, a modeling handbook', chap 6

Part of the University of Bristol unit
GEOGM0054 Introduction to Scientific Computing

@author: stephen cornford (s.l.cornford@bristol.ac.uk)

"""

import numpy as np
from ca_grid import CAGrid
from xarray import Dataset, open_dataset
# %%
# constant definitions
EMPTY, TREES, FIRE = 0, 1, 2


# %%
def update_forest(grid, grow, ignite):
    """

    Update a forest grid according to the CA rules

    Interior cells:
        1. fire cells -> empty cells
        2. trees cells adjacent to fire cells -> fire cells
        3. empty cell [i,j] -> tree cell if grow(forest_grid)[i,j]
        4. tree cell [i,j] -> fire cell if ignite(forest_grid)[i,j]
    Boundary cells:
        1. no changes

    Parameters
    ----------
    grid : CAGrid .
        current state of the forest

    grow : function(grid[:,:]).
        Determines cells with new trees. Should return ndarry(dtype='bool')
        with the same shape as grid[:,:] and dtype='bool'

    ignite: function(grid[:,:]).
        Determines cells with new fire. Should return a grid
        with the same shape as grid[:,:], and dtype='bool'

    Returns
    -------
    grid: the updated CAGrid

    """

    cell, neighbours = grid.interior, grid.interior_neighbours

    # allocate new grid and remove burning trees
    next_state = np.where(grid[cell] == FIRE, EMPTY, grid[cell])

    # spread fire
    for neighbour in neighbours:
        next_state = np.where(
            (grid[cell] == TREES) & (grid[neighbour] == FIRE),
            FIRE,  next_state)

    # new growth
    next_state = np.where(
        (grid[cell] == EMPTY) & grow(grid)[cell],
        TREES, next_state)

    # ignition
    next_state = np.where(
        (grid[cell] == TREES) & ignite(grid)[cell],
        FIRE, next_state)

    grid[cell] = next_state

    return grid

# %%


def grid_bernoulli_trial(probability):
    """

    Returns a *function* which returns the outcome of Bernoulli trials on a grid.

    Parameters
    ----------
    probability : float
        probability of sucess in each trial

    Returns
    -------
    bernoulli_trial_function: function(grid). 
        Returns an ndarray(type = bool) of outcomes
        with the same shape as grid

    """
    rng = np.random.default_rng()

    def bernoulli_trial_function(grid):
        return rng.uniform(size=grid.shape) < probability

    return bernoulli_trial_function
# %%


def evolve_forest(grid_shape, n_time_step,  prob_growth,
                  prob_new_fire, initial=None):
    """

    Create and evolve a forest fire grid, then save the final state
    together with the tree- and fire-covered areas at every step.

    Parameters
    ----------
    grid_shape : (int, int), grid dimensions in the west-east direction

    n_time_step : int, number of time steps

    prob_growth : float, probability of new growth per cell per unit time

    prob_new_fire : float, probability of new fire per cell per unit time

    Returns
    -------
    None.

    """

    # allocate storage
    forest_grid = CAGrid(grid_shape)
    
    if not (initial is None):
        forest_grid[:,:] = initial[:,:]    
        
    
    area_fire = np.zeros(n_time_step, dtype='int')
    area_trees = np.zeros(n_time_step, dtype='int')

    # define the growth and iginition functions
    random_grow = grid_bernoulli_trial(prob_growth)
    random_ignite = grid_bernoulli_trial(prob_new_fire)

    # time loop
    for step in range(0, n_time_step):
        # apply CA rules to forest_grid
        forest_grid = update_forest(forest_grid, random_grow, random_ignite)
        # calculate & store fire and tree areas
        area_fire[step] = np.sum(np.where(forest_grid[:,:] == FIRE, 1, 0))
        area_trees[step] = np.sum(np.where(forest_grid[:,:] == TREES, 1, 0))

    # output final state
    return forest_grid, area_trees, area_fire



def write_netcdf(file_name, grid,  area_trees, area_fire):

    Dataset({"grid": (("y","x"), grid[:,:]),
             "area_fire": (("time"), area_fire[:]),
             "area_trees": (("time"), area_trees[:])}
            ).to_netcdf(file_name)
    
def read_netcdf(file_name):
    
    with open_dataset(file_name) as dset:
        dset.load()
        grid = CAGrid(dset.grid.shape)
        grid[:,:] = dset.grid[:,:]
        area_trees = dset.area_trees.to_numpy()
        area_fire = dset.area_fire.to_numpy()
        
    return grid, area_trees, area_fire
    