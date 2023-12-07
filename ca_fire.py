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
from xarray import Dataset, open_dataset

# %%
# constant definitions
EMPTY, TREES, FIRE = 0, 1, 2

# %%
def _stencil_slices(grid_shape, stencil_n=9):
    """
    Provide slice vectors (slice_0, slice_1) that represent
    the meembers of a stencil.

    The 5-point stencil comprises the interior cells and their
    west, east, south, north neigbours

    The 9-point stencil adds southe-west etc

    Parameters
    ----------
    grid_shape : (int, int) : grid dimensions
    stencil : int, optional
        choice of stencil (5 or 9 pt). The default is 9.

    Returns
    -------
    cell: slice indexing all cells in the interior
    neighbours: tuple of slices indexing cells adjacent to cell
    """

    # obtain grid parameters
    n_i, n_j = grid_shape

    # slice tuples for interior cells
    cell = slice(1, n_i-1), slice(1, n_j-1)

    # slice tuples for neighbours
    # offset = ((0, 1), (0, 1), (1, 0), (-1, 0))
    
    offset = ((0, 1), (0, -1), (1, 0), (-1, 0))
    if stencil_n == 9:
        offset += ((1, 1), (-1, 1), (1, -1), (-1, -1))
    neighbours = tuple((slice(1 + o[0], n_i + o[0] - 1),
                         slice(1 + o[1], n_j + o[1] - 1))
                        for o in offset)

    return cell, neighbours


def grid_bernoulli_trial(grid, probability):
    """

    Returns the outcome of Bernoulli trials on a grid.

    Parameters
    ----------
    probability : float
        probability of sucess in each trial

    Returns
    -------
     ndarray(type = bool) of outcomes
        with the same shape as grid

    """
    rng = np.random.default_rng()
    return rng.uniform(size=grid.shape) < probability

# %%

# %%
def update_forest(grid, prob_grow, prob_ignite):
    """

    Update a forest grid according to the CA rules

    Interior cells:
        1. fire cells -> empty cells
        2. trees cells adjacent to fire cells -> fire cells
        3. empty cell -> tree cell with probability prob_grow
        4. tree cell -> fire cell with probability prob_ignite
    Boundary cells:
        1. no changes

    Parameters
    ----------
    grid : ndarray, dtype='int' .
        current state of the forest.

    prob_grow : float, probability of new growth per cell per unit time

    prob_ignite : float, probability of new fire per cell per unit time

    Returns
    -------
    grid: ndarray. the updated state.

    """

    cell, neighbours = _stencil_slices(grid.shape)

    # allocate new grid and remove burning trees
    next_state = np.where(grid[cell] == FIRE, EMPTY, grid[cell])

    # spread fire
    for neighbour in neighbours:
        next_state = np.where(
            (grid[cell] == TREES) & (grid[neighbour] == FIRE),
            FIRE,  next_state)

    # new growth
    next_state = np.where(
        (grid[cell] == EMPTY) & grid_bernoulli_trial(grid, prob_grow)[cell],
        TREES, next_state)

    # ignition
    next_state = np.where(
        (grid[cell] == TREES) & grid_bernoulli_trial(grid, prob_ignite)[cell],
        FIRE, next_state)

    grid[cell] = next_state

    return grid

# %%





def evolve_forest(grid_shape, n_time_step,  prob_grow,
                  prob_ignite, verbose=True):
    """

    Create and evolve a forest fire grid, then save the final state
    together with the tree- and fire-covered areas at every step.

    Parameters
    ----------
    grid_shape : (int, int), grid dimensions in the west-east direction

    n_time_step : int, number of time steps

    prob_grow : float, probability of new growth per cell per unit time

    prob_ignite : float, probability of new fire per cell per unit time

    verbose: bool, log progress to print()? default True.

    Returns
    -------

    forest_grid: ndarray, dtype='int', ndim =2 final state of the forest

    area_trees: ndarray, dtype='int', ndim = 1, count of tree cells
                ot each time step

    area_fire: ndarray, dtype='int', ndim = 1, count of fire cells at
                each time step

    """

    # allocate storage
    forest_grid = np.zeros(grid_shape, dtype='int')
    area_fire = np.zeros(n_time_step, dtype='int')
    area_trees = np.zeros(n_time_step, dtype='int')


    if verbose:
        print('---')
        print(f'evolving forest for {n_time_step} time steps')
        print(f'growth probability = {prob_grow:4.0e}')
        print(f'ignition probability = {prob_ignite:4.0e}')
        print('---')

    # time loop
    for step in range(0, n_time_step):
        # apply CA rules to forest_grid
        forest_grid = update_forest(forest_grid, prob_grow, prob_ignite)

        # calculate & store fire and tree areas
        area_fire[step] = np.sum(np.where(forest_grid[:,:] == FIRE, 1, 0))
        area_trees[step] = np.sum(np.where(forest_grid[:,:] == TREES, 1, 0))

        #log state every 1000 steps
        if verbose and (step + 1) % 1000 == 0:
            print(f'time step {step} : tree area = {area_trees[step]}')

    if verbose:
        print('---')

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
        grid = dset.grid.to_numpy()
        area_trees = dset.area_trees.to_numpy()
        area_fire = dset.area_fire.to_numpy()

    return grid, area_trees, area_fire
