#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 2022

@author: stephen cornford (s.l.cornford@bristol.ac.uk)

"""

import numpy as np

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
    offset = ((0, 1), (0, -1), (1, 0), (-1, 0))
    if stencil_n == 9:
        offset += ((1, 1), (-1, 1), (1, -1), (-1, -1))
    neighbours = tuple((slice(1 + o[0], n_i + o[0] - 1),
                         slice(1 + o[1], n_j + o[1] - 1))
                        for o in offset)

    return cell, neighbours


# %%

class CAGrid:
    """
    Grid class for 2D CA (cellular automata) models. ig

    Provides access to grid data together with slices
    for interior cells and their nearest neigbours
    """
    def __init__(self, grid_shape):

        self._grid = np.zeros(grid_shape, dtype='int')
        self._interior, self._interior_neighbours = _stencil_slices(grid_shape, 9)

    @property
    def interior(self):
        """
        Return slice that indexes interior cells
        """
        return self._interior

    @property
    def interior_neighbours(self):
        """
        Return slice that indexes the neigbours of interior cells
        """
        return self._interior_neighbours

    @property
    def shape(self):
        """
        Return the grid shape tuple
        """
        return self._grid.shape

    def __getitem__(self, index):
        """
        Read access to the grid data.

        """
        return self._grid[index]

    def __setitem__(self, index, data):
        """
        Write access to the grid data

        """
        self._grid[index] = data


#end
