#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 5 2022

@author: stephen cornford
"""

import unittest
import numpy as np
from ca_grid import CAGrid
import ca_fire


def _grid_false(grd):
    """
    Test growth/ignition function, False everywhere
    
    Parameters
    ----------
    grd : any type with a shape attribute

    Returns
    -------
    ndarray : grid of False bools

    """
    return np.full(grd.shape, False)


class TestCAFire(unittest.TestCase):
    """Unit tests for ca_fire functions."""

    def test_update_forest_spread(self):
        """
        Test fire-spreading in ca_fire.update_forest.

        Calls ca.fire_update_forest with the grid arg
        set to a single interior fire surrounded by
        trees, together with grow and ignite
        functions that return False in all cells. Then
        checks for the expected result (a ring of fire :)
        """
        # TODO extend test to edges and corners
        n_i, n_j = 5, 5
        grid = CAGrid((n_i, n_j))
        expect = CAGrid((n_i, n_j))

        # single interior fire, far from edge
        grid[grid.interior] = ca_fire.TREES
        grid[2, 2] = ca_fire.FIRE
        expect[grid.interior] = ca_fire.TREES
        expect[1:4, 1:4] = ca_fire.FIRE
        expect[2, 2] = ca_fire.EMPTY
        grid = ca_fire.update_forest(grid, _grid_false, _grid_false)
        self.assertTrue(np.all(grid[:, :] == expect[:, :]))



if __name__ == '__main__':
    unittest.main()
# end
