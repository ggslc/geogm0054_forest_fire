#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Oct 5 2022

@author: stephen cornford
"""

import unittest
import numpy as np
import ca_fire


class TestCAFire(unittest.TestCase):
    """Unit tests for ca_fire functions."""

    def test_update_forest_spread(self):
        """
        Test fire-spreading in ca_fire.update_forest.
        """
          
        """
        Calls ca.fire_update_forest with the grid arg
        set to a single interior fire surrounded by
        trees, together with grow and ignite
        functions that return False in all cells. Then
        checks for the expected result (a ring of fire :)
        """      

        # TODO extend test to edges and corners

        # define 5 x 5 grid (smallest grid for ring of fire)
        n_i, n_j = 5, 5
        grid = np.full((n_i, n_j), ca_fire.EMPTY)
        
        # create a grid to hold the expected result
        expect = np.full((n_i, n_j), ca_fire.EMPTY)

        # input: trees across interior, fire in centre cell, edges empty
        grid[1:4, 1:4] = ca_fire.TREES
        grid[2, 2] = ca_fire.FIRE

        # expected result: centre empty, fire across interior, edges empty
        expect[1:4, 1:4] = ca_fire.FIRE
        expect[2, 2] = ca_fire.EMPTY

        #apply rules to grid
        grid = ca_fire.update_forest(grid, 0, 0)

        #check all elements of grid and expect have the same values
        self.assertTrue(np.all(grid == expect))

    def test_update_forest_no_spread(self):
         """
         Test fire spreading in ca_fire.update_forest.
         Calls ca.fire_update_forest with the grid arg
         set to a single interior fire surrounded by
         empty cells, together with grow and ignite
         functions that return False in all cells.
         Check for the expected result (no fires)
         """
        
         # create a 5 x 5 grid. This is the smallest
         # grid where a fire in the interior
         # can spread in all directions
         n_i, n_j = 5, 5
         grid = np.full((n_i, n_j), ca_fire.EMPTY)
        
         # create a grid to hold the expected result
         expect = np.full((n_i, n_j), ca_fire.EMPTY)

         # input: empty cells and a single fire
         grid[:, :] = ca_fire.EMPTY
         grid[2, 2] = ca_fire.FIRE
       
         # expected output : all cells empty
         expect[:, :] = ca_fire.EMPTY
        
         # update grid according to the CA rules
         grid = ca_fire.update_forest(grid, 0, 0 )
        
         #check the results : all cells should be equal
         self.assertTrue(np.all(grid[:, :] == expect[:, :]))   
         

if __name__ == '__main__':
    unittest.main()
# end
