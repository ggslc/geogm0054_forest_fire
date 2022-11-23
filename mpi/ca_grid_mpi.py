#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 2022

@author: stephen cornford (s.l.cornford@bristol.ac.uk)

"""

import numpy as np
from mpi4py import MPI

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

def partition(grid_shape, part, n_part, halo=1):

    n_i = int(grid_shape[0] / n_part)
    if n_i * n_part != grid_shape[0]:
        raise ValueError('grid must be divisible by number of ranks')

    ilo = part * n_i - halo if part > 0 else 0
    ihi = ilo + n_i +  2 * halo # else grid_shape[0]

    part_grid_shape = (ihi-ilo, grid_shape[1] + 2*halo)

    global_slice = (slice(ilo, ihi), slice(0, grid_shape[0]))

    return part_grid_shape, global_slice

class CAGrid:
    """
    Grid class for 2D CA (cellular automata) models. ig

    Provides access to grid data together with slices
    for interior cells and their nearest neigbours
    """
    def __init__(self, grid_shape, halo=1):

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self._halo = halo
        self._global_shape = (grid_shape[0] + 2 * halo, grid_shape[1] + 2*halo)
        local_grid_shape, self._global_slice = partition(
            grid_shape, self._rank, self._size, halo=self._halo)

        print(f'On rank {self._rank}: local_grid_shape == {local_grid_shape}, halo = {self._halo}, global_slice = {self._global_slice}')

        self._grid = np.zeros(local_grid_shape, dtype='int')
        self._interior, self._interior_neighbours = _stencil_slices(local_grid_shape, 9)


    def mpi_sum(self, local):
        reduced = np.empty(1, dtype=type(local))
        self._comm.Allreduce(local, reduced, op=MPI.SUM)
        return reduced[0]

    def fetch_global_data(self):

        #point to point version. use of blocking comms makes this (essentially) serial
        #also assumes that all grids are the same shape - fix this!
        grid_global = None
        if self._rank == 0:
            grid_global = np.zeros(self._global_shape)
            #print('fetch_global_data:', grid_global.shape)
            grid_global[:self._grid.shape[0], :] = self._grid[:,:]

        for send_rank in range(1, self._comm.Get_size()):

            if self._rank == send_rank:
                sendbuf = self._grid[self._halo:-self._halo, :].copy()
                self._comm.Send(sendbuf, dest=0, tag=123)
            elif self._rank == 0:
                recvbuf = self._grid[self._halo:-self._halo, :].copy()
                self._comm.Recv(recvbuf, source=send_rank, tag=123)
                ilo = (self._grid.shape[0] - 2 * self._halo)*send_rank + self._halo
                ihi = ilo + self._grid.shape[0] - 2 * self._halo
                #print(self._rank, send_rank, ilo, ihi, ihi-ilo, recvbuf.shape)
                grid_global[ilo:ihi, :] = recvbuf


        return grid_global


    def exchange(self):

        req_a = None
        req_b = None
        halo = self._halo
        low_send_high_recv = 123
        high_send_low_recv = 321

        #send low interior
        if self._rank > 0:
            sendbuf = self._grid[halo:2 * halo, :]
            self._comm.Isend(sendbuf, dest=self._rank -1, tag=low_send_high_recv)
        #send high interior
        if self._rank < self._size - 1:
            sendbuf = self._grid[self.shape[0] - 2 * halo :self.shape[0] - halo:, :]
            self._comm.Isend(sendbuf, dest=self._rank + 1, tag=high_send_low_recv)
        #receice high halo
        if self._rank < self._size -1:
            recvbuf = self._grid[self.shape[0] - halo:self.shape[0], :]
            req_a = self._comm.Irecv(recvbuf, source=self._rank + 1, tag=low_send_high_recv)
        #receive low halo
        if self._rank > 0:
            recvbuf = self._grid[0:halo,]
            req_b = self._comm.Irecv(recvbuf, source=self._rank -1, tag=high_send_low_recv)

        if req_a:
            req_a.wait()
        if req_b:
            req_b.wait()

        #self._comm.Barrier()


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
    def global_slice(self):
        return self._global_slice

    @property
    def shape(self):
        """
        Return the grid shape tuple
        """
        return self._grid.shape

    @property
    def global_shape(self):
        """
        Return the global grid shape tuple
        """
        return self._global_shape


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
