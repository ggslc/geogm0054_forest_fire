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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ca_fire import EMPTY, FIRE, read_netcdf

EMPTY_COLOR = '#FFFFFF'
TREE_COLOR = '#0099FF'
FIRE_COLOR = '#FF6600'

def show_forest(grid, axg):
    """
    Add a forest 'aerial' image to axg.
    """
    cmap_forest = ListedColormap([EMPTY_COLOR, TREE_COLOR, FIRE_COLOR])
    axg.pcolormesh(grid[:,:], vmin=EMPTY, vmax=FIRE, cmap=cmap_forest)
    axg.set_xlabel('x')
    axg.set_ylabel('y')



def plot_time_series(data, axp, color, ylabel, label_side):
    """
    Add time series data to axp, color labels to match.
    """
    axp.plot(data, '-',color=color)
    axp.set_ylabel(ylabel, color=color)
    axp.spines[label_side].set_color(color)
    for label in axp.yaxis.get_ticklabels():
        label.set_color(color)


def plot_fire_results(grid, area_trees, area_fire, time_lim = None):
    """
    Plot the results of a forest fire simulation.

    An aerial view of the final grid, and time series
    of the fire-covered and tree-covered areas on the
    same panel, with colored axes
    """
    fig = plt.figure(figsize=(8, 4), dpi=300)

    ax_map = fig.add_subplot(1, 2, 1, aspect='equal')
    show_forest(grid, ax_map)

    ax_tree = fig.add_subplot(1, 2, 2)
    ax_fire = ax_tree.twinx()

    ax_tree.set_xlabel('Time step')
    if not time_lim is None:
        ax_tree.set_xlim(time_lim)

    plot_time_series(area_trees, ax_tree, TREE_COLOR, 'Tree area', 'left')
    plot_time_series(area_fire, ax_fire, FIRE_COLOR, 'Fire area', 'right')


    fig.subplots_adjust(wspace=0.25, bottom = 0.25)

    return fig

if  __name__ == '__main__':
    plot_fire_results(*read_netcdf('forest_grid_004000.nc'), time_lim=(0,4000))

#end
