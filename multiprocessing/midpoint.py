# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:31:13 2022

@author: steph
"""

from multiprocessing import Pool
import numpy as np

def midpoint_quad(f, a, b, n):
    """
    
    Integrate f(x) numerically over the interval
    [a, b]. [a, b] is divided into n sub-intervals
    and the midpoint rule applied to each 
    
    Parameters
    ----------
    f : function(x), the integrand
    a : float-like, the lower limit
    b : float-like, the upper limit
    n : integer, number of subdivisions

    Returns
    -------
    F : float-like, the approximate integral
    """
    
    h = (b - a)/n
    x = np.arange(a + 0.5 * h, b, h)
    return np.sum( f(x) * h)
 
def circle(x):
    """
    y(x) for the unit circle, valid for 0 <= x <= 1
    
    Parameters
    ----------
    x : ndarray, values of x

    Returns
    -------
    y: ndarray, y(x)

    """
    return np.sqrt(1 - x**2)


def midpoint_quad_wrapper(arg):
    """ 
    wrapper function that unpacks a tuple arg 
    """
    return midpoint_quad(*arg)


# divide the interval [0, 1] into how many INTERVALS?
INTERVALS = 40
# split 
PARTITIONS = 10
# check that INTERVALS / PARTITIONS is an integer!
assert (INTERVALS % PARTITIONS == 0), 'INTERVALS / PARTITIONS not an integer'

partition_size = 1.0/PARTITIONS    

# construct the arg list - needs to a list of (f, a, b, n) tuples
# where f, a, b, n match the arguments of midpoint_quad    
arg_list = [(circle, a, a + partition_size, int(INTERVALS/PARTITIONS) ) 
        for a in np.linspace(0, 1 - partition_size, PARTITIONS)]

if __name__ == '__main__':

    print ('estimating pi/4' )
    
    with Pool(PARTITIONS) as pool:
        part = pool.map(midpoint_quad_wrapper, arg_list)
        print (f' contributions: {part}')
        print (f' total: {np.sum(part)} (pi/4 =  {np.pi / 4})' )