"""Utility functions for cosmo_analysis.

This module provides various helper functions for array operations,
file handling, and cosmological calculations.
"""

import bisect as bsct
import h5py

def getClosestIdx(myList, myNumber):
    """Get the index of the closest but bigger element in a list.
    
    Args:
        myList: Sorted list of numbers
        myNumber: Value to find closest index for
    
    Returns:
        int: Index in list where myNumber would be inserted
    """
    return bsct.bisect_left(myList, myNumber)

def maxIdx(arr):
    """Get the index of maximum value in iterable.
    
    Args:
        arr: Array-like object
    
    Returns:
        int: Index of maximum element
    """
    maxAux = 0
    for i, el in enumerate(arr):
        if el > arr[maxAux]:
            maxAux = i
    return maxAux

def findNmax(arr, N):
    """Get indices of N maximum values in iterable.
    
    WARNING: This function modifies the input array. Send a copy if you need
    to preserve the original.
    
    Args:
        arr: Array-like object (will be modified)
        N: Number of maximum values to find
    
    Returns:
        list: Indices of N maximum elements
    """
    minV = min(arr)
    maxes = [0]*N
    for i in range(N):
        max1 = maxIdx(arr)
        maxes[i] = max1
        arr[max1] = minV
    return maxes

def h5printR(item, leading=''):
    """Recursively print HDF5 file structure.
    
    Args:
        item: HDF5 file or group object
        leading: Leading whitespace for indentation
    """
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

def h5print(filename):
    """Print HDF5 file structure.
    
    Args:
        filename: HDF5 file object to print structure of
    """
    h5printR(filename, '  ')

def padN(n, pad=3):
    """Pad a number with zeros and convert to string.
    
    Args:
        n: Number to pad
        pad: Total width of padded string (default: 3)
    
    Returns:
        str: Zero-padded string representation
    
    Example:
        >>> padN(5)
        '005'
        >>> padN(42, pad=5)
        '00042'
    """
    return str(n).zfill(pad)

def getZ(a):
    """Convert scale factor to redshift.
    
    Args:
        a: Scale factor
    
    Returns:
        float: Redshift (z = 1/a - 1)
    """
    return 1/a - 1

def getA(z):
    """Convert redshift to scale factor.
    
    Args:
        z: Redshift
    
    Returns:
        float: Scale factor (a = 1/(z + 1))
    """
    return 1/(z + 1)

def tToIdx(sim, time):
    """Get snapshot index closest to specified time.
    
    Args:
        sim: Simulation object with loaded snapshots
        time: Time in Myr
    
    Returns:
        int: Index of snapshot closest to specified time
    """
    dist = abs(sim.snap[0].time - time)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].time - time)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx

def zToIdx(sim, z):
    """Get snapshot index closest to specified redshift.
    
    Args:
        sim: Simulation object with loaded snapshots
        z: Redshift
    
    Returns:
        int: Index of snapshot closest to specified redshift
    """
    dist = abs(sim.snap[0].z - z)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].z - z)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx
