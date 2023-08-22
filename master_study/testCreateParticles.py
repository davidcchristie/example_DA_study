"""This is just to test the particle distribution generation for polar and cartesian modes"""
# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import json
import yaml
import logging
import numpy as np
import itertools
import pandas as pd
import tree_maker

# Import user-defined optics-specific tools
# import optics_specific_tools as ost

d_config_particles = {}

# Radius of the initial particle distribution
d_config_particles["r_min"] = 15
d_config_particles["n_split"] = 4
d_config_particles["r_max"] = 27
d_config_particles["n_r"] = 8 * (d_config_particles["r_max"] - d_config_particles["r_min"])
d_config_particles["n_x"] = 30
d_config_particles["delta_x"] = 0.5

# Number of angles for the initial particle distribution
d_config_particles["n_angles"] = 12

def build_polar_distribution(config_particles):
    # Define radius distribution
    r_min = config_particles["r_min"]
    r_max = config_particles["r_max"]
    n_r = config_particles["n_r"]
    radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

    # Define angle distribution
    n_angles = config_particles["n_angles"]
    theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

    # Define particle distribution as a cartesian product of the above
    particle_list = [
        (particle_id, ii[1], ii[0])
        for particle_id, ii in enumerate(itertools.product(theta_list, radial_list))
    ]

    # Split distribution into several chunks for parallelization
    n_split = config_particles["n_split"]
    particle_list = list(np.array_split(particle_list, n_split))

    # Return distribution
    return particle_list

    

def build_cartesian_distribution(config_particles):
    nOuterSquare = int(d_config_particles["n_x"]/d_config_particles["delta_x"])
    linXY = np.linspace(0, d_config_particles["r_max"], nOuterSquare)
    [X,Y] = np.meshgrid(linXY, linXY)
    # make X and Y into 1D arrays
    X = X.flatten()
    Y = Y.flatten()
    R = np.sqrt(X**2 + Y**2)
    # delete the points with R>= d_config_particles["r_max"] and R<= d_config_particles["r_min"]
    X = np.delete(X, np.where((R >= d_config_particles["r_max"]) | (R <= d_config_particles["r_min"])))
    Y = np.delete(Y, np.where((R >= d_config_particles["r_max"]) | (R <= d_config_particles["r_min"])))


    # make a list of tuples
    particle_list = [(particle_id, ii[0], ii[1]) for particle_id, ii in enumerate(zip(X, Y))]

    # how many particles have R <= r_max and R >= r_min?

    # Split distribution into several chunks for parallelization
    n_split = config_particles["n_split"]
    particle_list = list(np.array_split(particle_list, n_split))

    # Return distribution
    return particle_list




particle_listRT = build_polar_distribution(d_config_particles)
print('particle_listRT: ', particle_listRT)

particle_listXY = build_cartesian_distribution(d_config_particles)
print('particle_listXY: ', particle_listXY)