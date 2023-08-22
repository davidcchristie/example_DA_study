"""This script is used to build the base collider with Xmask, configuring only the optics. Functions
in this script are called sequentially.  This version puts the particle distribution in a cartesian grid"""
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
import optics_specific_tools as ost


# ==================================================================================================
# --- Function for tree_maker tagging
# ==================================================================================================
def tree_maker_tagging(config, tag="started"):
    # Start tree_maker logging if log_file is present in config
    if tree_maker is not None and "log_file" in config:
        tree_maker.tag_json.tag_it(config["log_file"], tag)
    else:
        logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Function to load configuration file
# ==================================================================================================
def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_particles = configuration["config_particles"]
    config_mad = configuration["config_mad"]

    return configuration, config_particles, config_mad


# ==================================================================================================
# --- Function to build particle distribution and write it to file
# ==================================================================================================

def build_particle_distribution(config_particles):
    nOuterSquare = int(config_particles["n_x"]/config_particles["delta_x"])
    linXY = np.linspace(0, config_particles["r_max"], nOuterSquare)
    [X,Y] = np.meshgrid(linXY, linXY)
    # make X and Y into 1D arrays
    X = X.flatten()
    Y = Y.flatten()
    R = np.sqrt(X**2 + Y**2)
    # delete the points with R>= d_config_particles["r_max"] and R<= d_config_particles["r_min"]
    X = np.delete(X, np.where((R >= config_particles["r_max"]) | (R <= config_particles["r_min"])))
    Y = np.delete(Y, np.where((R >= config_particles["r_max"]) | (R <= config_particles["r_min"])))


    # make a list of tuples
    particle_list = [(particle_id, ii[0], ii[1]) for particle_id, ii in enumerate(zip(X, Y))]

    # how many particles have R <= r_max and R >= r_min?

    # Split distribution into several chunks for parallelization
    n_split = config_particles["n_split"]
    particle_list = list(np.array_split(particle_list, n_split))

    # Return distribution
    return particle_list

def write_particle_distribution(particle_list):
    # Write distribution to parquet files
    distributions_folder = "particles"
    os.makedirs(distributions_folder, exist_ok=True)
    for idx_chunk, my_list in enumerate(particle_list):
        pd.DataFrame(
            my_list,
            columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"],
        ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")


# ==================================================================================================
# --- Function to build collider from mad model
# ==================================================================================================
def build_collider_from_mad(config_mad, sanity_checks=True):
    # Make mad environment
    xm.make_mad_environment(links=config_mad["links"])

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1)
    ost.build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])

    if sanity_checks:
        mad_b1b2.use(sequence="lhcb1")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)
        mad_b1b2.use(sequence="lhcb2")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Apply optics (only for b4, just for check)
    ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])
    if sanity_checks:
        mad_b4.use(sequence="lhcb2")
        mad_b4.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad["beam_config"],
        enable_imperfections=config_mad["enable_imperfections"],
        enable_knob_synthesis=config_mad["enable_knob_synthesis"],
        rename_coupling_knobs=config_mad["rename_coupling_knobs"],
        pars_for_imperfections=config_mad["pars_for_imperfections"],
        ver_lhc_run=config_mad["ver_lhc_run"],
        ver_hllhc_optics=config_mad["ver_hllhc_optics"],
    )
    collider.build_trackers()

    if sanity_checks:
        collider["lhcb1"].twiss(method="4d")
        collider["lhcb2"].twiss(method="4d")
    # Return collider
    return collider


def activate_RF_and_twiss(collider, config_mad, sanity_checks=True):
    # Define a RF system (values are not so immportant as they're defined later)
    print("--- Now Computing Twiss assuming:")
    if config_mad["ver_hllhc_optics"] == 1.6:
        dic_rf = {"vrf400": 16.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.5}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    elif config_mad["ver_lhc_run"] == 3.0:
        dic_rf = {"vrf400": 12.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.0}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    print("---")

    # Rebuild tracker if needed
    try:
        collider.build_trackers()
    except:
        print("Skipping rebuilding tracker")

    for knob, val in dic_rf.items():
        collider.vars[knob] = val

    if sanity_checks:
        for my_line in ["lhcb1", "lhcb2"]:
            ost.check_xsuite_lattices(collider[my_line])

    return collider


def clean():
    # Remove all the temporaty files created in the process of building collider
    os.remove("mad_collider.log")
    os.remove("mad_b4.log")
    shutil.rmtree("temp")
    os.unlink("errors")
    os.unlink("acc-models-lhc")


# ==================================================================================================
# --- Main function for building distribution and collider
# ==================================================================================================
def build_distr_and_collider(config_file="config.yaml"):
    # Get configuration
    configuration, config_particles, config_mad = load_configuration(config_file)

    # Get sanity checks flag
    sanity_checks = configuration["sanity_checks"]

    # Tag start of the job
    tree_maker_tagging(configuration, tag="started")

    # Build particle distribution
    particle_list = build_particle_distribution(config_particles)

    # Write particle distribution to file
    write_particle_distribution(particle_list)

    # Build collider from mad model
    collider = build_collider_from_mad(config_mad, sanity_checks)

    # Twiss to ensure eveyrthing is ok
    collider = activate_RF_and_twiss(collider, config_mad, sanity_checks)

    # Clean temporary files
    clean()

    # Save collider to json
    os.makedirs("collider", exist_ok=True)
    collider.to_json("collider/collider.json")

    # Tag end of the job
    tree_maker_tagging(configuration, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    build_distr_and_collider()
