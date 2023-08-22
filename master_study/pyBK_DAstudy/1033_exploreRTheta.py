# ==================================================================================================
# --- Imports
# ==================================================================================================
import tree_maker
import yaml
import pandas as pd
import time
import logging
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d


# ==================================================================================================
# --- Load tree of jobs
# ==================================================================================================

# Start of the script
print("Analysis of output simulation files started")
start = time.time()

# Load Data
study_name = "tune_scan_more_thetas"  # "octupoleScanMoreThetas" # 
fix = "/scans/" + study_name
root = tree_maker.tree_from_json(fix[1:] + "/tree_maker_" + study_name + ".json")
# Add suffix to the root node path to handle scans that are not in the root directory
root.add_suffix(suffix=fix)

###################################################################################################
# ==================================================================================================
# --- # Browse simulations folder and extract relevant observables
# ==================================================================================================
l_problematic_sim = []
l_df_to_merge = []
for node in root.generation(1):
    with open(f"{node.get_abs_path()}/config.yaml", "r") as fid:
        config_parent = yaml.safe_load(fid)
    for node_child in node.children:
        with open(f"{node_child.get_abs_path()}/config.yaml", "r") as fid:
            config_child = yaml.safe_load(fid)
        try:
            particle = pd.read_parquet(
                f"{node_child.get_abs_path()}/{config_child['config_simulation']['particle_file']}"
            )
            df_sim = pd.read_parquet(f"{node_child.get_abs_path()}/output_particles.parquet")

        except Exception as e:
            print(e)
            l_problematic_sim.append(node_child.get_abs_path())
            continue

        # Register paths and names of the nodes
        df_sim["path base collider"] = f"{node.get_abs_path()}"
        df_sim["name base collider"] = f"{node.name}"
        df_sim["path simulation"] = f"{node_child.get_abs_path()}"
        df_sim["name simulation"] = f"{node_child.name}"

        # Get node parameters as dictionnaries for parameter assignation
        dic_child_collider = node_child.parameters["config_collider"]
        dic_child_simulation = node_child.parameters["config_simulation"]
        try:
            dic_parent_collider = node.parameters["config_mad"]
        except:
            print("No parent collider could be loaded")
        dic_parent_particles = node.parameters["config_particles"]

        # Get which beam is being tracked
        df_sim["beam"] = dic_child_simulation["beam"]

        # Get scanned parameters (complete with the requested scanned parameters)
        df_sim["qx"] = dic_child_collider["config_knobs_and_tuning"]["qx"]["lhcb1"]
        df_sim["qy"] = dic_child_collider["config_knobs_and_tuning"]["qy"]["lhcb1"]
        df_sim["dqx"] = dic_child_collider["config_knobs_and_tuning"]["dqx"]["lhcb1"]
        df_sim["dqy"] = dic_child_collider["config_knobs_and_tuning"]["dqy"]["lhcb1"]
        df_sim["i_bunch_b1"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b1"
        ]
        df_sim["i_bunch_b2"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b2"
        ]
        df_sim["num_particles_per_bunch"] = dic_child_collider["config_beambeam"][
            "num_particles_per_bunch"
        ]
        df_sim["i_oct_b1"] = dic_child_collider["config_knobs_and_tuning"]["knob_settings"]["i_oct_b1"]
        df_sim["i_oct_b2"] = dic_child_collider["config_knobs_and_tuning"]["knob_settings"]["i_oct_b2"]

        # Merge with particle data
        df_sim_with_particle = pd.merge(df_sim, particle, on=["particle_id"])
        l_df_to_merge.append(df_sim_with_particle)

# ==================================================================================================
# --- # Merge all jobs outputs in one dataframe and save it
# ==================================================================================================

# Merge the dataframes from all simulations together
df_all_sim = pd.concat(l_df_to_merge)

# Extract the particles that were lost for DA computation
df_lost_particles = df_all_sim[df_all_sim["state"] != 1]  # Lost particles

# Check if the dataframe is empty
if df_lost_particles.empty:
    print("No unstable particles found, the output dataframe will be empty.")

# Group by working point (Update this with the knobs you want to group by !)
group_by_parameters = ["name base collider", "qx", "qy"]
# We always want to keep beam in the final result
group_by_parameters = ["beam"] + group_by_parameters
l_parameters_to_keep = [
    "normalized amplitude in xy-plane",
    "qx",
    "qy",
    "dqx",
    "dqy",
    "i_bunch_b1",
    "i_bunch_b2",
    "num_particles_per_bunch",
    "i_oct_b1",
    "i_oct_b2",
]

# Min is computed in the groupby function, but values should be identical
my_final = pd.DataFrame(
    [
        df_lost_particles.groupby(group_by_parameters)[parameter].min()
        for parameter in l_parameters_to_keep
    ]
).transpose()

# Save data and print time
my_final.to_parquet(f"scans/{study_name}/da.parquet")
print("Final dataframe for current set of simulations: ", my_final)
end = time.time()
print("Elapsed time: ", end - start)


# Min is computed in the groupby function, but values should be identical
my_full = pd.DataFrame(
    [
        df_all_sim.groupby(group_by_parameters)[parameter].min()
        for parameter in l_parameters_to_keep
    ]
).transpose()

# Save data and print time
my_full.to_parquet(f"scans/{study_name}/allParts.parquet")
print("Final dataframe for current set of simulations: ", my_final)
end = time.time()
print("Elapsed time: ", end - start)

# Scanning parameters potential
l_sc = [
    "normalized amplitude in xy-plane",
    "qx",
    "qy",
    "dqx",
    "dqy",
    "i_bunch_b1",
    "num_particles_per_bunch",
    "i_oct_b1",
    "angle in xy-plane [deg]"
]

# so, the amplitude is always going to be the first in the list!



scanningParameters = []
scanningParameterIndices = []
scanLengths = []
scanAxes = []




for i in l_sc:
    # print(i) 
    numUnique = len(df_all_sim[i].unique())
    if numUnique > 1:
        print(i)
        print(df_all_sim[i].unique())
        scanningParameterIndices.append(l_sc.index(i))
        scanningParameters.append(i)
        scanAxes.append(df_all_sim[i].unique())
        scanLengths.append(numUnique)





number_of_scanning_parameters = len(scanningParameters)
print("number_of_scanning_parameters: ", number_of_scanning_parameters)

# P = grid of ones with dimensions scanLengths
particleCounts = np.zeros(scanLengths)

# if number_of_scanning_parameters == 2:
#     for i in range(scanLengths[0]):
#         for j in range(scanLengths[1]):
#             particleCounts[i,j] = len(my_full[(my_full[scanningParameters[0]] == my_full[scanningParameters[0]].unique()[i]) & (my_full[scanningParameters[1]] == my_full[scanningParameters[1]].unique()[j])])

particleCounts = np.zeros(scanLengths)
numTurns = np.zeros(scanLengths)

# create doesParticleSurvive as an array of false bools with dimensions scanLengths
doesParticleSurvive = np.zeros(scanLengths, dtype=bool)

numTotalParts = len(df_all_sim)


for i in range(numTotalParts):
    paramLocs = []
    for j in range(number_of_scanning_parameters):
        paramValue = df_all_sim[scanningParameters[j]].to_numpy()[i]
        paramLocs.append(np.where(scanAxes[j]== paramValue)[0][0])
    numTurns[tuple(paramLocs)] = df_all_sim["at_turn"].to_numpy()[i]
    if df_all_sim["state"].to_numpy()[i] == 1:
       doesParticleSurvive[tuple(paramLocs)] = True
       particleCounts[tuple(paramLocs)] += 1

print(scanningParameters)



some_particles_survive = np.any(doesParticleSurvive, axis=-1)
all_particles_survive = np.all(doesParticleSurvive, axis=-1)
num_survivors_all_angles = np.sum(doesParticleSurvive, axis=-1)

# particle amplitude should be the first dimension (it was first on the list!)
assert(scanningParameters.index("normalized amplitude in xy-plane") ==0)
ampList = scanAxes[0]



# indices of the other dimensions (not amplitude) are 1, ..., number_of_scanning_parameters-1
scanLengthsReduced = scanLengths[1:]

# lenght of vector scanLengthsReduced
numReduced = len(scanLengthsReduced)

if numReduced==1:
    minLostAmplitudes = np.zeros(scanLengthsReduced) 
    maxSurvivingAmplitudes = np.zeros(scanLengthsReduced)
    for i in range (scanLengthsReduced[0]):
        popVector = doesParticleSurvive[:, i]
        if np.any(popVector):
                # index of last surviving particle
            minLostAmplitudes[i] = np.min(ampList[np.where(~popVector)[0]]) # this is the first amplitude that doesn't survive 
            maxSurvivingAmplitudes[i] = np.max(ampList[np.where(popVector)[0]]) # this is the last amplitude that survives

    # plot minLostAmplitudes vs. scanAxes[1] and maxSurvivingAmplitudes vs. scanAxes[1] on the same plot as points
    plt.scatter(scanAxes[1], maxSurvivingAmplitudes, label='Max. Amplitude Surviving')
    plt.scatter(scanAxes[1], minLostAmplitudes, label='Min. Amplitude Lost')

    plt.xlabel(scanningParameters[1])
    plt.ylabel('Amplitude (sigma))')
    plt.legend()
    plt.show()

    plt.scatter(minLostAmplitudes, maxSurvivingAmplitudes)
    plt.xlabel('min lost amplitude')
    plt.ylabel('max surviving amplitude')
    plt.legend()
    plt.show()



if numReduced==2:
    DA_table = np.zeros(scanLengthsReduced)
    for i in range (scanLengthsReduced[0]):
        for j in range (scanLengthsReduced[1]):
            popVector = doesParticleSurvive[:, i,j]
            if np.any(popVector):
                # index of last surviving particle
                DA_table[i,j] = ampList[np.where(popVector)[0][-1]] # this is the last amplitude that survives 


