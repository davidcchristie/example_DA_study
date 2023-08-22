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

# ==================================================================================================
# --- Load tree of jobs
# ==================================================================================================

# Start of the script
print("Analysis of output simulation files started")
start = time.time()

# Load Data
study_name = "no_bb_higher_r" # "example_tunescan_bb_off_23_09_06"
fix = "/scans/" + study_name
root = tree_maker.tree_from_json(fix[1:] + "/tree_maker_" + study_name + ".json")
# Add suffix to the root node path to handle scans that are not in the root directory
root.add_suffix(suffix=fix)


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

numLostParts = len(df_lost_particles)
totNumTurns = df_all_sim['at_turn'].max()


# Create 2D amplitude x turns grids
amplitudeList, indsOrig, indsNew = np.unique(df_all_sim['normalized amplitude in xy-plane'], return_inverse=True, return_index=True)
numUniqueAmps = len(indsOrig)
numTotalAmps = len(indsNew)
anglesPerAmp = numTotalAmps // numUniqueAmps
amplitudesAll = np.array(amplitudeList)
turnsAll = np.arange(1, totNumTurns + 1)
numParticlesAmp = np.tile(anglesPerAmp, (numUniqueAmps, totNumTurns))


lostPartTurns = df_lost_particles["at_turn"].to_numpy()
lostPartAmps = df_lost_particles["normalized amplitude in xy-plane"].to_numpy()



# Populate the grids
for particleNo in range(1,numLostParts):
    print(particleNo) 
    lastTurn = lostPartTurns[particleNo-1]
    lastAmp = lostPartAmps[particleNo-1]
    # find the index of the amplitude in the list of amplitudes
    ampIndx = np.where(amplitudesAll == lastAmp)[0][0]
    numParticlesAmp[ampIndx, lastTurn:] -= 1

# Flatten out amplitude for turns vs. population
numParticlesRemainingPerTurn = np.sum(numParticlesAmp, axis=0)

# Final population
ParticlesReminingFinalTurn = numParticlesAmp[:, -1]

TURNS_ALL, AMPS_ALL = np.meshgrid( turnsAll, amplitudesAll)
TURNS_ALL[numParticlesAmp < anglesPerAmp] = 0
AMPS_ALL[numParticlesAmp == anglesPerAmp] = 100
intactTurnsPerAmp = np.nanmax(TURNS_ALL, axis=1)
minAmpWithLosses = np.nanmin(AMPS_ALL, axis=0)

downsampleRate = 500
turnsCoarser = turnsAll[::downsampleRate]
numParticlesAmpCoarser = numParticlesAmp[:, ::downsampleRate]


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(turnsAll, numParticlesRemainingPerTurn)
plt.xlabel('Number of Turns')
plt.ylabel('Number of Survivors')
plt.title('Population at each turn')

plt.subplot(2, 2, 2)
plt.plot(amplitudeList, intactTurnsPerAmp)
plt.xlabel('Amplitude (\sigma)')
plt.ylabel('Number of Intact Turns')
plt.title('Number of turns for each amplitude before any particles are lost')

plt.subplot(2, 2, 3)
plt.plot(turnsAll, minAmpWithLosses)
plt.xlabel('Number of Turns')
plt.ylabel('Min Amplitude of Losses (\sigma)')
plt.ylim(7, 25)
plt.title('Amplitude Thresholds')
DA = minAmpWithLosses.min()
indxDA = np.argmin(minAmpWithLosses)
plt.plot(turnsAll[indxDA], DA, 'r*')
plt.text(turnsAll[indxDA], DA, f'DA = {DA}', horizontalalignment='center', verticalalignment='top')

plt.subplot(2, 2, 4)
AMPS, TURNS = np.meshgrid(amplitudeList, turnsCoarser)
P = plt.pcolor(TURNS, AMPS, numParticlesAmpCoarser.transpose(), shading='auto')
plt.xlabel('Number of Turns')
plt.ylabel('Amplitude (\sigma)')
C = plt.colorbar(P)
C.set_label("Num. Surviving Particles")
plt.title('Particle Survival')

plt.tight_layout()
plt.show()