# study_name =  "tune_scan_more_thetas"
# study_name = "BB_moreRTheta"
# study_name = "no_bb_higher_r"
study_name = "noDAWider"

import importlib
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fitDA


try:
    df_lost_particles = pd.read_pickle(f"scans/{study_name}/df_lost_particles.pkl")
    df_all_sim = pd.read_pickle(f"scans/{study_name}/df_all_sim.pkl")
except:
    print("No pickle file found. Creating new dataframes.")
    # execute python script 003b_postprocessingNoName.py
    exec(open("003b_postprocessingNoName.py").read())


angleListOrig = np.sort(np.unique(df_all_sim['angle in xy-plane [deg]']))
amplitudeListOrig = np.sort(np.unique(df_all_sim['normalized amplitude in xy-plane']))

minDA = []
maxDA = []
meanDA = []
lostPartTurnsSet = []

AmpAngleScales = [[1,1], [1, 2], [1,3], [2, 1], [3, 1], [2,2], [3,3]]
for AmpAnglePair in AmpAngleScales:
    print(AmpAnglePair)
    amplitudeDownsampling = AmpAnglePair[0]
    angleDownsampling = AmpAnglePair[1]

    # downsample (just set sample rate to 1 if no downsampling is required)
    amplitudeList = amplitudeListOrig[::amplitudeDownsampling]
    angleList = angleListOrig[::angleDownsampling]

    df_downsampled_sim = df_all_sim.loc[df_all_sim['normalized amplitude in xy-plane'].isin(amplitudeList) & df_all_sim['angle in xy-plane [deg]'].isin(angleList)]
    df_downsampled_sim.reset_index(drop=True, inplace=True)

    df_downsampled_lost = df_lost_particles.loc[df_lost_particles['normalized amplitude in xy-plane'].isin(amplitudeList) & df_lost_particles['angle in xy-plane [deg]'].isin(angleList)]
    df_downsampled_lost.reset_index(drop=True, inplace=True)

    # get the lost particles' turns, amplitudes and angles
    lostPartTurns = df_downsampled_lost["at_turn"].to_numpy()

    # sort df_downsampled_lost by "at_turn" in ascending order
    sortIndx = np.argsort(lostPartTurns)
    lostPartTurns = lostPartTurns[sortIndx]
    df_downsampled_lost = df_downsampled_lost.iloc[sortIndx]
    lostPartAmps = df_downsampled_lost["normalized amplitude in xy-plane"].to_numpy()
    lostPartAngles = df_downsampled_lost["angle in xy-plane [deg]"].to_numpy()  

    numUniqueAmps = len(amplitudeList)
    numAngles = len(angleList)
    numTurns = len(lostPartTurns)
    numLostParts = len(df_downsampled_lost)

# create a grid of ones with dimensions: length(turnsAllUnique) x length(amplitudesAll) x length(anglesUnique)
    surviveParticlesAmpAngle = np.ones((numTurns, numUniqueAmps, numAngles))


# track change in DA separately for each angular bin
    DAbyNandTheta = np.full((numTurns, numAngles), np.nan)

    # when is each particle lost?
    lastTurnRTheta = np.full([numUniqueAmps, numAngles], np.nan)

    # DAbyNandTheta gives dynamic aperture for each angular bin and each turn
    # surviveParticlesAmpAngle shows particle survival per turn for each amplitude and each angle
    # iterate over all particles in dataframe lost_particles
    for particleNo in range(1,numLostParts):
        lastTurn = lostPartTurns[particleNo-1]
        lastAmp = lostPartAmps[particleNo-1]
        lastAngle = lostPartAngles[particleNo-1]
        # find the index of the amplitude in the list of amplitudes
        ampIndx = np.where(amplitudeList == lastAmp)[0][0]
        angleIndx = np.where(angleList == lastAngle)[0][0]
        lastTurnIndx = np.where(lostPartTurns == lastTurn)[0][0]


        surviveParticlesAmpAngle[lastTurnIndx:, ampIndx, angleIndx] = 0
        lastTurnRTheta[ampIndx, angleIndx] = lastTurn


        #  DA for that angle and all subsequent turns (including the turn of loss)
        #  is the minimum of the current DA and the DA of the lost particle
        #  if another particle at that angle and lower amplitude is lost later, the DA will be updated again
        DAbyNandTheta[lastTurnIndx:, angleIndx] = np.nanmin([lastAmp, DAbyNandTheta[lastTurnIndx, angleIndx]])



    minDA.append([np.nanmin(DAbyNandTheta, 1)])
    maxDA.append([np.nanmax(DAbyNandTheta, 1)])
    meanDA.append([np.nanmean(DAbyNandTheta, 1)])
    lostPartTurnsSet.append([lostPartTurns])


# plot lostpartturnsset vs meanda for each entry in meanda

plt.plot(lostPartTurnsSet[0][0], meanDA[0][0])
plt.plot(lostPartTurnsSet[1][0], meanDA[1][0])
plt.plot(lostPartTurnsSet[2][0], meanDA[2][0])
plt.plot(lostPartTurnsSet[3][0], meanDA[3][0])
plt.plot(lostPartTurnsSet[4][0], meanDA[4][0])
plt.plot(lostPartTurnsSet[5][0], meanDA[5][0])
plt.plot(lostPartTurnsSet[6][0], meanDA[6][0])
#use AmpAngleScales as labels
plt.legend([str(AmpAngleScales[0]), str(AmpAngleScales[1]), str(AmpAngleScales[2]), str(AmpAngleScales[3]), str(AmpAngleScales[4]), str(AmpAngleScales[5]), str(AmpAngleScales[6])])
plt.show()

# do the same plot but with minda
plt.plot(lostPartTurnsSet[0][0], minDA[0][0])
plt.plot(lostPartTurnsSet[1][0], minDA[1][0])
plt.plot(lostPartTurnsSet[2][0], minDA[2][0])
plt.plot(lostPartTurnsSet[3][0], minDA[3][0])
plt.plot(lostPartTurnsSet[4][0], minDA[4][0])
plt.plot(lostPartTurnsSet[5][0], minDA[5][0])
plt.plot(lostPartTurnsSet[6][0], minDA[6][0])
#use AmpAngleScales as labels
plt.legend([str(AmpAngleScales[0]), str(AmpAngleScales[1]), str(AmpAngleScales[2]), str(AmpAngleScales[3]), str(AmpAngleScales[4]), str(AmpAngleScales[5]), str(AmpAngleScales[6])])
plt.show()
