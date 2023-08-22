# study_name =  "tune_scan_more_thetas"
# study_name = "BB_moreRTheta"
study_name = "cartMoreNoBB"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from scipy.optimize import curve_fit
import importlib
import fitDA as fitDA

XYDownSampling = 1

drawFourPlots = True
angleBinSizeDegs = 2.5
angleBinList = np.arange(0, 90+angleBinSizeDegs, angleBinSizeDegs)
# read df_lost_particles from pickle if the file exists

try:
    df_lost_particles = pd.read_pickle(f"scans/{study_name}/df_lost_particles.pkl")
    df_all_sim = pd.read_pickle(f"scans/{study_name}/df_all_sim.pkl")
except:
    print("No pickle file found. Creating new dataframes.")
    # execute python script 003b_postprocessingNoName.py
    exec(open("003b_postprocessingNoName.py").read())

# add x and y coordinates to df_all_sim
df_all_sim['xOrig'] = np.cos(df_all_sim['angle in xy-plane [deg]']*np.pi/180)*df_all_sim['normalized amplitude in xy-plane']
df_all_sim['yOrig'] = np.sin(df_all_sim['angle in xy-plane [deg]']*np.pi/180)*df_all_sim['normalized amplitude in xy-plane']
df_lost_particles['xOrig'] = np.cos(df_lost_particles['angle in xy-plane [deg]']*np.pi/180)*df_lost_particles['normalized amplitude in xy-plane']
df_lost_particles['yOrig'] = np.sin(df_lost_particles['angle in xy-plane [deg]']*np.pi/180)*df_lost_particles['normalized amplitude in xy-plane']

XYtol = 0.0001
df_all_sim['x'] = np.round(df_all_sim['xOrig']/XYtol)*XYtol
df_all_sim['y'] = np.round(df_all_sim['yOrig']/XYtol)*XYtol
df_lost_particles['x'] = np.round(df_lost_particles['xOrig']/XYtol)*XYtol
df_lost_particles['y'] = np.round(df_lost_particles['yOrig']/XYtol)*XYtol

# find unique X and Y values
XlistUnique = np.unique(df_all_sim['x'])
YlistUnique = np.unique(df_all_sim['y'])

# downsample (just set sample rate to 1 if no downsampling is required)
XlistUse = XlistUnique[::XYDownSampling]
YlistUse = YlistUnique[::XYDownSampling]
df_downsampled_sim = df_all_sim.loc[df_all_sim['x'].isin(XlistUse) & df_all_sim['y'].isin(YlistUse)]
df_downsampled_sim.reset_index(drop=True, inplace=True)

df_downsampled_lost = df_lost_particles.loc[df_lost_particles['x'].isin(XlistUse) & df_lost_particles['y'].isin(YlistUse)]
df_downsampled_lost.reset_index(drop=True, inplace=True)

# get the lost particles' turns, amplitudes and angles
lostPartTurns = df_downsampled_lost["at_turn"].to_numpy()

# sort df_downsampled_lost by "at_turn" in ascending order
sortIndx = np.argsort(lostPartTurns)
lostPartTurns = lostPartTurns[sortIndx]
df_downsampled_lost = df_downsampled_lost.iloc[sortIndx]
lostPartAmps = df_downsampled_lost["normalized amplitude in xy-plane"].to_numpy()
lostPartAngles = df_downsampled_lost["angle in xy-plane [deg]"].to_numpy()  
lostPartX = df_downsampled_lost["x"].to_numpy()
lostPartY = df_downsampled_lost["y"].to_numpy()


allR = df_all_sim["normalized amplitude in xy-plane"].to_numpy()
allTurns = df_all_sim["at_turn"].to_numpy()



# do not execute rest of script
# exit()

numUniqueX = len(XlistUse)
numUniqueY = len(YlistUse)
numTurns = len(lostPartTurns)
numLostParts = len(df_downsampled_lost)
numAngles = len(angleBinList)
# create a grid of xlistuse and ylistuse and put corresponding lostPartTurns in each cell
lastTurn = np.full((numUniqueX, numUniqueY), np.nan)
numParticlesStart = len(df_downsampled_sim)


minDAAll = []
maxDAAll = []
meanDAAll = []
lostPartTurnsSetAll = []

minDAEnd = []
maxDAEnd = []
meanDAEnd = []
# # when is each particle lost?
usePrcts = np.arange(0.1, 1.1, 0.1)
numSets = len(usePrcts)
for setNo in range(numSets):
    FractionToUse = usePrcts[setNo]    
    # create a logical array to select a fraction of the particles (FractionToUse) at random

    numPartsToUse = int(np.round(numLostParts*FractionToUse))
    useParts = [True]*numPartsToUse + [False]*(numLostParts-numPartsToUse)
    np.random.shuffle(useParts)

    minLostByAngle = np.full((numAngles), np.nan)
    runningMinLostAmp = np.full((numLostParts), np.nan)
    runningMaxExistingAmp = np.zeros(numPartsToUse)
    angularMeanDA = np.zeros(numPartsToUse)
    angularMedianDA = np.zeros(numPartsToUse)
    angularMaxDA = np.zeros(numPartsToUse)
    turns = np.zeros(numPartsToUse)

    usedParticleCount = 0
    for particleNo in range(1,numLostParts+1):
        if useParts[particleNo-1]:
            turns[usedParticleCount] = lostPartTurns[particleNo-1]
            lastAmp = lostPartAmps[particleNo-1]
            lastAngle = lostPartAngles[particleNo-1]
        # find the index of the angle in angleList nearest to lastAngle
            angleIndx = np.argmin(np.abs(angleBinList - lastAngle))
            minLostByAngle[angleIndx] = np.nanmin([lastAmp, minLostByAngle[angleIndx]])
            runningMinLostAmp[particleNo-1] = np.nanmin([lastAmp, runningMinLostAmp[particleNo-2]])
            angularMeanDA[usedParticleCount] = np.nanmean(minLostByAngle)
            angularMedianDA[usedParticleCount] = np.nanmedian(minLostByAngle)
            angularMaxDA[usedParticleCount] = np.nanmax(minLostByAngle)
            usedParticleCount += 1

    finalDAmin = runningMinLostAmp[-1]
    finalDAmax = angularMaxDA[-1]
    finalDAmean = angularMeanDA[-1]


    minDAAll.append([runningMinLostAmp])
    meanDAAll.append([angularMeanDA])


    lostPartTurnsSetAll.append(turns)
    minDAEnd.append(finalDAmin)
    maxDAEnd.append(finalDAmax)
    meanDAEnd.append(finalDAmean)



plt.plot(usePrcts, meanDAEnd)