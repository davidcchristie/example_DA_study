study_name = "tune_scan_more_thetas"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

amplitudeDownsampling = 1
angleDownsampling = 1

drawFourPlots = True

# read df_lost_particles from pickle if the file exists

try:
    df_lost_particles = pd.read_pickle(f"scans/{study_name}/df_lost_particles.pkl")
    df_all_sim = pd.read_pickle(f"scans/{study_name}/df_all_sim.pkl")
except:
    print("No pickle file found. Creating new dataframes.")
    # execute python script 003b_postprocessingNoName.py
    exec(open("003b_postprocessingNoName.py").read())

angleListOrig = np.sort(np.unique(df_all_sim['angle in xy-plane [deg]']))
amplitudeListOrig = np.sort(np.unique(df_all_sim['normalized amplitude in xy-plane']))


# downsample (just set sample rate to 1 if no downsampling is required)
amplitudeList = amplitudeListOrig[::amplitudeDownsampling]
angleList = angleListOrig[::angleDownsampling]

df_downsampled_sim = df_all_sim.loc[df_all_sim['normalized amplitude in xy-plane'].isin(amplitudeList) & df_all_sim['angle in xy-plane [deg]'].isin(angleList)]
df_downsampled_sim.reset_index(drop=True, inplace=True)

df_downsampled_lost = df_lost_particles.loc[df_lost_particles['normalized amplitude in xy-plane'].isin(amplitudeList) & df_lost_particles['angle in xy-plane [deg]'].isin(angleList)]
df_downsampled_lost.reset_index(drop=True, inplace=True)

# get the lost particles' turns, amplitudes and angles
lostPartTurns = df_downsampled_lost["at_turn"].to_numpy()
lostPartAmps = df_downsampled_lost["normalized amplitude in xy-plane"].to_numpy()
lostPartAngles = df_downsampled_lost["angle in xy-plane [deg]"].to_numpy()  
sortIndx = np.argsort(lostPartTurns)
lostPartTurns = lostPartTurns[sortIndx]
df_downsampled_lost = df_downsampled_lost.iloc[sortIndx]


numUniqueAmps = len(amplitudeList)
numAngles = len(angleList)
numTurns = len(lostPartTurns)
numLostParts = len(df_downsampled_lost)

# create a grid of ones with dimensions: length(turnsAllUnique) x length(amplitudesAll) x length(anglesUnique)
surviveParticlesAmpAngle = np.ones((numTurns, numUniqueAmps, numAngles))



# macroparricle charge elements dq: different for each r
r_dr = np.multiply( amplitudeList[:-1], np.diff(amplitudeList))
dtheta = 2*np.pi/numAngles
dq =  dtheta * np.multiply(np.append(r_dr, 1) , np.exp(-amplitudeList**2/2))

# track total charge lost 
dqPerTurn = np.zeros(numTurns)

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

    dqPerTurn[particleNo-1] += dq[ampIndx] # += just in case there's more than one lost per turn!
    
    #  DA for that angle and all subsequent turns (including the turn of loss)
    #  is the minimum of the current DA and the DA of the lost particle
    #  if another particle at that angle and lower amplitude is lost later, the DA will be updated again
    DAbyNandTheta[lastTurnIndx:, angleIndx] = np.nanmin([lastAmp, DAbyNandTheta[lastTurnIndx, angleIndx]])

# number of particles surviving at each turn for each amplitude 
numParticlesTurnsAmp = np.sum(surviveParticlesAmpAngle, axis=2)
particleUnscathed = np.all(surviveParticlesAmpAngle>0, axis=0) # survives all turns

# number of particles surviving at each turn 
numParticlesTot = np.sum(numParticlesTurnsAmp, axis=1)

numMegaturnsSurvived = np.sum(surviveParticlesAmpAngle, axis=0)
numTurnsSurvived = lostPartTurns[numMegaturnsSurvived.astype(int)-1]

# # do any particles survive at each amplitude?
numParticlesAmp = np.sum(surviveParticlesAmpAngle, axis=2)
lastTurnAllPresent = np.full((numUniqueAmps), np.nan)
firstTurnNoParticles = np.full((numUniqueAmps), np.nan)
firstTurnHalfGone = np.full((numUniqueAmps), np.nan)
numPartsHalfGone = round(numAngles/2)

for ampIndx in range(numUniqueAmps):
    popByTurn = numParticlesAmp[:, ampIndx]

    # turnsWithSomeParticles  = lostPartTurns[popByTurn > 0]
    turnsWithAllParticles = lostPartTurns[popByTurn == numAngles]
    turnsWithNoParticles = lostPartTurns[popByTurn == 0]
    turnsHalfGone = lostPartTurns[popByTurn <= numPartsHalfGone]

    if len(turnsWithAllParticles) > 0:
        lastTurnAllPresent[ampIndx] = turnsWithAllParticles[-1]

    if len(turnsWithNoParticles) > 0:
        firstTurnNoParticles[ampIndx] = turnsWithNoParticles[0]   

    if len(turnsHalfGone) > 0:
        firstTurnHalfGone[ampIndx] = turnsHalfGone[0]   







# anyParticlesAmp = numParticlesAmp > 0
# allParticlesAmp = numParticlesAmp == numAngles

# firstTurn



if drawFourPlots:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(lostPartTurns, numParticlesTot)
    plt.xlabel('Number of Turns')
    plt.ylabel('Number of Survivors')
    plt.title('Population at each turn')

    plt.subplot(2, 2, 2)
    # plot firstTurnWithLosses vs. amplitude and firstTurnWithAllLosses vs. amplitude
    plt.plot(amplitudeList, lastTurnAllPresent, label='Last turn all present')
    plt.plot(amplitudeList, firstTurnHalfGone, label='First turn half gone')
    plt.plot(amplitudeList, firstTurnNoParticles, label='First Turn with losses')
    plt.xlabel('Amplitude (\sigma)')
    plt.ylabel('Number of Turns')
    plt.title('When are Particles Lost?')
    plt.legend()


    # plt.xlabel('Amplitude (\sigma)')
    # plt.ylabel('Number of Intact Turns')
    # plt.title('Number of turns for each amplitude before any particles are lost')

    # plt.subplot(2, 2, 3)
    # plt.plot(turnsAll, minAmpWithLosses)
    # plt.xlabel('Number of Turns')
    # plt.ylabel('Min Amplitude of Losses (\sigma)')
    # # plt.ylim(7, 25)
    # plt.title('Amplitude Thresholds')
    # DA = minAmpWithLosses.min()
    # indxDA = np.argmin(minAmpWithLosses)
    # plt.plot(turnsAll[indxDA], DA, 'r*')
    # plt.text(turnsAll[indxDA], DA, f'DA = {DA}', horizontalalignment='center', verticalalignment='top')

    plt.subplot(2, 2, 4)
    # generate pcolor plot of numParticlesAmp against number of turns and amplitude
    TURNS_ALL, AMPS_ALL = np.meshgrid( lostPartTurns, amplitudeList)
    P = plt.pcolor(TURNS_ALL, AMPS_ALL, numParticlesTurnsAmp.transpose(), shading='auto')
    plt.xlabel('Number of Turns')
    plt.ylabel('Amplitude ($\sigma$)')
    C = plt.colorbar(P)
    C.set_label("Num. Surviving Particles")
    plt.title('Particle Survival')
    plt.savefig(f"scans/{study_name}/particleSurvival.png")
    plt.show()















# # find the first turn at which any particles are lost at each amplitude
# firstTurnWithLosses = np.argmax(anyParticlesAmp, axis=0)

# # find the location of the first entry in each row of anyParticlesAmp that is False
# # this is the first turn at which all particles are lost at each amplitude
# for i in range(numUniqueAmps):
#     firstTurnWithAllLosses[i] = np.argmax(~anyParticlesAmp[i, :])

# firstTurnIndxWithAnyLosses =np.argmax(~allParticlesAmp, axis=0)
# firstTurnWithAnyLosses = lostPartTurns[firstTurnIndxWithAnyLosses]

# firstTurnIndxWithAllLosses =np.argmax(~anyParticlesAmp, axis=0)
# firstTurnWithAllLosses = lostPartTurns[firstTurnIndxWithAllLosses]



# firstTurnWithLosses = np.argmax(numParticlesTot > 0)

# # find the first turn at which all particles are lost at each amplitude
# firstTurnWithAllLosses = np.argmax(allParticlesAmp, axis=0)