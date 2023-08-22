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



# macroparricle charge elements dq: different for each r
r_dr = np.multiply( amplitudeList[:-1], np.diff(amplitudeList))
dtheta = 2*np.pi/numAngles  # assuming it's equal, fix otherwise!
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

    # how much charge is lost at this turn?
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


# convert to Cartesians for colour plot
x = np.cos(lostPartAngles*np.pi/180)*lostPartAmps
y = np.sin(lostPartAngles*np.pi/180)*lostPartAmps






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



minDA = np.nanmin(DAbyNandTheta, 1)
maxDA = np.nanmax(DAbyNandTheta, 1)
meanDA = np.nanmean(DAbyNandTheta, 1)
deltaQ = np.cumsum(dqPerTurn)
#append leading zero to deltaQ


# def DFitted (N, Dinf, b, kappa):
#     return Dinf*(1+b/((np.log10(N))**kappa))

# # fit a power law to the DA vs. number of turns
# params, cov = curve_fit(DFitted, lostPartTurns, meanDA, p0=[24, 0.4, 1.4])
# D0_fit, b_fit, k_fit = params


if drawFourPlots:


    # plt.subplot(4,2, 1)
    plt.plot(lostPartTurns, numParticlesTot)
    plt.xlabel('Number of Turns')
    plt.ylabel('Number of Survivors')
    plt.title('Population at each turn')
    plt.show()
    # plt.subplot(4,2, 2)
    # plot firstTurnWithLosses vs. amplitude and firstTurnWithAllLosses vs. amplitude
    plt.plot(amplitudeList, lastTurnAllPresent, label='All present (end)')
    plt.plot(amplitudeList, firstTurnHalfGone, label='Half gone (start)')
    plt.plot(amplitudeList, firstTurnNoParticles, label='All lost (start)')
    plt.xlabel('Amplitude (\sigma)')
    plt.ylabel('Number of Turns')
    plt.title('When are Particles Lost?')
    plt.legend()
    plt.show()
    # plt.subplot(4,2, 3)
    # plot maxDA, minDA, meanDA vs. number of turns and shade in between
    plt.plot(lostPartTurns, maxDA, color='blue')
    plt.plot(lostPartTurns, minDA, color='red')
    plt.plot(lostPartTurns, meanDA, color='black', linestyle='dotted')
    plt.fill_between(lostPartTurns, maxDA, minDA, color='grey', alpha=0.5)
    plt.xlabel('Number of Turns')
    plt.ylabel('DA')
    plt.title('DA vs. Number of Turns')
    legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
    plt.gca().add_artist(legend)
    plt.show()

    # repeat above plot but with log scale in the x axis only
    plt.plot(lostPartTurns, maxDA, color='blue')
    plt.plot(lostPartTurns, minDA, color='red')
    plt.plot(lostPartTurns, meanDA, color='black', linestyle='dotted')
    plt.fill_between(lostPartTurns, maxDA, minDA, color='grey', alpha=0.5)
    plt.xlabel('Number of Turns')
    plt.ylabel('DA')
    plt.xscale('log')
    plt.title('DA vs. Number of Turns')
    legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
    plt.gca().add_artist(legend)
    plt.show()



    # plt.subplot(4,2, 4)
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

    finalDAmin = minDA[-1]
    finalDAmax = maxDA[-1]
    finalDAmean = meanDA[-1]

    # plt.subplot(4,2,5)
    # create coloured scatter for number of turns vs. amplitude and angle
    plt.scatter(x, y, c=lostPartTurns, cmap='viridis', s=1)
    # plt.title(study_descriptor)
    plt.xlabel('Starting $x/\sigma_x$')
    plt.ylabel('Starting $y/\sigma_y$')
    cbar = plt.colorbar()
    cbar.set_label('Last turn')
    draw = plt.Circle((0, 0), finalDAmin, fill=False, color='red')
    plt.gcf().gca().add_artist(draw)
    draw = plt.Circle((0, 0), finalDAmax, fill=False, color='blue')
    plt.gcf().gca().add_artist(draw)
    draw = plt.Circle((0, 0), finalDAmean, fill=False, color='black')
    plt.gcf().gca().add_artist(draw)
    # make legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='DA min',
                            markerfacecolor='r', markersize=5),
                        plt.Line2D([0], [0], marker='o', color='w', label='DA max',
                            markerfacecolor='b', markersize=5),
                        plt.Line2D([0], [0], marker='o', color='w', label='DA mean',
                            markerfacecolor='k', markersize=5)]
    plt.legend(handles=legend_elements, loc='upper right')


    # make aspect ratio equal
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    surviveAtEnd = df_all_sim.loc[df_all_sim['state'] == 1]
    
    survive_r = surviveAtEnd['normalized amplitude in xy-plane'].to_numpy()
    survive_theta = surviveAtEnd['angle in xy-plane [deg]'].to_numpy()*np.pi/180
    survive_x = np.cos(survive_theta)*survive_r
    survive_y = np.sin(survive_theta)*survive_r
    survive_turns = surviveAtEnd['at_turn'].to_numpy()
    maxTurns = np.max(survive_turns)
    plt.scatter(survive_x, survive_y, s=1)
    plt.title(f"Survivors at end of {maxTurns} turns")
    plt.xlabel('x')
    plt.ylabel('y')
    # make aspect ratio equal
    plt.gca().set_aspect('equal', adjustable='box')
    draw = plt.Circle((0, 0), finalDAmin, fill=False, color='red')
    plt.gcf().gca().add_artist(draw)
    draw = plt.Circle((0, 0), finalDAmax, fill=False, color='blue')
    plt.gcf().gca().add_artist(draw)
    draw = plt.Circle((0, 0), finalDAmean, fill=False, color='black')
    plt.gcf().gca().add_artist(draw)
    # make legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='DA min',
                            markerfacecolor='r', markersize=5),
                        plt.Line2D([0], [0], marker='o', color='w', label='DA max',
                            markerfacecolor='b', markersize=5),
                        plt.Line2D([0], [0], marker='o', color='w', label='DA mean',
                            markerfacecolor='k', markersize=5)]
    plt.legend(handles=legend_elements, loc='upper right')
    

    plt.show()
    # plt.subplot(4,2,6)
    # plot number of turns against Q
    plt.plot(lostPartTurns, deltaQ, color='black', linestyle='dotted')
    plt.plot(lostPartTurns, np.exp(-meanDA**2/2), color='black')
    plt.plot(lostPartTurns, np.exp(-minDA**2/2), color='red')
    plt.plot(lostPartTurns, np.exp(-maxDA**2/2), color='blue')
    # plt.title(study_descriptor)
    plt.xlabel('Turn')
    plt.ylabel('Delta Q')
    legend = plt.legend(['Macroparticles', 'DA mean', 'DA min', 'DA max'], loc='lower left')
    plt.show()

    # plt.subplot(4,2,6)
    # plot number of turns against Q
    plt.loglog(lostPartTurns, deltaQ, color='black', linestyle='dotted')
    plt.loglog(lostPartTurns, np.exp(-meanDA**2/2), color='black')
    plt.loglog(lostPartTurns, np.exp(-minDA**2/2), color='red')
    plt.loglog(lostPartTurns, np.exp(-maxDA**2/2), color='blue')
    # plt.title(study_descriptor)
    plt.xlabel('Turn')
    plt.ylabel('Delta Q')
    legend = plt.legend(['Macroparticles', 'DA mean', 'DA min', 'DA max'], loc='lower left')
    plt.show()




    plt.plot(lostPartTurns, np.exp(-meanDA**2/2), color='black')




# fitDA.simplestFitPlot(lostPartTurns, meanDA, Nskip=0, topN = 1e6)
# importlib.reload(fitDA)
# fitDA.MultiFitPlot(lostPartTurns, meanDA, Nskip=20000, topN = 1e6,  plotEverything=True)
# fitDA.MultiFitPlot(lostPartTurns, meanDA, Nskip=20000, topN = 1e6)
# fitDA.MultiFitPlot(lostPartTurns, minDA, Nskip=20000, topN = 1e6)
# fitDA.MultiFitPlot(lostPartTurns, minDA, Nskip=20000, topN = 1e6,  logScale=True)
# # find the first turn at which any particles are lost at each amplitude
importlib.reload(fitDA)
fitDA.MultiFitPlot(lostPartTurns, meanDA, Nskip=20000, topN = 1e6,  plotEverything=True)
# firstTurnWithLosses = np.argmax(anyParticlesAmp, axis=0)

# DInf, b, kvalues, resids = fitDA.scanKfit(lostPartTurns, meanDA, kvalues=np.linspace(0.1, 2, 20), Nskip=0)
# # find the location of the first entry in each row of anyParticlesAmp that is False
# # this is the first turn at which all particles are lost at each amplitude
# for i in range(numUniqueAmps):
#     firstTurnWithAllLosses[i] = np.argmax(~anyParticlesAmp[i, :])

# firstTurnIndxWithAnyLosses =np.argmax(~allParticlesAmp, axis=0)
# firstTurnWithAnyLosses = lostPartTurns[firstTurnIndxWithAnyLosses]

# firstTurnIndxWithAllLosses =np.argmax(~anyParticlesAmp, axis=0)
# firstTurnWithAllLosses = lostPartTurns[firstTurnIndxWithAllLosses]

skipVals = np.linspace(0, 120000, 5000)
skipVals[0]=1
fitDA.scanNKfit(lostPartTurns, meanDA, Nskipvalues=skipVals, plot=True)

# firstTurnWithLosses = np.argmax(numParticlesTot > 0)

# # find the first turn at which all particles are lost at each amplitude
# firstTurnWithAllLosses = np.argmax(allParticlesAmp, axis=0)