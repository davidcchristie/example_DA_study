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

# grid x, y, last turn (not sure if this is necessary for anything!)
for particleNo in range(numParticlesStart):
    thisX = df_downsampled_sim["x"][particleNo]
    thisY = df_downsampled_sim["y"][particleNo]
    thisTurn = df_downsampled_sim["at_turn"][particleNo]
    thisXIndx = np.where(XlistUse == thisX)[0][0]
    thisYIndx = np.where(YlistUse == thisY)[0][0]
    lastTurn[thisXIndx, thisYIndx] = thisTurn



surviveAtEnd = df_all_sim.loc[df_all_sim['state'] == 1]
amplitudeSurvivors = surviveAtEndsurviveAtEnd = df_all_sim.loc[df_all_sim['state'] == 1]
amplitudeSurvivors = surviveAtEnd["normalized amplitude in xy-plane"].to_numpy()
maxAmpSurvived = np.max(amplitudeSurvivors)

# # create a grid of ones with dimensions: length(turnsAllUnique) x length(amplitudesAll) x length(anglesUnique)
# surviveParticlesAmpAngle = np.ones((numTurns, numUniqueAmps, numAngles))


# # # macroparricle charge elements dq: different for each r
# # r_dr = np.multiply( amplitudeList[:-1], np.diff(amplitudeList))
# # dtheta = 2*np.pi/numAngles  # assuming it's equal, fix otherwise!
# # dq =  dtheta * np.multiply(np.append(r_dr, 1) , np.exp(-amplitudeList**2/2))
# # # track total charge lost 
# # dqPerTurn = np.zeros(numTurns)

# # track change in DA separately for each angular bin
# DAbyNandTheta = np.full((numTurns, numAngles), np.nan)

# # when is each particle lost?
minLostByAngle = np.full((numAngles), np.nan)
runningMinLostAmp = np.full((numLostParts), np.nan)
runningMaxExistingAmp = np.zeros(numLostParts)
angularMeanDA = np.zeros(numLostParts)
angularMedianDA = np.zeros(numLostParts)
angularMaxDA = np.zeros(numLostParts)

# # DAbyNandTheta gives dynamic aperture for each angular bin and each turn
# # surviveParticlesAmpAngle shows particle survival per turn for each amplitude and each angle
# # iterate over all particles in dataframe lost_particles
for particleNo in range(1,numLostParts+1):
    
    lastTurn = lostPartTurns[particleNo-1]
    lastAmp = lostPartAmps[particleNo-1]
    lastAngle = lostPartAngles[particleNo-1]
   # find the index of the angle in angleList nearest to lastAngle
    angleIndx = np.argmin(np.abs(angleBinList - lastAngle))
    minLostByAngle[angleIndx] = np.nanmin([lastAmp, minLostByAngle[angleIndx]])
    runningMinLostAmp[particleNo-1] = np.nanmin([lastAmp, runningMinLostAmp[particleNo-2]])
    runningMaxExistingAmp[particleNo-1] = np.max(allR[allTurns >= lastTurn])
    angularMeanDA[particleNo-1] = np.nanmean(minLostByAngle)
    angularMedianDA[particleNo-1] = np.nanmedian(minLostByAngle)
    angularMaxDA[particleNo-1] = np.nanmax(minLostByAngle)

finalDAmin = runningMinLostAmp[-1]
finalDAmax = angularMaxDA[-1]
finalDAmean = angularMeanDA[-1]

# # plot runningMinLostAmp, runningMaxExistingAmp, angularMeanDA, angularMedianDA vs. particleNo
# plt.plot(lostPartTurns, runningMinLostAmp, color='black')
# # plt.plot(lostPartTurns, runningMaxExistingAmp, color='red')
# plt.plot(lostPartTurns, angularMeanDA, color='blue')
# plt.plot(lostPartTurns, angularMedianDA, color='green')
# plt.xlabel('Turns')
# plt.ylabel('Amplitude')
# plt.title('Amplitude vs. Particle No.')
# legend = plt.legend(['Min. Lost', 'Mean DA', 'Median DA'], loc='upper right')
# plt.gca().add_artist(legend)
# plt.show()


# repeat above plot but with log scale in the x axis only
for plotno in range(1,3):
    plt.plot(lostPartTurns, angularMaxDA, color='blue')
    plt.plot(lostPartTurns, runningMinLostAmp, color='red')
    plt.plot(lostPartTurns, angularMeanDA, color='black', linestyle='dotted')
    plt.plot(lostPartTurns, angularMedianDA, color='black', linestyle='dashdot')
    plt.fill_between(lostPartTurns, angularMaxDA, runningMinLostAmp, color='grey', alpha=0.5)

    plt.xlabel('Number of Turns')
    plt.ylabel('DA')
    if plotno == 2:
        plt.xscale('log')
    plt.title('DA vs. Number of Turns')
    legend = plt.legend(['Max DA', 'Min DA', 'Mean DA', 'Median DA'], loc='upper right')
    plt.gca().add_artist(legend)
    plt.show()



plt.scatter(surviveAtEnd["x"], surviveAtEnd["y"], s=1)
plt.title(f"Survivors at end of {np.max(allTurns)} turns")
plt.xlabel('x')
plt.ylabel('y')
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


# scatter polot lostPartX and lostPartY with colour being lostPartTurns
plt.scatter(lostPartX, lostPartY, c=lostPartTurns, cmap='viridis', s=1)
plt.title(f"Lost Particles at end of {np.max(allTurns)} turns")
plt.xlabel('x')
plt.ylabel('y')
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
# make colour bar title "last turn"

cbar = plt.colorbar()
cbar.set_label('Last turn')
plt.show()

# fitDA.simplestFitPlot(lostPartTurns, meanDA, Nskip=0, topN = 1e6)
importlib.reload(fitDA)

fitDA.MultiFitPlot(lostPartTurns, angularMeanDA, Nskip=20000, topN = 1e6,  plotEverything=True)
skipVals = np.linspace(0, 120000, 5000)
skipVals[0]=1
fitDA.scanNKfit(lostPartTurns, angularmeanDA, Nskipvalues=skipVals, plot=True)


#     surviveParticlesAmpAngle[lastTurnIndx:, ampIndx, angleIndx] = 0
#     lastTurnRTheta[ampIndx, angleIndx] = lastTurn

#     # how much charge is lost at this turn?
#     # dqPerTurn[particleNo-1] += dq[ampIndx] # += just in case there's more than one lost per turn!
    
#     #  DA for that angle and all subsequent turns (including the turn of loss)
#     #  is the minimum of the current DA and the DA of the lost particle
#     #  if another particle at that angle and lower amplitude is lost later, the DA will be updated again
#     DAbyNandTheta[lastTurnIndx:, angleIndx] = np.nanmin([lastAmp, DAbyNandTheta[lastTurnIndx, angleIndx]])

# # number of particles surviving at each turn for each amplitude 
# numParticlesTurnsAmp = np.sum(surviveParticlesAmpAngle, axis=2)
# particleUnscathed = np.all(surviveParticlesAmpAngle>0, axis=0) # survives all turns

# # number of particles surviving at each turn 
# numParticlesTot = np.sum(numParticlesTurnsAmp, axis=1)

# numMegaturnsSurvived = np.sum(surviveParticlesAmpAngle, axis=0)
# numTurnsSurvived = lostPartTurns[numMegaturnsSurvived.astype(int)-1]

# # # do any particles survive at each amplitude?
# numParticlesAmp = np.sum(surviveParticlesAmpAngle, axis=2)
# lastTurnAllPresent = np.full((numUniqueAmps), np.nan)
# firstTurnNoParticles = np.full((numUniqueAmps), np.nan)
# firstTurnHalfGone = np.full((numUniqueAmps), np.nan)
# numPartsHalfGone = round(numAngles/2)


# # convert to Cartesians for colour plot
# x = np.cos(lostPartAngles*np.pi/180)*lostPartAmps
# y = np.sin(lostPartAngles*np.pi/180)*lostPartAmps






# for ampIndx in range(numUniqueAmps):
#     popByTurn = numParticlesAmp[:, ampIndx]

#     # turnsWithSomeParticles  = lostPartTurns[popByTurn > 0]
#     turnsWithAllParticles = lostPartTurns[popByTurn == numAngles]
#     turnsWithNoParticles = lostPartTurns[popByTurn == 0]
#     turnsHalfGone = lostPartTurns[popByTurn <= numPartsHalfGone]

#     if len(turnsWithAllParticles) > 0:
#         lastTurnAllPresent[ampIndx] = turnsWithAllParticles[-1]

#     if len(turnsWithNoParticles) > 0:
#         firstTurnNoParticles[ampIndx] = turnsWithNoParticles[0]   

#     if len(turnsHalfGone) > 0:
#         firstTurnHalfGone[ampIndx] = turnsHalfGone[0]   



# minDA = np.nanmin(DAbyNandTheta, 1)
# maxDA = np.nanmax(DAbyNandTheta, 1)
# meanDA = np.nanmean(DAbyNandTheta, 1)
# deltaQ = np.cumsum(dqPerTurn)
# #append leading zero to deltaQ


# # def DFitted (N, Dinf, b, kappa):
# #     return Dinf*(1+b/((np.log10(N))**kappa))

# # # fit a power law to the DA vs. number of turns
# # params, cov = curve_fit(DFitted, lostPartTurns, meanDA, p0=[24, 0.4, 1.4])
# # D0_fit, b_fit, k_fit = params


# if drawFourPlots:


#     # plt.subplot(4,2, 1)
#     plt.plot(lostPartTurns, numParticlesTot)
#     plt.xlabel('Number of Turns')
#     plt.ylabel('Number of Survivors')
#     plt.title('Population at each turn')
#     plt.show()
#     # plt.subplot(4,2, 2)
#     # plot firstTurnWithLosses vs. amplitude and firstTurnWithAllLosses vs. amplitude
#     plt.plot(amplitudeList, lastTurnAllPresent, label='All present (end)')
#     plt.plot(amplitudeList, firstTurnHalfGone, label='Half gone (start)')
#     plt.plot(amplitudeList, firstTurnNoParticles, label='All lost (start)')
#     plt.xlabel('Amplitude (\sigma)')
#     plt.ylabel('Number of Turns')
#     plt.title('When are Particles Lost?')
#     plt.legend()
#     plt.show()
#     # plt.subplot(4,2, 3)
#     # plot maxDA, minDA, meanDA vs. number of turns and shade in between
#     plt.plot(lostPartTurns, maxDA, color='blue')
#     plt.plot(lostPartTurns, minDA, color='red')
#     plt.plot(lostPartTurns, meanDA, color='black', linestyle='dotted')
#     plt.fill_between(lostPartTurns, maxDA, minDA, color='grey', alpha=0.5)
#     plt.xlabel('Number of Turns')
#     plt.ylabel('DA')
#     plt.title('DA vs. Number of Turns')
#     legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
#     plt.gca().add_artist(legend)
#     plt.show()

#     # repeat above plot but with log scale in the x axis only
#     plt.plot(lostPartTurns, maxDA, color='blue')
#     plt.plot(lostPartTurns, minDA, color='red')
#     plt.plot(lostPartTurns, meanDA, color='black', linestyle='dotted')
#     plt.fill_between(lostPartTurns, maxDA, minDA, color='grey', alpha=0.5)
#     plt.xlabel('Number of Turns')
#     plt.ylabel('DA')
#     plt.xscale('log')
#     plt.title('DA vs. Number of Turns')
#     legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
#     plt.gca().add_artist(legend)
#     plt.show()


#     # plt.subplot(4,2, 4)
#     # generate pcolor plot of numParticlesAmp against number of turns and amplitude
#     TURNS_ALL, AMPS_ALL = np.meshgrid( lostPartTurns, amplitudeList)
#     P = plt.pcolor(TURNS_ALL, AMPS_ALL, numParticlesTurnsAmp.transpose(), shading='auto')
#     plt.xlabel('Number of Turns')
#     plt.ylabel('Amplitude ($\sigma$)')
#     C = plt.colorbar(P)
#     C.set_label("Num. Surviving Particles")
#     plt.title('Particle Survival')
#     plt.savefig(f"scans/{study_name}/particleSurvival.png")
#     plt.show()

#     # plt.subplot(4,2,5)
#     # create coloured scatter for number of turns vs. amplitude and angle
#     plt.scatter(x, y, c=lostPartTurns, cmap='viridis', s=1)
#     # plt.title(study_descriptor)
#     plt.xlabel('Starting $x/\sigma_x$')
#     plt.ylabel('Starting $y/\sigma_y$')
#     cbar = plt.colorbar()
#     cbar.set_label('Last turn')
#     # make aspect ratio equal
#     # plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

#     # plt.subplot(4,2,6)
#     # plot number of turns against Q
#     plt.plot(lostPartTurns, deltaQ, color='black', linestyle='dotted')
#     plt.plot(lostPartTurns, np.exp(-meanDA**2/2), color='black')
#     plt.plot(lostPartTurns, np.exp(-minDA**2/2), color='red')
#     plt.plot(lostPartTurns, np.exp(-maxDA**2/2), color='blue')
#     # plt.title(study_descriptor)
#     plt.xlabel('Turn')
#     plt.ylabel('Delta Q')
#     legend = plt.legend(['Macroparticles', 'DA mean', 'DA min', 'DA max'], loc='lower left')
#     plt.show()

#     # plt.subplot(4,2,6)
#     # plot number of turns against Q
#     plt.loglog(lostPartTurns, deltaQ, color='black', linestyle='dotted')
#     plt.loglog(lostPartTurns, np.exp(-meanDA**2/2), color='black')
#     plt.loglog(lostPartTurns, np.exp(-minDA**2/2), color='red')
#     plt.loglog(lostPartTurns, np.exp(-maxDA**2/2), color='blue')
#     # plt.title(study_descriptor)
#     plt.xlabel('Turn')
#     plt.ylabel('Delta Q')
#     legend = plt.legend(['Macroparticles', 'DA mean', 'DA min', 'DA max'], loc='lower left')
#     plt.show()











# # # find the first turn at which any particles are lost at each amplitude
# # firstTurnWithLosses = np.argmax(anyParticlesAmp, axis=0)

# # # find the location of the first entry in each row of anyParticlesAmp that is False
# # # this is the first turn at which all particles are lost at each amplitude
# # for i in range(numUniqueAmps):
# #     firstTurnWithAllLosses[i] = np.argmax(~anyParticlesAmp[i, :])

# # firstTurnIndxWithAnyLosses =np.argmax(~allParticlesAmp, axis=0)
# # firstTurnWithAnyLosses = lostPartTurns[firstTurnIndxWithAnyLosses]

# # firstTurnIndxWithAllLosses =np.argmax(~anyParticlesAmp, axis=0)
# # firstTurnWithAllLosses = lostPartTurns[firstTurnIndxWithAllLosses]



# # firstTurnWithLosses = np.argmax(numParticlesTot > 0)

# # # find the first turn at which all particles are lost at each amplitude
# # firstTurnWithAllLosses = np.argmax(allParticlesAmp, axis=0)