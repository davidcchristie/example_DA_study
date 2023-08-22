# study_name = "no_bb_higher_r"
# study_descriptor = "No Beam Beam"


# study_name = "tune_scan_more_thetas"
study_name = "example_tunescan"
study_descriptor = "BB On, I_OCT = 300"
doPLots = False

with open("112_postprocess_sort.py") as f:
    exec(f.read())

if doPLots:
    with open("121_fourplots.py") as f:
        exec(f.read())

anglesUnique = np.unique(df_all_sim['angle in xy-plane [deg]'])

#dataframe show the colummn names
# print(df_all_sim.columns)

# create a grid of NANs with dimensions: number of unique amplitudes x number of unique angles 
# (this is the number of particles that we expect to have at each amplitude and angle)
LastTurnAmpAngle = np.empty((len(amplitudesAll), len(anglesUnique)))
LastTurnAmpAngle[:] = np.nan

ampsOrdered = np.sort(amplitudesAll)
anglesOrdered = np.sort(anglesUnique)


# iterate over all particles in dataframe lost_particles
for particleNo in range(1,numLostParts):
    # print(particleNo) 
    lastTurn = lostPartTurns[particleNo-1]
    lastAmp = lostPartAmps[particleNo-1]
    lastAngle = lostPartAngles[particleNo-1]
    # find the index of the amplitude in the list of amplitudes
    ampIndx = np.where(ampsOrdered == lastAmp)[0][0]
    angleIndx = np.where(anglesOrdered == lastAngle)[0][0]
    LastTurnAmpAngle[ampIndx, angleIndx] = lastTurn


# convert ampsordered and anglesordered to a row vector
r = ampsOrdered.reshape(len(ampsOrdered),1)
theta = anglesOrdered.reshape(1,len(anglesOrdered))*np.pi/180

x = np.cos(theta)*r
y = np.sin(theta)*r


# create colorbar z
plt.scatter(x, y, c=LastTurnAmpAngle, cmap='viridis', s=1)
plt.title(study_descriptor)
plt.xlabel('Starting $x/\sigma_x$')
plt.ylabel('Starting $y/\sigma_y$')
cbar = plt.colorbar()
cbar.set_label('Last turn')
plt.show()


# max value of LastTurnAmpAngle ignoring nans, dimension 1
maxLastTurnAmpAngle = np.nanmax(LastTurnAmpAngle, axis=1)
# min value of LastTurnAmpAngle ignoring nans, dimension 1
minLastTurnAmpAngle = np.nanmin(LastTurnAmpAngle, axis=1)
# mean value of LastTurnAmpAngle ignoring nans, dimension 1
meanLastTurnAmpAngle = np.nanmean(LastTurnAmpAngle, axis=1)
# std value of LastTurnAmpAngle ignoring nans, dimension 1
stdLastTurnAmpAngle = np.nanstd(LastTurnAmpAngle, axis=1)

# plot with shaded area between maxLastTurnAmpAngle and minLastTurnAmpAngle and dotted line at meanLastTurnAmpAngle

# plt.plot(ampsOrdered, maxLastTurnAmpAngle, color='grey')
# plt.plot(ampsOrdered, minLastTurnAmpAngle, color='grey')
# plt.plot(ampsOrdered, meanLastTurnAmpAngle, color='black', linestyle='dotted')
# plt.fill_between(ampsOrdered, maxLastTurnAmpAngle, minLastTurnAmpAngle, color='grey', alpha=0.5)
# plt.title(study_descriptor)
# plt.xlabel('Starting $r$')
# plt.ylabel('Last turn')
# plt.show()
 

 # DA using surviveParticlesAmpAngle
anyParticlesAmpAngle = np.sum(surviveParticlesAmpAngle, axis=2)

# create MaxMinMeanDA as grid of zeros with dimensions length(turnsAllUnique) x 3
MaxMinMeanDA = np.zeros((len(turnsAllUnique), 3))


DAperThetas = np.zeros((len(turnsAllUnique), len(anglesUnique)))



[NN, RR, TT] = np.meshgrid(turnsAllUnique.astype(float), amplitudesAll.astype(float), anglesUnique.astype(float), indexing='ij')
RRLost = RR
RRLost[surviveParticlesAmpAngle == 1] = np.nan

for turn in range(0,len(turnsAllUnique)):
    ampSlice = RRLost[turn,:,:] 
    DAperThetas[turn,:] = np.nanmin(ampSlice, 0) 

minDA = np.nanmin(DAperThetas, 1)
maxDA = np.nanmax(DAperThetas, 1)
meanDA = np.nanmean(DAperThetas, 1)

# plot turnsAllUnique vs. maxDA, minDA, meanDA 
plt.plot(turnsAllUnique, maxDA, color='blue')
plt.plot(turnsAllUnique, minDA, color='red')
plt.plot(turnsAllUnique, meanDA, color='black', linestyle='dotted')
plt.fill_between(turnsAllUnique, maxDA, minDA, color='grey', alpha=0.5)
plt.title(study_descriptor)
plt.xlabel('Turn')
plt.ylabel('DA')
legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
plt.gca().add_artist(legend)


plt.show()


# alternative DA per turn using DAbyNandTheta
minDA2 = np.nanmin(DAbyNandTheta, 1)
maxDA2 = np.nanmax(DAbyNandTheta, 1)
meanDA2 = np.nanmean(DAbyNandTheta, 1)

# plot turnsAllUnique vs. maxDA, minDA, meanDA 
plt.plot(turnsAllUnique, maxDA2, color='blue')
plt.plot(turnsAllUnique, minDA2, color='red')
plt.plot(turnsAllUnique, meanDA2, color='black', linestyle='dotted')
plt.fill_between(turnsAllUnique, maxDA2, minDA2, color='grey', alpha=0.5)
plt.title(study_descriptor)
plt.xlabel('Turn')
plt.ylabel('DA')
legend = plt.legend(['Max DA', 'Min DA', 'Mean DA'], loc='upper right')
plt.gca().add_artist(legend)


plt.show()


# intensity as a function of turn based on mean DA
intensityChange = - np.exp(-meanDA2**2/2)

# plot number of turns against Q
plt.plot(turnsAllUnique, deltaQ, color='blue')
plt.plot(turnsAllUnique, intensityChange, color='red')
plt.plot(turnsAllUnique, - np.exp(-minDA2**2/2), color='black')
plt.title(study_descriptor)
plt.xlabel('Turn')
plt.ylabel('Delta Q')
legend = plt.legend(['Macroparticles', 'DA mean', 'DA min'], loc='lower left')
plt.show()


# same plot as log-log 
plt.loglog(turnsAllUnique, -deltaQ, color='blue')
plt.loglog(turnsAllUnique, -intensityChange, color='red')
plt.loglog(turnsAllUnique,  np.exp(-minDA2**2/2), color='black')
plt.title(study_descriptor) 
plt.xlabel('Turn')
plt.ylabel('Delta Q')

legend = plt.legend(['Macroparticles', 'DA mean', 'DA min'], loc='lower left')
plt.show()
