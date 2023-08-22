import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def simplestFit(turns, DA, Nskip=0):
    # Simplest fit: assume D(N) = D_Inf (1 + b/log10(N))
    # i.e. [D(N)log N] = D_Inf [log N] + b

#    Nskip: number of turns to skip at the beginning   
    DAUsed = DA[turns>=Nskip]
    turnsUsed = turns[turns>=Nskip]

    logN = np.log10(turnsUsed)
    DNlogN = DAUsed*logN
        # linear fit DN*logN = D0_1*logN + DB_1
    DInf, DinfB = np.polyfit(logN, DNlogN, 1)
    b = DinfB/DInf

    return DInf, b

def simplestFitGivenK(turns, DA, k=1, Nskip=0, plot=False):
    # Simplest fit: assume D(N) = D_Inf (1 + b/log10(N)**k)
    # i.e. [D(N)(log N)**k] = D_Inf [(log N)**k] + b
    # Note: k is given
    # Nskip: number of turns to skip at the beginning
    DAUsed = DA[turns>=Nskip]
    turnsUsed = turns[turns>=Nskip]
    logNk = np.log10(turnsUsed)**k
    DNlogNk = DAUsed*logNk
        # linear fit DN*logNk = D0_1*logN + DB_1
    DInf, DinfB = np.polyfit(logNk, DNlogNk, 1)
    b = DinfB/DInf
    if plot:
        plt.plot(logNk, DNlogNk)
        plt.plot(logNk, DInf*logNk + DinfB)

        plt.xlabel('log10(N)^k')
        plt.ylabel('D(N)log10(N)^k')
        plt.show()
    return DInf, b

def scanKfit(turns, DA, kvalues=np.linspace(0.1, 3, 25), Nskip=0):
    # Scan k values and fit D_Inf and b
    # Return D_Inf and b for each k
    # Nskip: number of turns to skip at the beginning
    DA = DA[turns>=Nskip]
    turns = turns[turns>=Nskip]

    DInf = np.zeros(len(kvalues))
    b = np.zeros(len(kvalues))
    resids = np.zeros(len(kvalues))
    for i in range(len(kvalues)):
        DInf[i], b[i] = simplestFitGivenK(turns, DA, kvalues[i], Nskip)
        DAfitted = fullLog(turns, DInf[i], b[i], kvalues[i])
        resids[i] = np.sqrt(np.sum((DA - DAfitted)**2)/(len(DA)-2))
    return DInf, b, kvalues, resids
    


def logFitScannedK(turns, DA, kvalues=np.linspace(0.1, 3, 25), Nskip=0, plot=False):
    DInf, b, kvalues, resids = scanKfit(turns, DA, kvalues, Nskip)
    bestK = kvalues[np.argmin(resids)]
    DInf, b = simplestFitGivenK(turns, DA, bestK, Nskip, plot=plot)
    if plot:
        plt.plot(kvalues, resids)
        plt.plot(bestK, np.min(resids), 'o')
        plt.xlabel('k')
        plt.ylabel('RMSE')
        plt.show()
    return DInf, b, bestK

def scanNKfit(turns, DA, kvalues=np.linspace(0.1, 3, 16), Nskipvalues=[1, 10000, 20000, 50000, 100000], plot=False):
    # for each Nskip, do logFIgScannedK and return DInf, b, k, Nskip, RMSE
    DInf = np.zeros(len(Nskipvalues))
    b = np.zeros(len(Nskipvalues))
    k = np.zeros(len(Nskipvalues))
    resids = np.zeros(len(Nskipvalues))
    for i in range(len(Nskipvalues)):
        turnsUsed = turns[turns>=Nskipvalues[i]]
        DAUsed = DA[turns>=Nskipvalues[i]]
        DInf[i], b[i], k[i] = logFitScannedK(turnsUsed, DAUsed, kvalues, Nskip=0)
        DAfitted = fullLog(turnsUsed, DInf[i], b[i], k[i])
        resids[i] = np.sqrt(np.sum((DAUsed - DAfitted)**2)/(len(DAUsed)-2))
    if plot:
        # plot nskip vs k (right axis) and nskip vs RMSE (left axis)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(Nskipvalues, k, 'b-')
        ax2.plot(Nskipvalues, resids, 'r-')
        ax1.set_xlabel('Nskip')
        ax1.set_ylabel('k', color='b')
        ax2.set_ylabel('RMSE', color='r')
        plt.show()
    return DInf, b, k, Nskipvalues, resids
    
def scanNKfitOld(turns, DA, kvalues=np.linspace(0.1, 3, 16), Nskipvalues=[1, 10000, 20000, 50000, 100000], plot=False):
    # for each Nskip, do logFIgScannedK and return DInf, b, k, Nskip, RMSE
    DInf = np.zeros(len(Nskipvalues))
    b = np.zeros(len(Nskipvalues))
    k = np.zeros(len(Nskipvalues))
    resids = np.zeros(len(Nskipvalues))
    for i in range(len(Nskipvalues)):
        DInf[i], b[i], k[i] = logFitScannedK(turns, DA, kvalues, Nskipvalues[i])
        DAfitted = fullLog(turns, DInf[i], b[i], k[i])
        resids[i] = np.sqrt(np.sum((DA - DAfitted)**2)/(len(DA)-2))
    if plot:
        # plot nskip vs k (right axis) and nskip vs RMSE (left axis)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(Nskipvalues, k, 'b-')
        ax2.plot(Nskipvalues, resids, 'r-')
        ax1.set_xlabel('Nskip')
        ax1.set_ylabel('k', color='b')
        ax2.set_ylabel('RMSE', color='r')
        plt.show()
    return DInf, b, k, Nskipvalues, resids
#                 , printAll=False):)

# def scanKNfit(turns, DA, kvalues=np.linspace(0.1, 3, 16), Nskipvalues=[1, 10000, 20000, 50000, 100000]
#                 , printAll=False):
#      # Scan k values and fit D_Inf and b
#      # Return D_Inf and b for each k
#      # Nskip: number of turns to skip at the beginning
#      DInf = np.zeros((len(kvalues), len(Nskipvalues)))
#      resids = np.zeros((len(kvalues), len(Nskipvalues)))
#      b = np.zeros((len(kvalues), len(Nskipvalues)))
#      for i in range(len(kvalues)):
#           for j in range(len(Nskipvalues)):
#                 DInf[i,j], b[i,j] = simplestFitGivenK(turns, DA, kvalues[i], Nskipvalues[j])
#                 DAfitted = fullLog(turns, DInf[i,j], b[i,j], kvalues[i])
#                 resids[i,j] = np.sqrt(np.sum((DA - DAfitted)**2)/(len(DA)-2))
#      if printAll:
#         print('k = ', kvalues)
#         print('Nskip = ', Nskipvalues)
#         print('DInf = ', DInf)
#         print('b = ', b)
#         print('RMSE = ', resids)
#      return DInf, b, kvalues, Nskipvalues, resids


def fullFit(turns, DA, Nskip=0):
    initialDInf, initialB = simplestFit(turns, DA, Nskip)
    params, covariance = curve_fit(fullLog, turns, DA, p0=[initialDInf, initialB, 1])
    DInf_optimized, b_optimized, k_optimized = params
    return DInf_optimized, b_optimized, k_optimized

def fullLog(turns, DInf, b, k):
    logN = np.log10(turns)
    DA = DInf*(1 + b/(logN)**k)
    return DA

def simpleLog(turns, DInf, b):
    logN = np.log10(turns)
    DA = DInf*(1 + b/logN)
    return DA

def simplestParametric(DInf, b, topN = 1e6):
    turns = np.linspace(1, topN, 500)
    DA = simpleLog(turns, DInf, b)
    return turns, DA

def fullParametric(DInf, b, k, topN = 1e6):
    turns = np.linspace(1, topN, 500)
    logN = np.log10(turns)
    DA = DInf*(1 + b/(logN)**k)
    return turns, DA

def MultiFitPlot(turns, DA, Nskip=10000, topN = 1e6, plotEverything = False):
    DInf, b = simplestFit(turns, DA, Nskip)
    turnsFit, DAFit = simplestParametric(DInf, b, topN)
    # DInfF, bF, kF = fullFit(turns, DA, Nskip)
    DInfF2, bF2, kF2 = logFitScannedK(turns, DA, Nskip=Nskip, plot = plotEverything)
    # turnsFitF, DAFitF = fullParametric(DInfF, bF, kF, topN)
    turnsFitF2, DAFitF2 = fullParametric(DInfF2, bF2, kF2, topN)
    logScale = False
    for plotNo in range(0,2):
        plt.plot(turnsFit, DAFit, label='Simple fit: D_Inf = %.2f, b = %.2f' % (DInf, b))
        # plt.plot(turnsFitF, DAFitF, label='full fit')
        plt.plot(turnsFitF2, DAFitF2, label='Full fit: D_Inf = %.2f, b = %.2f, k = %.2f' % (DInfF2, bF2, kF2))
        plt.plot(turns, DA, label='Data')
        plt.xlabel('N')
        plt.ylabel('DA')
        if logScale:
            plt.xscale('log')
        plt.legend()
        plt.show()
        logScale = True


def simplestFitPlot(turns, DA, Nskip=10000, topN = 1e6, logScale=False):
    DInf, b = simplestFit(turns, DA, Nskip)
    turnsFit, DAFit = simplestParametric(DInf, b, topN)
    plt.plot(turnsFit, DAFit, label='fit')
    plt.plot(turns, DA, label='data')
    plt.xlabel('N')
    plt.ylabel('DA')
    if logScale:
        plt.xscale('log')
    plt.legend()
    plt.show()


