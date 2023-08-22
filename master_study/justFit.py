logN = np.log10(lostPartTurns)
DN = meanDA*logN

# linear fit DN*logN = D0_1*logN + DB_1
D0_1, DB_1 = np.polyfit(logN, DN, 1)
b_1 = DB_1/D0_1

# full fit to meanDA = D0*(1+ b/(logN)**k)
def fitFunc(logN, D0, b, k):
    return D0*(1 + b/(logN)**k)
popt, pcov = curve_fit(fitFunc, logN, meanDA, p0=[D0_1, b_1, 1])
D0, b, k = popt
D0_err, b_err, k_err = np.sqrt(np.diag(pcov))
print('D0 = ', D0, '+/-', D0_err)
print('b = ', b, '+/-', b_err)
print('k = ', k, '+/-', k_err)

# plot
plt.figure()
plt.plot(logN, meanDA, 'o', label='data')
plt.plot(logN, fitFunc(logN, D0, b, k), label='fit')
plt.xlabel('log(N)')
plt.ylabel('mean DA')
plt.legend()
plt.show()

