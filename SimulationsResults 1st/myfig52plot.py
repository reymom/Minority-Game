import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append("/home/raymun/Documentos/Minority_Game/Minority-Game")

alpha201 = []
var201 = []
S = 2
N201 = 201
numponds = 50
file_name = "VARvsALPHA_" + str(N201) + "N_" + str(numponds) + "numponds"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha201.append(float(value[0]))
        var201.append(float(value[1]))

alpha401 = []
var401 = []
S = 2
N401 = 401
numponds = 50
file_name = "VARvsALPHA_" + str(N401) + "N_" + str(numponds) + "numponds"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha401.append(float(value[0]))
        var401.append(float(value[1]))

alpha601 = []
var601 = []
S = 2
N601 = 601
numponds = 50
file_name = "VARvsALPHA_" + str(N601) + "N_" + str(numponds) + "numponds"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha601.append(float(value[0]))
        var601.append(float(value[1]))

alpha801 = []
var801 = []
S = 2
N801 = 801
numponds = 25
file_name = "VARvsALPHA_" + str(N801) + "N_" + str(numponds) + "numponds"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha801.append(float(value[0]))
        var801.append(float(value[1]))

alpha501 = []
var501 = []
S = 2
N501 = 501
numponds = 100
file_name = "VARvsALPHA_" + str(N501) + "N_" + str(numponds) + "numponds"
with open('Data/{}.dat'.format(file_name), 'r') as file:
    for line in file:
        value = line.split()
        value[1].split('\n')
        alpha501.append(float(value[0]))
        var501.append(float(value[1]))

fig, ax = plt.subplots()
ax.set(xscale='log', yscale='log', xlabel=r'$\alpha$', ylabel=r'$\sigma^2/N$')
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('{{:.{:1d}f}}'.format(
        int(np.maximum(-np.log10(x), 0)))).format(x)))
ax.tick_params(which='both',
               labelbottom=True,
               labeltop=False,
               labelleft=True,
               labelright=False,
               bottom=True,
               top=True,
               left=True,
               right=True)

ax.plot(alpha201,
        var201,
        marker='o',
        ms=6,
        mec='r',
        mfc='w',
        ls='',
        label=r'$N = 201$')
ax.plot(alpha401,
        var401,
        marker='^',
        ms=6,
        mec='b',
        mfc='w',
        ls='',
        label=r'$N = 401$')
ax.plot(alpha501,
        var501,
        marker='*',
        ms=5,
        mec='yellow',
        mfc='w',
        ls='',
        label=r'$N = 501$')
ax.plot(alpha601,
        var601,
        marker='s',
        ms=6,
        mec='lime',
        mfc='w',
        ls='',
        label=r'$N = 601$')
ax.plot(alpha801,
        var801,
        marker='P',
        ms=5,
        mec='m',
        mfc='w',
        ls='',
        label=r'$N = 801$')

ax.axhline(y=1, xmin=0, xmax=1, color='k', alpha=0.7, linestyle='--')
ax.legend(loc=0)
plt.show()
