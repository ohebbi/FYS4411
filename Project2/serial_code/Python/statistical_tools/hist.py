import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Where to save the figures and data files
DATA_ID = "Results/Task_g"

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

X0 = pd.read_fwf(data_path("IdealOnebody_Density.dat"))
X1 = pd.read_fwf(data_path("Onebody_Density.dat"))
X2 = pd.read_fwf(data_path("RepulsiveOnebody_Density.dat"))
X3 = pd.read_fwf(data_path("StrongRepulsiveOnebody_Density.dat"))
X4 = pd.read_fwf(data_path("StrongestRepulsiveOnebody_Density.dat"))


MCcycles = 2**20

X0["Counter"]/=MCcycles
X1["Counter"]/=MCcycles
X2["Counter"]/=MCcycles
X3["Counter"]/=MCcycles
X4["Counter"]/=MCcycles

counts = []
bins = []

y = np.zeros(len(X1["Counter"]))
r = 0.01
for i in range(len(X1["Counter"])):
    V = 4*(i*(i+1)+ 1/3)*np.pi*r**3
    y[i] = int(X1["Counter"][i]/V)

x = np.linspace(0, 2, 12)

import seaborn as sns
sns.set()
plt.plot(x, X0["Counter"])
plt.plot(x, X1["Counter"])
plt.plot(x, X2["Counter"])
plt.plot(x, X3["Counter"])
plt.plot(x, X4["Counter"])
#plt.legend(["No interaction", "a = 0.0043", "a = 0.043"])
plt.legend(["Ideal-Spherical", "No interaction", "a = 0.0043", "a = 0.043", "a = 0.43"])

plt.show()
#
