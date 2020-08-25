import os
import pandas as pd
from pandas import DataFrame
import numpy as np
# Where to save the figures and data files
DATA_ID = "Results/Task_d"

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)



from numpy import log2, zeros, mean, var, sum, loadtxt, arange, array, cumsum, dot, transpose, diagonal, sqrt
from numpy.linalg import inv

def block(x):
    # preliminaries
    n = len(x)
    d = int(log2(n))
    s, gamma = zeros(d), zeros(d)
    mu = mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, 36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

     # use magic to determine when we should have stopped blocking
    for k in arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    return mu, s[k]/2**(d-k)

def use_blocking_importance():
    frame = {}
    mean_list = []
    std_list = []
    var_list = []
    for i in [2, 10]:
        infile =  open(data_path("Blocking_Importance_Sampling"+str(i)+"_particles_3_dim.dat"),'r')
        infile.readline()
        data = np.loadtxt(infile)
        data = data[0:]
        (mean, var) = block(data[int(len(data)/4):])
        std = np.sqrt(var)
        mean_list.append(mean)
        std_list.append(std)
        var_list.append(var)
    frame['Particles'] = [2, 10]
    frame['Mean block'] = mean_list
    frame['Variance block'] = var_list
    frame['STD block'] = std_list
    frame = pd.DataFrame(frame)
    return frame

frameblocking = use_blocking_importance()
frame = pd.read_fwf(data_path("STD_Importance_Sampling.dat"))

print(frame)
print(frameblocking)
