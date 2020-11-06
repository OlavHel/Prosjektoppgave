import numpy as np
import pandas as pd

def sample_data(n_samples,data_size,rho):
    data = pd.DataFrame(columns=["SSx", "SSy", "SSxy"])

    mean = np.array([0,0])
    covar = np.array([
        [1,rho],
        [rho,1]
    ])
    for i in range(n_samples):
        data_points = np.random.multivariate_normal(mean,covar,data_size)
        data.loc[i] = [
            np.sum(data_points[:,0]**2),
            np.sum(data_points[:,1]**2),
            np.sum(data_points[:,0]*data_points[:,1])
        ]
    return data

def sample_data_multiple_rho(n_samples,data_size,rhos,total_data=None):
    if total_data is None:
        total_data = {}
    for rho in rhos:
        total_data[rho] = sample_data(n_samples=n_samples,data_size=data_size,rho=rho)
    return total_data

def store_data(data, filename):
    pd.to_pickle(data,"./SimData/"+name+".pkl")

def load_data(filename):
    return pd.read_pickle("./SimData/"+filename+".pkl")

n_samples = 10000
data_size = 10
rho = 0.5

total_data = sample_data_multiple_rho(n_samples=n_samples,data_size=data_size,rhos=[0.0,0.2,0.4,0.6,0.8])

name = "size"+str(data_size)+"points"+str(n_samples)
pd.to_pickle(total_data,"./SimData/"+name+".pkl")
#print(pd.read_pickle("./SimData/"+name+".pkl"))

