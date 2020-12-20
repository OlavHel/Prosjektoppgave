import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from results_storing import load_data
from finding_estimates import EstimatorClass

est = EstimatorClass()
cm = plt.get_cmap('gist_rainbow')
rhos = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


def get_legends(min_list):
    table = pd.DataFrame.from_dict(min_list)
    uniques = table.apply(pd.unique,axis=1)
    l_list = {}
    cols = {}
    for i,row in uniques.items():
        m = len(row)
        l_list[i] = [mpatches.Patch(color=cm(1.*j/m),label=uniques[i][j]) for j in range(m)]
        cols[i] = {row[j]:cm(1.*j/m) for j in range(m)}
    return l_list,cols

def get_data_per_eval(data):
    new_data = {name:pd.DataFrame(index=rhos,columns=data[0.0].index) for name in data[0.0].columns}
    for rho in rhos:
        for eval in data[rho].columns:
            new_data[eval].loc[rho] = data[rho][eval]
    return new_data

def plot_estimator(shifted_data,est_names):
    plt.figure()
    for i in range(len(est.eval_names)):
        plt.subplot(4, 1, i + 1)
        plt.title(est.eval_names[i])
        for name in est_names:
            plt.plot(rhos, shifted_data[est.eval_names[i]][name], label=name)
    plt.legend(loc="upper left")
    plt.show()

def plot_best_of_all(shifted_data,blocks,names):
    plt.figure()
    for i in range(len(est.eval_names)):
        plt.subplot(4,1,i+1)
        plt.title(est.eval_names[i])
        for j in range(len(blocks)):
            block = blocks[j]
            new_data = shifted_data[est.eval_names[i]].filter(items=block)
            plt.plot(new_data.min(axis=1),label=names[j])
        plt.legend()
    plt.show()



n = 10

data = load_data(n)
shifted_data = get_data_per_eval(data)
print(shifted_data["MSE"].transpose().to_latex(float_format="{:0.4f}".format))

plot_best_of_all(shifted_data,[
    ["uniformE","uniformM","uniformKL2","uniformFI2","uniformMAP"],
    ["arcsineE","arcsineM","arcsineKL2","arcsineFI2","arcsineMAP"],
    ["PC10-4E","PC10-4M","PC10-4KL2","PC10-4FI2","PC10-4MAP"],
    ["jeffreyE","jeffreyM","jeffreyKL2","jeffreyFI2","jeffreyMAP"],
    ["Sample corr","Sample corr, var=1","Sample corr, var=1, trunc","MLE"]
],
                 ["uniform","arcsine","PC10-4","jeffrey","frequentist"])
plot_estimator(shifted_data, ["uniformE","jeffreyM","MLE"])
plot_estimator(shifted_data, np.append(["jeffrey"+name for name in est.est_names],"MLE"))
plot_estimator(shifted_data, np.append(["arcsine"+name for name in est.est_names],"MLE"))
plot_estimator(shifted_data, ["uniform"+name for name in est.est_names])#np.append(["uniform"+name for name in est.est_names],"MLE"))
plot_estimator(shifted_data, np.append(["PC10-4"+name for name in est.est_names],"MLE"))
plot_estimator(shifted_data, ["jeffreyE","arcsineE","uniformE","PC10-4E"])
plot_estimator(shifted_data, ["jeffreyM","arcsineM","uniformM","PC10-4M","Sample corr, var=1, trunc"])
plot_estimator(shifted_data, ["jeffreyFI2","arcsineFI2","uniformFI2","PC10-4FI2"])
plot_estimator(shifted_data, ["jeffreyMAP","arcsineMAP","uniformMAP","PC10-4MAP"])
plot_estimator(shifted_data, est.freq_names)

print(data)
for rho in data:
    data[rho] = data[rho].apply(pd.to_numeric, axis=1)


min_list = {rho:data[rho].idxmin() for rho in rhos}
print(min_list)
plt.figure(1)
leg_hands, leg_cols = get_legends(min_list)
for i in range(len(est.eval_names)):
    col = est.eval_names[i]
    plt.subplot(4,1,i+1)
    plt.title(col)
    for rho in rhos:
        min_est = min_list[rho][col]
        plt.scatter([rho],[data[rho].loc[min_est][col]],color=leg_cols[col][min_est])
    plt.legend(handles=leg_hands[col])
plt.show()

plt.figure(2)
for i in range(len(est.eval_names)):
    plt.subplot(4,1,i+1)
    plt.title(est.eval_names[i])
    for j,row in shifted_data[est.eval_names[i]].iteritems():
        plt.plot(rhos,row,label=j)
    plt.legend()
plt.show()











