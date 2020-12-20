import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from finding_estimates import EstimatorClass


n_samples = 100000
n = 20
estimator_class = EstimatorClass()
rhos = np.linspace(0, 0.9, 10)
print(rhos)

final_evaluates = {}

for rho in rhos:
    print("rho", rho)
    estimates = np.empty((n_samples, estimator_class.n_freq))
    evaluations = np.empty((n_samples, estimator_class.n_evaluators, estimator_class.n_freq))
    for i in range(n_samples):
        if i % 10000 == 0:
            print(i)

        estimators, names = estimator_class.get_all_freq_estimators(n,rho)
        evaluators = estimator_class.fully_evaluate_estimate(rho, estimators)
        estimates[i, :] = estimators
        evaluations[i, :] = evaluators

    final_evaluates[rho] = (np.mean(evaluators, axis=0), np.mean(evaluations, axis=0))
    mean_evaluations = np.mean(evaluations, axis=0)
#    eval_table = np.hstack((np.transpose([estimator_class.freq_names]), mean_evaluations))
#    eval_table = np.vstack((np.hstack(([[0]], [estimator_class.eval_names])), eval_table))
# print(eval_table)

#    plt.figure(1)
#    for i in range(len(estimates[0])):
#        plt.plot(estimates[:,i],"o",label=estimator_class.est_names[i])
#    plt.legend()
#    plt.show()

#    plt.figure(2)
#    for i in range(estimator_class.n_estimators):
#        plt.plot(estimator_class.eval_names,mean_evaluations[i,:],label=estimator_class.est_names[i])
#    plt.legend()
#    plt.show()

print(final_evaluates)

import results_storing

estimator_names = estimator_class.freq_names

for rho in final_evaluates:
#        results_storing.save_data(n,np.transpose(final_evaluates[rho][1]),estimator_class.eval_names,["PC10-4MAP"],np.round(rho,2))
    results_storing.save_data(n,np.transpose(final_evaluates[rho][1]),estimator_class.eval_names,estimator_names,np.round(rho,2))

plt.figure()
for i in range(estimator_class.n_evaluators):
    plt.subplot(estimator_class.n_evaluators, 1, i + 1)
    plt.title(estimator_class.eval_names[i])
    for j in range(estimator_class.n_freq):
        dataset = np.array([final_evaluates[rho][1][i, j] for rho in rhos])
        plt.plot(rhos, dataset, label=estimator_class.freq_names[j])
        plt.legend()
plt.show()










