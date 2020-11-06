import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
from posteriors import Posterior

class EstimatorClass:
    def __init__(self):
        self.n_estimators = 4
        self.n_evaluators = 4
        self.est_funcs = np.array([
            self.e_estimator,
            self.m_estimator,
            self.fi2_estimator,
            self.kl2_estimator#,
#            self.MAP_estimator
        ])
        self.est_names = np.array([
            "E(rho)",
            "Median(rho)",
            "FI2",
            "KL2"#,
#            "MAP"
        ])
        self.evaluations = np.array([
            self.fisher_information_metric,
            self.MSE,
            self.MAE,
            self.kullback_leibler
        ])
        self.eval_names = np.array([
            "FIM",
            "MSE",
            "MAE",
            "KL"
        ])

    def get_all_estimators(self,distr,n,rho):
        estimators = np.zeros(self.n_estimators)
        for i in range(self.n_estimators):
            S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho))  # 2*n*(1+rho)
            S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho))  # 2*n*(1-rho)

            T1 = 1 / 2 * (S1 + S2)
            T2 = 1 / 4 * (S1 - S2)
            estimators[i] = self.est_funcs[i](distr,n,T1,T2)
        return estimators, self.est_names

    def fisher_information(self,x):
        if type(x) == type(np.array([])):
            x[x<= -1] = 0
            x[x>= 1] = 0
        elif (x <= (-1) or x >= (1)):
            return 0
        return np.sqrt(2)*np.arctanh(np.sqrt(2)*x/np.sqrt(1+x**2))-np.arcsinh(x)

    def kullback_leibler(self,rho1,rho2):
        return -1/2*np.log((1-rho1**2)/(1-rho2**2))+(1-rho1*rho2)/(1-rho2**2)-1

    def expt(self,distr,func,n,T1,T2,norm):
        def exp_func(x):
            return distr(x,n,T1,T2)*func(x)
        return quad(exp_func,-1,1)[0]/norm

    def e_estimator(self,post,n,T1,T2):
        norm = post.normalization(n,T1,T2)
        return self.expt(post.distribution,self.lin_x,n,T1,T2,norm)

    def fi2_estimator(self,post,n,T1,T2):
        initial = np.tanh(2*T2/T1)
        norm = post.normalization(n,T1,T2)
        Ef = self.expt(post.distribution,self.fisher_information,n,T1,T2,norm)
        def equation(x):
            return Ef-self.fisher_information(x)
        solution = fsolve(equation,initial)
        return solution[0]

    def m_estimator(self,post,n,T1,T2):
        #print("m")
        initial = np.tanh(2*T2/T1)
        norm = post.normalization(n,T1,T2)
        def cum_func(x):
            return quad(post.distribution,-1,x,args=(n,T1,T2))[0]/norm-1/2
        solution = fsolve(cum_func,initial)
        return solution[0]

    def MAP_estimator(self,post,n,T1,T2):
        initial = np.tanh(2*T2/T1)
        solution = minimize(lambda x: -np.log(post.distribution(x,n,T1,T2)),initial)
        return solution.x[0]

    def log_func(self,x):
        return np.log(1 - x ** 2)

    def quad_x(self,x):
        return x ** 2

    def lin_x(self,x):
        return x

    def xlog_func(self,x):
        return x * np.log(1 - x ** 2)

    def kl2_estimator(self,post,n,T1,T2):
        initial = np.tanh(2*T2/T1)
        norm = post.normalization(n,T1,T2)
        elog = self.expt(post.distribution,self.log_func,n,T1,T2,norm)
        exlog = self.expt(post.distribution,self.xlog_func,n,T1,T2,norm)
        Erho2 = self.expt(post.distribution,self.quad_x,n,T1,T2,norm)
        Erho = self.expt(post.distribution,self.lin_x,n,T1,T2,norm)
        solution = fsolve(lambda x:
                     -x * (1 - Erho * x) / (1 - x ** 2) + 1 / 2 * x * elog - 1 / 2 * x * np.log(1 - x ** 2) + x +
                     (Erho - x * Erho2) / (1 - x ** 2) + 1 / 2 * Erho * np.log(1 - x ** 2) - 1 / 2 * exlog - Erho
                     , initial)
        return solution[0]

    def fisher_information_metric(self,rho,rho_hat):
        return np.abs(self.fisher_information(rho)-self.fisher_information(rho_hat))

    def MSE(self,rho,rho_hat):
        return (rho-rho_hat)**2

    def MAE(self,rho,rho_hat):
        return np.abs(rho-rho_hat)

    def fully_evaluate_estimate(self,rho,rho_estimate):
        evals = np.zeros((self.n_estimators,self.n_evaluators))
        for i in range(self.n_evaluators):
            evals[i] = self.evaluations[i](rho,rho_estimate)
        return evals

if __name__ == "__main__":

    n_samples = 10000
    n = 3
    estimator_class = EstimatorClass()
    j_postr = Posterior("jeffrey")
    rhos = np.linspace(0,0.9,10)
    print(rhos)

    final_evaluates = {}

    for rho in rhos:
        print("rho",rho)
        estimates = np.empty((n_samples,estimator_class.n_estimators))
        evaluations = np.empty((n_samples,estimator_class.n_estimators,estimator_class.n_evaluators))
        for i in range(n_samples):
            if i%100 == 0:
                print(i)

            estimators, names = estimator_class.get_all_estimators(j_postr,n,rho)
            evaluators = estimator_class.fully_evaluate_estimate(rho,estimators)
            estimates[i,:] = estimators
            evaluations[i,:] = evaluators

        final_evaluates[rho] = (np.mean(evaluators,axis=0),np.mean(evaluations,axis=0))
        mean_evaluations = np.mean(evaluations,axis=0)
        eval_table = np.hstack((np.transpose([estimator_class.est_names]),mean_evaluations))
        eval_table = np.vstack((np.hstack(([[0]],[estimator_class.eval_names])),eval_table))
    #    print(eval_table)

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

    import pickle

    pickle.dump(final_evaluates, open("jeffrey.p","wb"))

    plt.figure()
    for i in range(estimator_class.n_evaluators):
        plt.subplot(estimator_class.n_evaluators,1,i+1)
        plt.title(estimator_class.eval_names[i])
        for j in range(estimator_class.n_estimators):
            dataset = np.array([final_evaluates[rho][1][j,i] for rho in rhos])
            plt.plot(rhos,dataset,label=estimator_class.est_names[j])
            plt.legend()
    plt.show()

