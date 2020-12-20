import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import cmath
from posteriors import Posterior

class EstimatorClass:
    def __init__(self):
        self.n_estimators = 4#5
        self.n_evaluators = 4
        self.est_funcs = np.array([
            self.e_estimator,
            self.m_estimator,
            self.fi2_estimator,
#            self.kl2_estimator,
            self.MAP_estimator
        ])
        self.est_names = np.array([
            "E",
            "M",
            "FI2",
#            "KL2",
            "MAP"
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
        self.freq_est = np.array([
            self.sample_corr,
            self.c_sample_corr,
            self.t_c_sample_corr,
            self.mle
        ])
        self.freq_names = np.array([
            "Sample corr",
            "Sample corr, var=1, trunc",
            "MLE"
        ])
        self.n_freq = 3

    def get_all_freq_estimators(self,n,rho):
        estimators = np.zeros(len(self.freq_est))
        for i in range(len(self.freq_est)):
            data = np.random.multivariate_normal(
                np.array([0,0]),
                np.array([[1,rho],[rho,1]]),
                size=n
            )
            Xs = data[:,0]
            Ys = data[:,1]
            estimators[i] = self.freq_est[i](Xs,Ys,n)
        return estimators, self.freq_names

    def sample_corr(self,Xs,Ys,n):
        return np.sum(Xs*Ys)/np.sqrt(np.sum(Xs**2)*np.sum(Ys**2))

    def c_sample_corr(self,Xs,Ys,n):
        return np.sum(Xs*Ys)/n

    def t_c_sample_corr(self,Xs,Ys,n):
        temp = np.sum(Xs*Ys)/n
        if temp < -1:
            return -1
        elif temp > 1:
            return 1
        return temp

    def mle(self,Xs,Ys,n):
        SSx = np.sum(Xs**2)
        SSy = np.sum(Ys**2)
        SSxy = np.sum(Xs*Ys)
        phi = -3*n*(n-SSx-SSy)-SSxy**2
        gamma = -36*n**2*SSxy+9*n*SSx*SSxy+9*n*SSy*SSxy-2*SSxy**3
        temp = (gamma+cmath.sqrt(4*phi**3+gamma**2))**(1/3)
        temp1 = phi/(3*n*temp)
        if abs(temp) < 10**(-5):
            print("hei")
            temp1 = (np.abs(gamma)/2)**(1/3)/(3*n)
        temp2 = temp/(3*2**(1/3)*n)
        temp3 = SSxy/(3*n)
        est1 = temp3+2**(1/3)*temp1-temp2
        est2 = temp3-(1+cmath.sqrt(-3))*temp1/(2**(2/3))+(1-cmath.sqrt(-3))*temp2/2
        est3 = temp3-(1-cmath.sqrt(-3))*temp1/(2**(2/3))+(1+cmath.sqrt(-3))*temp2/2
        ests = np.array([est1,est2,est3])
        pdf = lambda x,y,rho: 1/(2*np.pi*np.sqrt(1-rho**2))**n*np.exp(-1/(2*(1-rho**2))*(SSx+SSy-2*rho*SSxy))
        min_arg = -1
        min_val = 0
        for i in range(len(ests)):
            if np.abs(ests[i].imag) < 10**(-10):
                if pdf(Xs,Ys,ests[i].real) > min_val:
                    min_val = pdf(Xs,Ys,ests[i].real)
                    min_arg = i

        return ests[min_arg].real

    def get_all_estimators(self,distr,n,rho):
        estimators = np.zeros(self.n_estimators)
        for i in range(self.n_estimators):
            S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho))  # 2*n*(1+rho)
            S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho))  # 2*n*(1-rho)

            T1 = 1 / 2 * (S1 + S2)
            T2 = 1 / 4 * (S1 - S2)
#            print("hei",S1,S2,T1,T2)
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
        initial = 2 * T2 / T1
        if initial >= 0.95:
            initial = 0.95
        elif initial <= -0.95:
            initial = -0.95
        norm = post.normalization(n,T1,T2)
        Ef = self.expt(post.distribution,self.fisher_information,n,T1,T2,norm)
        def equation(x):
            return Ef-self.fisher_information(x)
        solution = fsolve(equation,initial)
        return solution[0]

    def m_estimator(self,post,n,T1,T2):
        norm = post.normalization(n,T1,T2)
        temp_rhos = np.linspace(-1 + 1 / 100, 1 - 1 / 100, 100)
        temp = post.distribution(temp_rhos, n, T1, T2) / norm
        temp2 = np.cumsum(temp)
        temp3 = temp2 / temp2[-1]
        initial_med = temp_rhos[np.argmin(np.abs(temp3 - 1 / 2))]
        def cum_func(x):
            return quad(post.distribution,-1,x,args=(n,T1,T2))[0]/norm-1/2
        solution = fsolve(cum_func,initial_med)[0]
        cum_val = quad(post.distribution,-1,solution,args=(n,T1,T2))[0]/norm
        if np.abs(cum_val-1/2)>10**(-3):
            print("med_feil",n,T1,T2,solution,cum_val)
        return solution

    def MAP_estimator(self,post,n,T1,T2):
        norm = post.normalization(n,T1,T2)
        temp_rhos = np.linspace(-1 + 1 / 100, 1 - 1 / 100, 50)
        temp = post.distribution(temp_rhos, n, T1, T2) /norm
        initial_map = temp_rhos[np.argmax(temp)]
        def map_func(x,n,T1,T2):
            temp = post.distribution(x,n,T1,T2)/norm
            if temp < 10**(-5):
                return np.infty
            return -np.log(temp)

        solution = minimize(map_func,initial_map,args=(n,T1,T2))

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
        initial = np.tanh(2 * T2 / T1)
#        if initial >= 0.95:
#            initial = 0.95
#        elif initial <= -0.95:
#            initial = -0.95
        norm = post.normalization(n,T1,T2)
        elog = self.expt(post.distribution,self.log_func,n,T1,T2,norm)
        exlog = self.expt(post.distribution,self.xlog_func,n,T1,T2,norm)
        Erho2 = self.expt(post.distribution,self.quad_x,n,T1,T2,norm)
        Erho = self.expt(post.distribution,self.lin_x,n,T1,T2,norm)
        def kl2_func(x):
            return x*np.log(1-x**2)-np.log(1-x**2)*Erho+exlog-x*elog+2*x**2*Erho2
        solution = fsolve(kl2_func,initial)[0]
        if np.abs(kl2_func(solution))<10**(-4):
            print("kl2-feil",T1,T2,initial,solution,kl2_func(solution))
            print("more numbers",elog,exlog,Erho,Erho2)
            plt.figure()
            rhos = np.linspace(-1,1,100)
            plt.plot(rhos,kl2_func(rhos))
            plt.plot(rhos,post.distribution(rhos,n,T1,T2)/norm)
            plt.show()
        return solution

    def fisher_information_metric(self,rho,rho_hat):
        return np.abs(self.fisher_information(rho)-self.fisher_information(rho_hat))

    def MSE(self,rho,rho_hat):
        return (rho-rho_hat)**2

    def MAE(self,rho,rho_hat):
        return np.abs(rho-rho_hat)

    def fully_evaluate_estimate(self,rho,rho_estimate):
        evals = np.zeros((self.n_evaluators,len(rho_estimate)))
        for i in range(self.n_evaluators):
            evals[i,:] = self.evaluations[i](rho,rho_estimate)
        return evals

if __name__ == "__main__":

    n_samples = 100000
    n = 5
    estimator_class = EstimatorClass()
    j_postr = Posterior("PC",lam=10**(-4))
    rhos = np.linspace(0,0.9,10)
    print(rhos)

    print(estimator_class.eval_names)

    final_evaluates = {}

    for rho in rhos:
        print("rho",rho)
        estimates = np.empty((n_samples,estimator_class.n_estimators))
        evaluations = np.empty((n_samples,estimator_class.n_evaluators,estimator_class.n_estimators))
        t = time.time()
        for i in range(n_samples):
            if i%1000 == 0:
                print("time elapsed:",time.time()-t)
                t = time.time()
                print(i)

            estimators, names = estimator_class.get_all_estimators(j_postr,n,rho)
            evaluators = estimator_class.fully_evaluate_estimate(rho,estimators)
            estimates[i,:] = estimators
            evaluations[i,:] = evaluators

        final_evaluates[rho] = (np.mean(estimators,axis=0),np.mean(evaluations,axis=0))
#        mean_evaluations = np.mean(evaluations,axis=0)
#        eval_table = np.hstack((np.transpose([estimator_class.est_names]),mean_evaluations))
#        eval_table = np.vstack((np.hstack(([[0]],[estimator_class.eval_names])),eval_table))
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

    import results_storing

    estimator_names = ["PC10-4"+estimator_class.est_names[i] for i in range(len(estimator_class.est_names))]

    for rho in final_evaluates:
#        results_storing.save_data(n,np.transpose(final_evaluates[rho][1]),estimator_class.eval_names,["PC10-4MAP"],np.round(rho,2))
        results_storing.save_data(n,np.transpose(final_evaluates[rho][1]),estimator_class.eval_names,estimator_names,np.round(rho,2))



#    plt.figure()
#    for i in range(estimator_class.n_evaluators):
#        plt.subplot(estimator_class.n_evaluators,1,i+1)
#        plt.title(estimator_class.eval_names[i])
#        for j in range(estimator_class.n_estimators):
#            dataset = np.array([final_evaluates[rho][1][j,i] for rho in rhos])
#            plt.plot(rhos,dataset,label=estimator_class.est_names[j])
#            plt.legend()
#    plt.show()

