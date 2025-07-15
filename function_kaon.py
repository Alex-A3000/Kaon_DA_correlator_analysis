# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:12:44 2025

@author: s9503
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
from multiprocessing import Pool, freeze_support
import warnings
import random
import math
from scipy import linalg
import sys
from sklearn.covariance import LedoitWolf, OAS
from scipy.linalg import pinv
warnings.filterwarnings("ignore")

def progress_bar(progress, total, length=40):
    progress = progress + 1
    percent = 100 * (progress / float(total))
    bar = '█' * int(length * progress // total) + '-' * (length - int(length * progress // total))
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')
    sys.stdout.flush()
    if percent == 100:
        print("finish")

class Pair:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, str(k), v)
            
def R_uv(corre,E_p,Z_p,tau_e,T):
    corre = np.array(corre)
    R = np.zeros(corre.shape)
    for tau_m in range(T):
        R[tau_e - tau_m] = 2*E_p*corre[tau_m]/(Z_p*np.exp(-E_p*(tau_e + tau_m)/2))
    return R

def create_3pt_name(kappa_l,kappa_s,s_l,k_h,g_k,tau_e,g_h,p_e,p_m):
    return ("correlator-kl-"
                + kappa_l + "-ks-"
                + kappa_s + "-"
                + s_l + "-kh-"
                + k_h + "-"
                + g_k + "-"
                + "tau_e-" + str(tau_e) + "-"
                + g_h
                + "-pe-"
                + str(p_e[0])
                + str(p_e[1])
                + str(p_e[2])
                + "-pm-"
                + str(p_m[0])
                + str(p_m[1])
                + str(p_m[2])+".dat")

def hadronic_tensor(dir_3pt, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, two_pt_fitting_result):
    R = np.zeros((len(tau_es), 4, Lt))
    R_b = np.zeros((len(tau_es), 4, 4*Nconf, Lt))
    C = np.zeros((len(tau_es), 4, Lt))
    C_b = np.zeros((len(tau_es), 4, Nconf, Lt))
    (R_im_odd, R_re_odd, R_im_even, R_re_even) = ([[] for i in range(len(tau_es))] for j in range(4))
    (R_im_odd_b, R_re_odd_b, R_im_even_b, R_re_even_b) = ([[] for i in range(len(tau_es))] for j in range(4))
    eigenvectors = two_pt_fitting_result["eigenvector"]
    OPfitting_exponential = two_pt_fitting_result["exponential"]
    (A, B) = eigenvectors[1]
    Ek = OPfitting_exponential[1].best_fit.res[0]
    Zk = (OPfitting_exponential[1].best_fit.res[1] *2 *Ek)**0.5
    conf_name = dir_3pt.rstrip("/")
    reweightingfactors = np.loadtxt("reweightingfactors/" + conf_name + ".dat")[:4*Nconf:4,3]
    for n_t, tau_e in enumerate(tau_es):
        g_k = "15"
        filename15 = [
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            ]
        g_k = "7"
        filename7 = [
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            ]
        for n_f in range(4):
            data_15 = Cut_conf(np.loadtxt(dir_3pt + filename15[n_f]), Lt)[:,:,1] * reweightingfactors.reshape((Nconf,1))
            data_7  = Cut_conf(np.loadtxt(dir_3pt + filename7[n_f]), Lt)[:,:,1] * reweightingfactors.reshape((Nconf,1))
            data_f = abs(A)*data_7 + abs(B)*data_15 
            C[n_t,n_f,:] = np.mean(data_15,0)
            C_b[n_t,n_f,:,:] = data_15
            R[n_t,n_f,:] = R_uv(np.mean(data_f,0), Ek, Zk, tau_e, Lt)
            R_b[n_t,n_f,:,:] = R_uv(Bootstrap(data_f, 4, 0).T, Ek, Zk, tau_e, Lt).T
        
        R_tau_p_odd = 1/2*(R[n_t,0] - R[n_t,2])
        R_tau_n_odd =  1/2*(- R[n_t,3] + R[n_t,1])
        R_tau_p_b_odd = 1/2*(R_b[n_t,0] - R_b[n_t,2])
        R_tau_n_b_odd = 1/2*( - R_b[n_t,3] + R_b[n_t,1])
        R_tau_sym_odd =  1/2*(R_tau_p_odd + R_tau_n_odd) 
        R_tau_ant_odd =  1/2*(R_tau_p_odd - R_tau_n_odd)
        R_tau_sym_b_odd = 1/2*(R_tau_p_b_odd + R_tau_n_b_odd)
        R_tau_ant_b_odd = 1/2*(R_tau_p_b_odd - R_tau_n_b_odd)
        
        R_tau_p_eve =  1/2*(R[n_t,0] + R[n_t,2])
        R_tau_n_eve =  1/2*(- R[n_t,3] - R[n_t,1])
        R_tau_p_b_eve = 1/2*(R_b[n_t,0] + R_b[n_t,2])
        R_tau_n_b_eve = 1/2*(- R_b[n_t,3] - R_b[n_t,1])
        R_tau_sym_eve =  1/2*(R_tau_p_eve + R_tau_n_eve) 
        R_tau_ant_eve =  1/2*(R_tau_p_eve - R_tau_n_eve)
        R_tau_sym_b_eve = 1/2*(R_tau_p_b_eve + R_tau_n_b_eve)
        R_tau_ant_b_eve = 1/2*(R_tau_p_b_eve - R_tau_n_b_eve)
               
        R_re_odd [n_t].append(R_tau_ant_odd)
        R_re_even[n_t].append(R_tau_ant_eve)
        R_re_odd_b [n_t].append(R_tau_ant_b_odd)
        R_re_even_b[n_t].append(R_tau_ant_b_eve)

        R_im_odd [n_t].append(R_tau_sym_odd)
        R_im_even[n_t].append(R_tau_sym_eve)
        R_im_odd_b [n_t].append(R_tau_sym_b_odd)
        R_im_even_b[n_t].append(R_tau_sym_b_eve)
    return Pair(im_odd=R_im_odd, im_even=R_im_even, re_odd=R_re_odd, re_even=R_re_even,
                im_odd_b=R_im_odd_b, im_even_b=R_im_even_b, re_odd_b=R_re_odd_b, re_even_b=R_re_even_b
                ,R=R,R_b=R_b,C=C,C_b=C_b)


def hadronic_tensor_exc(fun, Ntau, dir_3pt, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, two_pt_fitting_result):
    R = np.zeros((len(tau_es), 4, Lt))
    R_b = np.zeros((len(tau_es), 4, 4*Nconf, Lt))
    ER = np.zeros((4, Ntau))
    ER_b = np.zeros((4, 4*Nconf, Ntau))
    eigenvectors = two_pt_fitting_result["eigenvector"]
    OPfitting_exponential = two_pt_fitting_result["exponential"]
    (A, B) = eigenvectors[1]
    Ek = OPfitting_exponential[1].best_fit.res[0]
    Zk = (OPfitting_exponential[1].best_fit.res[1] *2 *Ek)**0.5
    conf_name = dir_3pt.rstrip("/")
    reweightingfactors = np.loadtxt("reweightingfactors/" + conf_name + ".dat")[:4*Nconf:4,3]
    for n_f in range(4):
        for n_t, tau_e in enumerate(tau_es):
            g_k = "15"
            filename15 = [
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            ]
            g_k = "7"
            filename7 = [
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"l-s", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (0,0,1), (1,0,-1)),
            "3pt_" + create_3pt_name(kappa_l, kappa_s,"s-l", kappa_h, g_k, tau_e, "ge-14-gm-13", (1,0,-1), (0,0,1)),
            ]
            data_15 = Cut_conf(np.loadtxt(dir_3pt + filename15[n_f]), Lt)[:,:,1] * reweightingfactors.reshape((Nconf,1))
            data_7  = Cut_conf(np.loadtxt(dir_3pt + filename7[n_f]), Lt)[:,:,1] * reweightingfactors.reshape((Nconf,1))
            data_f = A*data_15 - B*data_7
            R[n_t,n_f,:] = R_uv(np.mean(data_f,0), Ek, Zk, tau_e, Lt)
            R_b[n_t,n_f,:,:] = R_uv(Bootstrap(data_f, 4, 0).T, Ek, Zk, tau_e, Lt).T
        for tau in range(Ntau):
            fitting = Find_best_fitting_AIC_range_linear(fun, np.array(tau_es), R[:,n_f,-tau], R_b[:,n_f,:,-tau])
            ER[n_f,tau] = fitting.best_fit.res[0]
            ER_b[n_f,:,tau] = np.array(fitting.best_fit.boots_res)[:,0]
            
        R_tau_p_odd = 1/2*(ER[0] - ER[2])
        R_tau_n_odd =  1/2*(- ER[3] + ER[1])
        R_tau_p_b_odd = 1/2*(ER_b[0] - ER_b[2])
        R_tau_n_b_odd = 1/2*( - ER_b[3] + ER_b[1])
        R_tau_sym_odd =  1/2*(R_tau_p_odd + R_tau_n_odd) 
        R_tau_ant_odd =  1/2*(R_tau_p_odd - R_tau_n_odd)
        R_tau_sym_b_odd = 1/2*(R_tau_p_b_odd + R_tau_n_b_odd)
        R_tau_ant_b_odd = 1/2*(R_tau_p_b_odd - R_tau_n_b_odd)
        
        R_tau_p_eve =  1/2*(ER[0] + ER[2])
        R_tau_n_eve =  1/2*(- ER[3] - ER[1])
        R_tau_p_b_eve = 1/2*(ER_b[0] + ER_b[2])
        R_tau_n_b_eve = 1/2*(- ER_b[3] - ER_b[1])
        R_tau_sym_eve =  1/2*(R_tau_p_eve + R_tau_n_eve) 
        R_tau_ant_eve =  1/2*(R_tau_p_eve - R_tau_n_eve)
        R_tau_sym_b_eve = 1/2*(R_tau_p_b_eve + R_tau_n_b_eve)
        R_tau_ant_b_eve = 1/2*(R_tau_p_b_eve - R_tau_n_b_eve)
               
    return Pair(im_odd=R_tau_sym_odd, im_even=R_tau_sym_eve, re_odd=R_tau_ant_odd, re_even=R_tau_ant_eve,
                im_odd_b=R_tau_sym_b_odd, im_even_b=R_tau_sym_b_eve, re_odd_b=R_tau_ant_b_odd, re_even_b=R_tau_ant_b_eve
                ,R=ER, R_b=ER_b)
    
def minimize_to_curve_fit(bounds_minimize):
    """
    Convert bounds from minimize format [(min1, max1), (min2, max2), ...]
    to curve_fit format ([min1, min2, ...], [max1, max2, ...])
    """
    lower_bounds, upper_bounds = zip(*bounds_minimize)  # Unzip into two lists
    return (list(lower_bounds), list(upper_bounds))  # Convert to list format

def Fit_cov(fun, x, meandata, bootsdata, boots=True, mini=True, shrinkage=0, tol=1e-10, p0=[1e-5], maxfev=1e5, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead', shows=True):
    # Suppose bootsdata is [ndata, nboot]
    # Transpose to [nboot, ndata] for sklearn (each row = one sample)
    bootsdata_T = bootsdata.T
    
    if shrinkage=='LW':        
        # Fit the shrinkage covariance estimator
        lw = LedoitWolf().fit(bootsdata_T)      
        # Extract the regularized covariance matrix
        cov_matrix = lw.covariance_
        lambdas = lw.shrinkage_
        inv_cov_matrix = pinv(cov_matrix)
        
    elif shrinkage=='OAS':
        oas = OAS().fit(bootsdata_T)
        cov_matrix = oas.covariance_
        lambdas = oas.shrinkage_
        inv_cov_matrix = pinv(cov_matrix)
        
    else:
        sample_cov = np.cov(bootsdata_T, rowvar=False)
        target = np.trace(sample_cov) / sample_cov.shape[0] * np.eye(sample_cov.shape[0])
        lambdas = shrinkage
        cov_matrix = (1 - lambdas) * sample_cov + lambdas * target
        inv_cov_matrix = pinv(cov_matrix)
    
    # Define the chi-square function
    def chisqfun(a):
        residuals = meandata - fun(x,*a)
        return np.dot(residuals, np.dot(inv_cov_matrix, residuals.T))
    
    # Estimate initial parameters if x0 is not provided
    if x0 is None:
        #x0, _ = opt.curve_fit(fun, x, meandata, p0=np.ones(p0))
        try:
            x0, _ = opt.curve_fit(fun, x, meandata, p0=p0, sigma=cov_matrix, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
            if shows == True:
                print(f"curve fit initial serch: {x0}")
        except:
            if shows == True:          
                print("curve fit faild")
            x0 = np.array(p0)
    # Perform fitting for mean chi-square
    if mini == True:      
        result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
        red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
        main_result = result.x
    else :
        red_chi_square = chisqfun(x0) / (len(meandata) - len(x0))
        main_result = x0
    if shows == True:
        print(f"main result: {main_result}")
        print(f"chi square: {red_chi_square}")
    main_chi_square = red_chi_square
    
    # Perform fitting for each bootstrapped sample
    results = []
    chi_squares = []
    if boots == True:        
        for i in range(len(bootsdata[0])):
            def chisqfun(a):
                residuals = bootsdata[:, i] - fun(x,*a)
                return np.dot(residuals, np.dot(inv_cov_matrix, residuals.T))
            if mini == True:              
                result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
                red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
                results.append(result.x)
            else :
                result, _ = opt.curve_fit(fun, x, bootsdata[:, i], p0=p0, sigma=cov_matrix, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
                red_chi_square = 0
                results.append(result)
            chi_squares.append(red_chi_square)
            if shows == True:    
                progress_bar(i, len(bootsdata[0]))
        if shows == True:
            print(f"erro: {Bootstrap_erro(np.array(results))}")
            print("fin")
    return Pair(res=main_result, chi=main_chi_square, chi_aic=main_chi_square*(len(meandata) - len(x0)), boots_res=np.array(results), boots_chi=chi_squares, x0=x0 , x=x, lambdas=lambdas)

def Fit_cov_AIC(fun, x, meandata, bootsdata, boots=True, mini=True, shrinkage=0, tol=1e-10, p0=[1e-5], maxfev=1e5, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead', shows=True, min_subset_size=None):
    # Determine the number of data points
    num_points = len(x)

    # Default max subset size to the total number of points if not specified
    if min_subset_size is None:
        min_subset_size = len(p0) + 1

    # Generate all combinations of data points for subsets of size len(p0)+1 to max_subset_size
    all_AIC = []
    all_chi_square = []
    all_results = []
    subsets_to_fit = []
    print("Start fitting")
    start_time = time.time()
    for start in range(0, num_points - min_subset_size + 1):
        progress_bar(start, len(range(0, num_points - min_subset_size + 1)))
        for end in range(start + min_subset_size - 1, num_points):
            subsets_to_fit.append(list(range(start, end + 1)))
            fit = Fit_cov(fun, x[start: end + 1], meandata[start: end + 1], bootsdata[start: end + 1, :], boots=False, mini=True, shrinkage=shrinkage, tol=tol, p0=p0, maxfev=maxfev, bounds=bounds, x0=None, Method=Method, shows=False)
            all_chi_square.append(fit.chi)
            all_AIC.append(Akaike_Information_Criterion(fit, len(x) - len(fit.x)))
            all_results.append(fit.res)
    end_time = time.time()
    print(f"execution time: {end_time - start_time} seconds")        
    # Find the best fit based on AIC
    max_AIC_index = np.argmax(all_AIC)
    best_subset = subsets_to_fit[max_AIC_index]
    best_range = x[best_subset]
    print(f"AIC range: {best_range[0]} to {best_range[-1]}")
    best_fit = Fit_cov(fun, x[best_subset], meandata[best_subset], bootsdata[best_subset], boots=boots, mini=True, shrinkage=shrinkage, tol=tol, p0=p0, maxfev=maxfev, bounds=bounds, x0=None, Method=Method, shows=True)
    best_result = best_fit.res
    best_chi_square = best_fit.chi
    best_boots_reslut = best_fit.boots_res
    return Pair(AIC=all_AIC[max_AIC_index], chi=best_chi_square, subset=best_subset, best_range=(best_range[0], best_range[-1]),
                res=best_result, boots_res=best_boots_reslut, best_fit=best_fit, 
                all_results=all_results, all_chi_square=all_chi_square, all_AIC=all_AIC, all_subsets=subsets_to_fit)


def Fit_cov_Shrinkage(fun, x, meandata, bootsdata, shrinkage='LW', tol=1e-10, p0=[1e-5], maxfev=1e5, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead'):
    # Suppose bootsdata is [ndata, nboot]
    # Transpose to [nboot, ndata] for sklearn (each row = one sample)
    bootsdata_T = bootsdata.T
    
    if shrinkage=='LW':        
        # Fit the shrinkage covariance estimator
        lw = LedoitWolf().fit(bootsdata_T)      
        # Extract the regularized covariance matrix
        cov_matrix = lw.covariance_
        lambdas = lw.shrinkage_
        inv_cov_matrix = pinv(cov_matrix)
        
    elif shrinkage=='OAS':
        oas = OAS().fit(bootsdata_T)
        cov_matrix = oas.covariance_
        lambdas = oas.shrinkage_
        inv_cov_matrix = pinv(cov_matrix)
        
    else:
        sample_cov = np.cov(bootsdata_T, rowvar=False)
        target = np.trace(sample_cov) / sample_cov.shape[0] * np.eye(sample_cov.shape[0])
        lambdas = shrinkage
        cov_matrix = (1 - lambdas) * sample_cov + lambdas * target
        inv_cov_matrix = pinv(cov_matrix)
    
    # Define the chi-square function
    def chisqfun(a):
        residuals = meandata - fun(x,*a)
        return np.dot(residuals, np.dot(inv_cov_matrix, residuals.T))
    
    # Estimate initial parameters if x0 is not provided
    if x0 is None:
        #x0, _ = opt.curve_fit(fun, x, meandata, p0=np.ones(p0))
        try:
            x0, _ = opt.curve_fit(fun, x, meandata, p0=p0, sigma=cov_matrix, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
        except:
            x0 = np.array(p0)
    print(x0)
    # Perform fitting for mean chi-square
    result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
    red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
    main_result = result.x
    print(red_chi_square)
    main_chi_square = red_chi_square
    
    # Perform fitting for each bootstrapped sample
    results = []
    chi_squares = []
    for i in range(len(bootsdata[0])):
        def chisqfun(a):
            residuals = bootsdata[:, i] - fun(x,*a)
            return np.dot(residuals, np.dot(inv_cov_matrix, residuals.T))
        
        result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
        red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
        results.append(result.x)
        chi_squares.append(red_chi_square)
        progress_bar(i, len(bootsdata[0]))                            
    return Pair(res=main_result, chi=main_chi_square, chi_aic=main_chi_square*(len(meandata) - len(x0)), boots_res=np.array(results), boots_chi=chi_squares, x0=x0 , x=x, lambdas=lambdas)


def Fit_opt(fun, x, meandata, bootsdata, p0=[1e-5], maxfev=1000, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead'):
    ### bootsdata -> conf 
    # Calculate covariance matrix and its inverse
    sigma = Bootstrap_erro(bootsdata,1)
    # Define the chi-square function
    def chisqfun(a):
        residuals = meandata - fun(x,*a)
        return np.sum(residuals**2 /(sigma**2 + sys.float_info.epsilon))
    
    # Estimate initial parameters if x0 is not provided
    if x0 is None:
        #x0, _ = opt.curve_fit(fun, x, meandata, p0=np.ones(p0))
        try:
            x0, _ = opt.curve_fit(fun, x, meandata, p0=p0, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
        except:
            x0 = np.array(p0)
    print(x0)
    # Perform fitting for mean chi-square
    red_chi_square = chisqfun(x0) / (len(meandata) - len(x0))
    main_result = x0
    main_chi_square = red_chi_square
    
    # Perform fitting for each bootstrapped sample
    results = []
    chi_squares = []
    for i in range(len(bootsdata[0])):
        def chisqfun(a):
            residuals = bootsdata[:, i] - fun(x,*a)
            return np.sum(residuals**2 /(sigma**2 + sys.float_info.epsilon))
        
        try:
            x0, _ = opt.curve_fit(fun, x, bootsdata[:, i], p0=p0, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
        except:
            x0 = np.array(p0)
        red_chi_square = chisqfun(x0) / (len(meandata) - len(x0))
        results.append(x0)
        chi_squares.append(red_chi_square)
        progress_bar(i, len(bootsdata[0]))                        
    return Pair(res=main_result, chi=main_chi_square, chi_aic=main_chi_square*(len(meandata) - len(x0)), boots_res=np.array(results), boots_chi=chi_squares, x=x, x0=x0)

def Fit_min(fun, x, meandata, bootsdata, p0=[1e-5], maxfev=1000, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead', tol=1e-7):
    ### bootsdata -> conf 
    # Calculate covariance matrix and its inverse
    sigma = Bootstrap_erro(bootsdata,1)
    # Define the chi-square function
    def chisqfun(a):
        residuals = meandata - fun(x,*a)
        return np.sum(residuals**2 /(sigma**2 + sys.float_info.epsilon))
    
    # Estimate initial parameters if x0 is not provided
    if x0 is None:
        #x0, _ = opt.curve_fit(fun, x, meandata, p0=np.ones(p0))
        try:
            x0, _ = opt.curve_fit(fun, x, meandata, p0=p0, maxfev=maxfev, bounds=minimize_to_curve_fit(bounds))
        except:
            x0 = np.array(p0)
    print(x0)
    # Perform fitting for mean chi-square
    result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
    red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
    main_result = result.x
    main_chi_square = red_chi_square
    
    # Perform fitting for each bootstrapped sample
    results = []
    chi_squares = []
    for i in range(len(bootsdata[0])):
        def chisqfun(a):
            residuals = bootsdata[:, i] - fun(x,*a)
            return np.sum(residuals**2 /(sigma**2 + sys.float_info.epsilon))
        result = opt.minimize(chisqfun, x0, method=Method, tol=tol, bounds=bounds)
        red_chi_square = chisqfun(result.x) / (len(meandata) - len(x0))
        results.append(result.x)
        chi_squares.append(red_chi_square)
        progress_bar(i, len(bootsdata[0]))                  
    return Pair(res=main_result, chi=main_chi_square, chi_aic=main_chi_square*(len(meandata) - len(x0)), boots_res=np.array(results), boots_chi=chi_squares, x=x, x0=x0)


def Akaike_Information_Criterion(fit_result, Ncut):      
    return np.exp(-0.5 * fit_result.chi_aic - len(fit_result.x0) - Ncut)

def GEVP(meandata, bootsdata, section, t0=0, vec0=None, flip=True):
    Lmatrix = len(section)
    A  = np.zeros((Lmatrix,Lmatrix))
    B  = np.zeros((Lmatrix,Lmatrix))
    vec0=np.array(vec0)
    
    Lt = meandata.shape[0]
    Nconf = bootsdata.shape[1]
    vel = np.zeros((Lt-t0,Lmatrix),dtype=complex)
    vec = np.zeros((Lt-t0,Lmatrix,Lmatrix),dtype=complex)
    vel_boots = np.zeros((Lt-t0,Nconf,Lmatrix),dtype=complex)
    vec_boots = np.zeros((Lt-t0,Nconf,Lmatrix,Lmatrix),dtype=complex)
    
    optimize = np.zeros((Lt-t0,Lmatrix),dtype=complex)
    optimize_boots = np.zeros((Lt-t0,Nconf,Lmatrix),dtype=complex)
    
    for t in range(Lt - t0):
        for x in range(Lmatrix):
            for y in range(Lmatrix):
                A[y,x] = meandata[t0,section[y],section[x]]
                B[y,x] = meandata[t0+t,section[y],section[x]]
        
        A = np.asmatrix(A)
        B = np.asmatrix(B)
        eigenValues, eigenVectors = linalg.eig(B,A,left=False,right=True)
        idx = eigenValues.argsort()[::-1]
        vel[t] = eigenValues[idx]
        if flip == True:
            eigenVectors[:,idx][0] < 0
            eigenVectors = -eigenVectors
        vec[t] = eigenVectors[:,idx]
        
        if vec0.all() != None:
            for x in range(Lmatrix):
                for y in range(Lmatrix):
                    optimize[t] += meandata[t0+t,section[y],section[x]]*vec0[x]*np.conjugate(vec0[y])
        
    for conf in range(Nconf):
        for t in range(Lt - t0):
            for x in range(Lmatrix):
                for y in range(Lmatrix):
                    A[y,x] = bootsdata[t0,conf,section[y],section[x]]
                    B[y,x] = bootsdata[t0+t,conf,section[y],section[x]]
            
            A = np.asmatrix(A)
            B = np.asmatrix(B)
            eigenValues, eigenVectors = linalg.eig(B,A,left=False,right=True)
            idx = eigenValues.argsort()[::-1]
            vel_boots[t,conf] = eigenValues[idx]
            if flip == True:
                eigenVectors[:,idx][0] < 0
                eigenVectors = -eigenVectors
            vec_boots[t,conf] = eigenVectors[:,idx]
            
            if vec0.all() != None:
                for x in range(Lmatrix):
                    for y in range(Lmatrix):
                        optimize_boots[t,conf] += bootsdata[t0+t,conf,section[y],section[x]]*vec0[x]*np.conjugate(vec0[y])
    
    return Pair(vel=vel,vec=vec,vel_bs=vel_boots,vec_bs=vec_boots,optimize=optimize,optimize_bs=optimize_boots)

def Find_best_fitting_AIC_range(fun, x, meandata, bootsdata, p0=[1e-5], maxfev=100000, bounds=[[-1], [1]], num_processes=4, x0=None, Method='Nelder-Mead', min_subset_size=None):
    """
    Find the best fitting using the Akaike Information Criterion (AIC) with arbitrary data point choices.
    :param fun: The function to fit.
    :param x: The x data.
    :param meandata: Mean of the data points.
    :param bootsdata: Bootstrapped data points.
    :param p0: Initial guess for the parameters.
    :param maxfev: Maximum number of function evaluations.
    :param bounds: Bounds for the parameters.
    :param num_processes: Number of processes for parallel computation.
    :param x0: Initial parameters for fitting.
    :param Method: Optimization method.
    :param max_subset_size: Maximum size of subsets to consider for fitting.
    :return: Pair object containing best fitting results.
    """
    # Determine the number of data points
    num_points = len(x)

    # Default max subset size to the total number of points if not specified
    if min_subset_size is None:
        min_subset_size = len(p0) + 1

    # Generate all combinations of data points for subsets of size len(p0)+1 to max_subset_size
    subsets_to_fit = []

    for start in range(0, num_points - min_subset_size + 1):
        for end in range(start + min_subset_size - 1, num_points):
            subsets_to_fit.append(range(start, end + 1))

    # Prepare ranges to fit for multiprocessing
    ranges_to_fit = [(fun, x[list(subset)], meandata[list(subset)], bootsdata[list(subset), :], p0, maxfev, bounds, x0, Method) for subset in subsets_to_fit]

    start_time = time.time()
    print("Start fitting with arbitrary subsets of data points")
    freeze_support()
    # Parallel processing using Pool
    with Pool(processes=num_processes) as pool:
        all_fit = pool.starmap(Fit_cov, ranges_to_fit)

    end_time = time.time()
    print(f"Multiprocessing execution time: {end_time - start_time} seconds")

    # Extract results and compute AIC
    all_results = [fit.res for fit in all_fit]
    all_chi_square = [fit.chi for fit in all_fit]
    all_AIC = [Akaike_Information_Criterion(fit, len(x) - len(fit.x)) for fit in all_fit]

    # Find the best fit based on AIC
    max_AIC_index = np.argmax(all_AIC)
    best_fit = all_fit[max_AIC_index]
    best_result = best_fit.res
    best_chi_square = best_fit.chi
    best_subset = subsets_to_fit[max_AIC_index]
    best_range = x[list(best_subset)]
    return Pair(max_AIC=all_AIC[max_AIC_index], best_chi=best_chi_square, best_subset=best_subset, best_range=(best_range[0], best_range[-1]),
                best_result=best_result, best_fit=best_fit, all_fit=all_fit, all_results=all_results, 
                all_chi_square=all_chi_square, all_AIC=all_AIC, all_subsets=subsets_to_fit)

def Find_best_fitting_AIC_range_linear(fun, x, meandata, bootsdata, p0=[1e-5], maxfev=1e5, bounds=[(-np.inf,np.inf)], x0=None, Method='Nelder-Mead', min_subset_size=None):
    """
    Find the best fitting using the Akaike Information Criterion (AIC) with arbitrary data point choices.
    :param fun: The function to fit.
    :param x: The x data.
    :param meandata: Mean of the data points.
    :param bootsdata: Bootstrapped data points.
    :param p0: Initial guess for the parameters.
    :param maxfev: Maximum number of function evaluations.
    :param bounds: Bounds for the parameters.
    :param num_processes: Number of processes for parallel computation.
    :param x0: Initial parameters for fitting.
    :param Method: Optimization method.
    :param max_subset_size: Maximum size of subsets to consider for fitting.
    :return: Pair object containing best fitting results.
    """
    # Determine the number of data points
    num_points = len(x)

    # Default max subset size to the total number of points if not specified
    if min_subset_size is None:
        min_subset_size = len(p0) + 1

    # Generate all combinations of data points for subsets of size len(p0)+1 to max_subset_size
    all_fit = []
    subsets_to_fit = []
    start_time = time.time()
    print("Start fitting")
    for start in range(0, num_points - min_subset_size + 1):
        for end in range(start + min_subset_size - 1, num_points):
            subsets_to_fit.append(range(start, end + 1))
            print(f"range:({start}, {end+1}")
            all_fit.append(Fit_cov(fun, x[start: end + 1], meandata[start: end + 1], bootsdata[start: end + 1, :], p0=p0, maxfev=maxfev, bounds=bounds, x0=x0, Method=Method, shows=False, boots=False))
    end_time = time.time()
    print(f"Multiprocessing execution time: {end_time - start_time} seconds")

    # Extract results and compute AIC
    all_results = [fit.res for fit in all_fit]
    all_chi_square = [fit.chi for fit in all_fit]
    all_AIC = [Akaike_Information_Criterion(fit, len(x) - len(fit.x)) for fit in all_fit]

    # Find the best fit based on AIC
    max_AIC_index = np.argmax(all_AIC)
    best_subset = subsets_to_fit[max_AIC_index]
    best_fit = Fit_cov(fun, x[start: end + 1], meandata[start: end + 1], bootsdata[start: end + 1, :], p0=p0, maxfev=maxfev, bounds=bounds, x0=x0, Method=Method)
    best_result = best_fit.res
    best_chi_square = best_fit.chi
    best_range = x[list(best_subset)]
    return Pair(max_AIC=all_AIC[max_AIC_index], best_chi=best_chi_square, best_subset=best_subset, best_range=(best_range[0], best_range[-1]),
                best_result=best_result, best_fit=best_fit, all_fit=all_fit, all_results=all_results, 
                all_chi_square=all_chi_square, all_AIC=all_AIC, all_subsets=subsets_to_fit)

def Bootstrap(data,boots,conf_axi=0,seed0=0):
    data = np.swapaxes(np.array(data), 0, conf_axi)
    a = list(data.shape)
    a[0] = boots*a[0]
    A = np.zeros(a)
    random.seed(seed0)
    for i in range(boots*data.shape[0]):
        x = random.choices(data , k = data.shape[0])
        A[i] = np.mean(x,0)
    return np.swapaxes(A, conf_axi, 0)

def Bootstrap_erro(data,conf_axi=0):
    data = np.swapaxes(np.array(data), 0, conf_axi)
    B = np.sort(data,0)
    Erro = (1/2)*(B[math.ceil(data.shape[0]*(0.5 + 0.341344746))] - B[math.ceil(data.shape[0]*(0.5 - 0.341344746))])
    return np.squeeze(np.swapaxes(np.array([Erro]), conf_axi, 0),conf_axi)

def Cut_conf(data,T):
    a = 0
    A = np.zeros((np.int64(data.shape[0]/T),T,data.shape[1]))
    for i in range(np.int64(data.shape[0]/T)):
        A[i] = data[a:a+T]
        a += T
    return A

def remove_outliers_mad(data, threshold=3):
    median = np.median(data,0)
    abs_dev = np.abs(data - median)
    mad = np.median(abs_dev)
    
    # Define lower and upper bounds
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad
    
    return [x for x in data if lower_bound <= x <= upper_bound]

def Effectmass(data,L,t0=0): # L is half of lattice size, t0 is start point
    a = (data[1:] + np.sqrt(np.abs(data[1:]**2 - data[L - t0]**2)))/(data[0:-1] + np.sqrt(np.abs(data[0:-1]**2 - data[L - t0]**2)))
    return(-1*np.log(a))

def Effectmass_ln(data,L=1,t0=0): # L is half of lattice size, t0 is start point
    a = (data[1:])/(data[0:-1])
    return(-1*np.log(a))

def Fold_data_cosh(data): 
    A = []
    A.append(data[0])
    for i in range(1,np.int64(len(data)/2)):
        A.append((data[i] + data[-i])/2)
    A.append(data[np.int64(len(data)/2)])
    A = np.array(A)
    return A

def Fold_data_sinh(data):
    A = []
    A.append(data[0])
    for i in range(1,np.int64(len(data)/2)):
        A.append((data[i] - data[-i])/2)    
    A.append(data[np.int64(len(data)/2)])
    A = np.array(A)
    return A

def Fold_data(data,sign=1):
    if sign == 1:
        A = []
        A.append(data[0])
        for i in range(1,np.int64(len(data)/2)):
            A.append((data[i] + data[-i])/2)
        A.append(data[np.int64(len(data)/2)])
        A = np.array(A)
    elif sign == -1:
        A = []
        A.append(data[0])
        for i in range(1,np.int64(len(data)/2)):
            A.append((data[i] - data[-i])/2)    
        A.append(data[np.int64(len(data)/2)])
        A = np.array(A)
    return A

def Plot(data,erro=None,origin=[0,1],x=None,xshift=0,Title='',Xlabel='',Ylabel='',Xlimt=None,Ylimt=None,Label='',Ltitle='',Line='',Mark='o'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
    data = np.array(data)
    erro = np.array(erro)
    x = np.array(x)
    d = len(list(data.shape))
    plt.title(Title,fontsize = 14)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    if erro.all() != None:
        xdata = np.linspace(origin[0],origin[0] + origin[1]*(data.T.shape[0] - 1),data.T.shape[0],axis= 0)
        dx = xshift*np.ones(len(xdata))
        if x.all() != None:
            xdata = x
        if d != 1:
            if Label != '':    
                if len(Label) <= 1:
                    Label = [Label[0] for i in range(data.shape[0])]
            for i in range(data.shape[0]):
                if Label != '':
                    ax.errorbar(xdata+i*dx,data[i],erro[i],linestyle=Line,marker=Mark,ms=5,label=Label[i])
                else:
                    ax.errorbar(xdata+i*dx,data[i],erro[i],linestyle=Line,marker=Mark,ms=5,label=Label)
        elif d == 1:
            ax.errorbar(xdata,data,erro,linestyle=Line,marker=Mark,ms=5,label=Label)
    elif erro.all() == None:
        xdata = np.linspace(origin[0]*np.ones(1),(origin[0] + origin[1]*(data.T.shape[0] - 1))*np.ones(1),data.T.shape[0],axis= 1)
        if x.all() != None:
            xdata = x
        ax.plot(xdata.T,data.T,linestyle=Line,marker=Mark,ms=5,label=Label)
    if Xlimt != None:
        plt.xlim(Xlimt)
    if Ylimt != None:
        plt.ylim(Ylimt)
    if Label != '':
        plt.legend(loc = 0,title=Ltitle)
    plt.show()
    
def Currect_Swape(conf_name,data,Lt,swap=1):
    if conf_name in ("N450", "N304", "N305"):
        adata = Cut_conf(data, Lt)
        data = np.zeros(adata.shape)
        data[::2,:,0] = adata[::2,:,0] # correct
        data[::2,:,1] = adata[::2,:,1] # correct
        if swap==1: # odd cfg flip real and image in some situation ? reason ?
            data[1::2,:,0] = adata[1::2,:,0] 
            data[1::2,:,1] = adata[1::2,:,1]
        else:
            data[1::2,:,0] = -adata[1::2,:,1] # flip sign ?
            data[1::2,:,1] = adata[1::2,:,0]
    else:
        data = Cut_conf(data, Lt)
    return data

def Throw_Odd(conf_name,data,Lt,T=1):
    if conf_name in ("N450", "N304", "N305"):
        adata = Cut_conf(data, Lt)
        data = adata[::2,:,:]
    else:
        data = Cut_conf(data, Lt)
    return data

def Csw_dynamic(beta):
    g_square=6/beta
    csw=(1 - 0.1921*g_square - 0.1378*g_square**2 + 0.0717*g_square**3)/(1 - 0.3881*g_square)
    print("Csw: ", csw)
    return csw