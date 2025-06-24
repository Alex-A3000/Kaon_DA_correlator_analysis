import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
import warnings
import random
import math
import pickle
import os
from scipy import linalg
from multiprocessing import Pool, freeze_support
warnings.filterwarnings("ignore")
import function_kaon as tb

#def Parameters and Constants
List = [
    ["B451", 500, 0.136981, 0.136409, 64 ],
    ["B452", 400, 0.137045, 0.136378, 64 ],
    ["N450", 280, 0.137099, 0.136353, 128],
    ["N304", 420, 0.137079, 0.136665, 128],
    ["N305", 250, 0.137025, 0.136676, 128],
]

i = 0
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_1 = str(List[i][2])
kappa_2 = str(List[i][3])

#GEVP analize
moment = ["000","100"]
gs = "-gs90-gs90"
two_pt_dir = conf_name + "/average_data/2pt_data/"
data_m, data_b, data, gevp_vel, gevp_vel_b, gevp_vec = ([[] for _ in range(len(moment))] for _ in range(6))
t0 = 0
xdata = np.linspace(t0,int(Lt/2),int(Lt/2+1 - t0))

if __name__=="__main__":
    freeze_support()
    for m, mom in enumerate(moment):
        base_path = two_pt_dir + "correlator-"
        suffix = f"-{kappa_1}-{kappa_2}-{mom}{gs}.dat"
        filename = [            
            f"{base_path}7-7{suffix}",
            f"{base_path}7-15{suffix}",
            f"{base_path}15-7{suffix}",
            f"{base_path}15-15{suffix}",
        ]

        for f, name in enumerate(filename):
            adata = tb.Cut_conf(np.loadtxt(name), Lt)[:Nconf, :int(Lt/2+1), 0]
            if f == 1 or f == 2:
                sign = -1
            else:
                sign = 1
            #adata = np.swapaxes(tb.Fold_data(np.swapaxes(adata, 0, 1), sign), 0, 1)
            data[m].append(adata)

        data_m[m] = np.array([np.mean(d, axis=0) for d in data[m]]).reshape(2, 2, int(Lt/2+1))
        data_b[m] = np.array([tb.Bootstrap(d, 4, 0) for d in data[m]]).reshape(2, 2, 4 * Nconf, int(Lt/2+1))

        # Flip sign
        data_m[m][0,0] = -1*data_m[m][0,0]
        data_m[m][0,1] = -1*data_m[m][0,1]
        data_b[m][0,0] = -1*data_b[m][0,0]
        data_b[m][0,1] = -1*data_b[m][0,1]

        # Swap axes
        data_m[m] = np.swapaxes(data_m[m], 0, 2)
        data_m[m] = np.swapaxes(data_m[m], 1, 2)
        data_b[m] = np.swapaxes(data_b[m], 0, 3)
        data_b[m] = np.swapaxes(data_b[m], 1, 2)
        data_b[m] = np.swapaxes(data_b[m], 2, 3)

        # GEVP results
        gevp_res = tb.GEVP(data_m[m], data_b[m], [0, 1], t0 = t0)
        gevp_vel[m], gevp_vel_b[m], gevp_vec[m] = gevp_res.vel, gevp_res.vel_bs, gevp_res.vec

    # fitting model
    def con_fun(x, *a):
        return a[0]

    def exp_fun(x,*a):
        """
        Exponential decay function for fitting.

        Parameters
        ----------
        x : array_like
            The time slices of the correlator.
        *a : array_like
            The parameters of the function. The first parameter is the energy,
            the second parameter is the amplitude.

        Returns
        -------
        array_like
            The fitted exponential decay function.
        """
        return a[1]*np.exp(-1*a[0]*x)

#%%
    # AIC fittting
    run = 1
    run2 = 1
    up1, down1 = 0, 32 
    # save the fitting result
    dir = "kaon_result/"
    name = conf_name + "GEVP_fitting_result.pickle"
    if run == 1:
        # control minimum and maximum range
        # the best fitting range is defined by AIC
        GEVPfitting_effectmass = [[],[]]
        GEVPfitting_exponential = [[],[]]
        for m in range(2):
            GEVPfitting_effectmass[m] = tb.Fit_cov_AIC(con_fun, xdata[up1:down1-1], tb.Effectmass_ln(gevp_vel[m][up1:down1,0],int(Lt/2))
                                                                    ,tb.Effectmass_ln(gevp_vel_b[m][up1:down1,:,0],int(Lt/2)),p0=[1e-5], bounds=[(0,100)])
            GEVPfitting_exponential[m] = tb.Fit_cov_AIC(exp_fun, xdata[up1:down1], gevp_vel[m][up1:down1,0], gevp_vel_b[m][up1:down1,:,0], p0=[1e-5,1e-5], bounds=[(0,100),(0,100)])
        if not os.path.exists(dir):
            os.makedirs(dir)
        fitting_result = {
        "effectmass": GEVPfitting_effectmass,
        "exponential": GEVPfitting_exponential
        }
        with open(dir + name, 'wb') as f:
            pickle.dump(fitting_result, f)

    # load the fitting result
    dir = "kaon_result/"
    dirp = "kaon_2ptplot/"
    name = conf_name + "GEVP_fitting_result.pickle"
    with open(dir + name, 'rb') as f:
        fitting_result = pickle.load(f)

    GEVPfitting_effectmass = fitting_result["effectmass"]
    GEVPfitting_exponential = fitting_result["exponential"]
#%%
    for m in range(2):
        if m == 0:
            state = "ground state"
        else:
            state = "1st_ex state"
            
        # plot Effectmass
        a, b = GEVPfitting_effectmass[m].best_range
        up, down = int(a), int(b)
        print(f'range: {up} to {down}')
        res = GEVPfitting_effectmass[m].best_fit.res[0]
        err = tb.Bootstrap_erro(np.array(GEVPfitting_effectmass[m].best_fit.boots_res)[:,0])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
        plt.title(conf_name + " GEVP Effectmass " + state,fontsize = 14)
        plt.xlabel("t")
        plt.ylabel("ln|Correlator|")
        ax.errorbar(xdata[:-1], tb.Effectmass_ln(gevp_vel[m][:,0],int(Lt/2)), tb.Bootstrap_erro(tb.Effectmass_ln(gevp_vel_b[m][:,:,0],int(Lt/2)),1)
                        , label = 'data', linestyle = '', color = 'blue', marker = 's', ms = 2)
        ax.fill_between(xdata[up:down-1], res - err,  res + err, color = 'red', alpha = 0.5
                        , label='fit:\nmass= %4f +- %4f\n$\chi^2$: %4f' %(res, err ,GEVPfitting_effectmass[m].best_fit.chi))
        plt.legend(loc = 0)
        # plt.savefig(dirp + conf_name + "GEVP_Effectmass_" + state + ".png")


        # plot exponential
        a, b = GEVPfitting_exponential[m].best_range
        up, down = int(a), int(b)
        xline = np.linspace(xdata[up],xdata[down],1000)
        fitting = GEVPfitting_exponential[m].best_fit
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
        plt.title(conf_name + " GEVP Exponential " + state,fontsize = 14)
        plt.xlabel("t")
        plt.ylabel("Correlator")
        ax.errorbar(xdata, gevp_vel[m][:,0], tb.Bootstrap_erro(gevp_vel_b[m][:,:,0],1)
                        , label = 'data', linestyle = '', color = 'blue', marker = 's', ms = 2)
        ax.plot(xline, exp_fun(xline, fitting.res[0], fitting.res[1])
            , linestyle = '-', color = 'red', linewidth = 2
            , label='fit:\nmass= %4f +- %4f \ncoeff= %4E +- %4E \n$\chi^2$: %4f' %(fitting.res[0],tb.Bootstrap_erro(np.array(fitting.boots_res)[:,0])
                                                                                   ,fitting.res[1],tb.Bootstrap_erro(np.array(fitting.boots_res)[:,1])
                                                                                   ,fitting.chi))
        plt.legend(loc = 0)
        # plt.savefig(dirp + conf_name + "GEVP_Exponential_" + state + ".png")

#%%
    #Optimized Matrix
    name = conf_name + "GEVP_OP_fitting_result.pickle"

    op_vel, op_vel_b, eigenvectors= ([[] for _ in range(len(moment))] for _ in range(3))
    for m in range(2):
        if m == 0:
            state = "ground state"
        else:
            state = "1st_ex state"
        
        a, b = GEVPfitting_exponential[m].best_range
        eigenvectors[m] = gevp_vec[m][int((a + b)/2)][:,0]
        print(f'{state} eigen vector: {eigenvectors}')
        op_res = tb.GEVP(data_m[m], data_b[m], [0, 1], vec0=eigenvectors[m], t0 = t0)
        op_vel[m], op_vel_b[m] = op_res.optimize, op_res.optimize_bs

    if run2 == 1:
        # the best fitting range is defined by AIC
        OPfitting_exponential = [[],[]]
        for m in range(2):
            OPfitting_exponential[m] = tb.Fit_cov_AIC(exp_fun, xdata[up1:down1], op_vel[m][up1:down1,0], op_vel_b[m][up1:down1,:,0], p0=[1e-5,1e-5], bounds=[(0,100),(0,100)])
        if not os.path.exists(dir):
            os.makedirs(dir)
        fitting_result = {
        "eigenvector": eigenvectors,
        "exponential": OPfitting_exponential
        }
        with open(dir + name, 'wb') as f:
            pickle.dump(fitting_result, f)

    # load the fitting result
    dir = "kaon_result/"
    name = conf_name + "GEVP_OP_fitting_result.pickle"

    with open(dir + name, 'rb') as f:
        fitting_result = pickle.load(f)

    eigenvectors = fitting_result["eigenvector"]
    OPfitting_exponential = fitting_result["exponential"]

    for m in range(2):
        if m == 0:
            state = "ground state"
        else:
            state = "1st_ex state"

        # plot exponential
        a, b = OPfitting_exponential[m].best_range
        up, down = int(a), int(b)
        xline = np.linspace(xdata[up],xdata[down],1000)
        fitting = OPfitting_exponential[m].best_fit
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
        plt.title(conf_name + " GEVP OP Exponential " + state,fontsize = 14)
        plt.xlabel("t")
        plt.ylabel("Correlator")
        ax.errorbar(xdata, op_vel[m][:,0], tb.Bootstrap_erro(op_vel_b[m][:,:,0],1)
                        , label = 'data', linestyle = '', color = 'blue', marker = 's', ms = 2)
        ax.plot(xline, exp_fun(xline, fitting.res[0], fitting.res[1])
            , linestyle = '-', color = 'red', linewidth = 2
            , label='fit:\nmass= %4f +- %4f \ncoeff= %4E +- %4E \n$\chi^2$: %4f' %(fitting.res[0],tb.Bootstrap_erro(np.array(fitting.boots_res)[:,0])
                                                                                   ,fitting.res[1],tb.Bootstrap_erro(np.array(fitting.boots_res)[:,1])
                                                                                   ,fitting.chi))
        plt.legend(loc = 0)
        # plt.savefig(dirp + conf_name + "GEVP_OP_Exponential_" + state + ".png")
