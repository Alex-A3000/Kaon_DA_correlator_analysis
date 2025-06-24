# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:22:50 2025

@author: s9503
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")
import function_kaon as tb
from math import  pi
from scipy import special
from numpy.lib.scimath import sqrt
import glob
import os
import matplotlib.colors as mcolors
# import scipy.optimize as opt
# from scipy import linalg
# import sys
# from sklearn.covariance import LedoitWolf
plt.rcParams.update({"text.usetex": False})  # Enable LaTeX rendering

# fitting model
def con_fun(x, *a):
    return a[0]

def exp_fun(x,*a):
    return a[0] + a[1]*np.exp(-1*a[2]*x)

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14, 16],    0.075, 32],
    ["B452", 400, 0.137045, 0.136378, 64,  [10, 12, 14, 16], 0.075, 32],
    ["N450", 279, 0.137099, 0.136353, 128, [10, 12, 14, 16], 0.075, 48],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24], 0.049, 48],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24], 0.049, 48],
]
fminv_to_GEV = 1/5.068
i = 0
latt_space = List[i][6]
Lx = List[i][7]
Ntau = 30
cutdata = 0
bounds = [(0,np.inf), (0,np.inf), (0,np.inf)]
p0 = [1e-5,1e-5,1e-5]
bounds = [(0,np.inf)]
p0 = [1e-5]
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])

Itau = 0#cutdata
r = Ntau - Itau
q_max = np.pi
#q_max = 3/fminv_to_GEV * latt_space
t_space = np.linspace(Itau, Ntau-1, r)
k_space = np.linspace(-q_max, q_max, 2*r+1)
fit_range = 9
k_space = k_space[int(len(k_space)/2+1/2)-1-fit_range:int(len(k_space)/2+1/2)+fit_range]
k_spaces = np.linspace(k_space[0], k_space[-1], 2*r + 1)

dirp = "Vuv_fittingresults/"
#%%
# load the fitting result
dir = "kaon_result/"
name = conf_name + "GEVP_OP_fitting_result.pickle"

with open(dir + name, 'rb') as f:
    fitting_result = pickle.load(f)

eigenvectors = fitting_result["eigenvector"]
OPfitting_exponential = fitting_result["exponential"]
#define the directory of input data
three_pt_dir = conf_name + "/"
kappa_hs = ["0.104000"]#,"0.115000","0.124500"]
kappa_hs = ["0.115000"]
tau_es = List[i][5]
R_im_odd, R_re_odd, R_im_eve, R_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
MR_im_odd, MR_re_odd, MR_im_eve, MR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
BR_im_odd, BR_re_odd, BR_im_eve, BR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
Odd_im, Even_im, Odd_re, Even_re = [np.zeros((len(kappa_hs), Ntau, len(tau_es))) for _ in range(4)]
Odd_im_b, Even_im_b, Odd_re_b, Even_re_b = [np.zeros((len(kappa_hs), Ntau, len(tau_es), 4*Nconf)) for _ in range(4)]
R = np.zeros((len(kappa_hs), Ntau, len(tau_es), 4))
R_b = np.zeros((len(kappa_hs), Ntau, len(tau_es), 4, 4*Nconf))
for i, kappa_h in enumerate(kappa_hs):
    three_pt_data = tb.hadronic_tensor(three_pt_dir, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, fitting_result)
    for tau in range(Ntau):
        for t_e in range(len(tau_es)):
            R[i][tau][t_e] = 1*three_pt_data.R[t_e,:,-tau]
            R_b[i][tau][t_e] = 1*three_pt_data.R_b[t_e,:,:,-tau]
            #flip the sign (according to the results of Excited state analysis)
            Odd_im[i][tau][t_e] = 1*three_pt_data.im_odd[t_e][0][-tau]
            Even_im[i][tau][t_e] = 1*three_pt_data.im_even[t_e][0][-tau]
            Odd_im_b[i][tau][t_e] = 1*three_pt_data.im_odd_b[t_e][0][:,-tau]
            Even_im_b[i][tau][t_e] = 1*three_pt_data.im_even_b[t_e][0][:,-tau]
            Odd_re[i][tau][t_e] = -1*three_pt_data.re_odd[t_e][0][-tau]
            Even_re[i][tau][t_e] = -1*three_pt_data.re_even[t_e][0][-tau]
            Odd_re_b[i][tau][t_e] = -1*three_pt_data.re_odd_b[t_e][0][:,-tau]
            Even_re_b[i][tau][t_e] = -1*three_pt_data.re_even_b[t_e][0][:,-tau]
        if tau >= cutdata-1:
            R_re_odd[i][tau] = tb.Fit_cov_AIC(con_fun, np.array(tau_es)[:], Odd_re[i][tau][:], Odd_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            R_re_eve[i][tau] = tb.Fit_cov_AIC(con_fun, np.array(tau_es)[:], Even_re[i][tau][:], Even_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            R_im_odd[i][tau] = tb.Fit_cov_AIC(con_fun, np.array(tau_es)[:], Odd_im[i][tau][:], Odd_im_b[i][tau][:][:], bounds=bounds, p0=p0)
            R_im_eve[i][tau] = tb.Fit_cov_AIC(con_fun, np.array(tau_es)[:], Even_im[i][tau][:], Even_im_b[i][tau][:][:], bounds=bounds, p0=p0)

            MR_re_odd[i][tau] = R_re_odd[i][tau].res[0]
            MR_re_eve[i][tau] = R_re_eve[i][tau].res[0]
            MR_im_odd[i][tau] = R_im_odd[i][tau].res[0]
            MR_im_eve[i][tau] = R_im_eve[i][tau].res[0]
            
            BR_re_odd[i][tau] = np.array(R_re_odd[i][tau].boots_res)[:,0]
            BR_re_eve[i][tau] = np.array(R_re_eve[i][tau].boots_res)[:,0]
            BR_im_odd[i][tau] = np.array(R_im_odd[i][tau].boots_res)[:,0]
            BR_im_eve[i][tau] = np.array(R_im_eve[i][tau].boots_res)[:,0]
#%%
plt.rcParams.update({"text.usetex": False})  # Enable LaTeX rendering
for h, kappa_h in enumerate(kappa_hs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.title(f"$R_{{even,Im}}$ for kappa_h={kappa_h}")
    for i in range(len(tau_es)):        
        ax.errorbar(np.array(range(Ntau)) + 0.1*i, Even_im[h, :, i], tb.Bootstrap_erro(Even_im_b[h, :, i, :], 1), linestyle='',
                  marker='s', ms=4)
    for j in range(cutdata, Ntau):
        y_fit = MR_im_eve[h][j]
        y_err = tb.Bootstrap_erro(BR_im_eve[h][j], 0)
        ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='red', alpha=0.3)
        
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.title(f"$R_{{even,Re}}$ for kappa_h={kappa_h}")
    for i in range(len(tau_es)):        
        ax.errorbar(np.array(range(Ntau)) + 0.1*i, Even_re[h, :, i], tb.Bootstrap_erro(Even_re_b[h, :, i, :], 1), linestyle='',
                  marker='s', ms=4)
    for j in range(cutdata, Ntau):
        y_fit = MR_re_eve[h][j]
        y_err = tb.Bootstrap_erro(BR_re_eve[h][j], 0)
        ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='red', alpha=0.3)
        
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.title(f"$R_{{odd,Im}}$ for kappa_h={kappa_h}")
    for i in range(len(tau_es)):        
        ax.errorbar(np.array(range(Ntau)) + 0.1*i, Odd_im[h, :, i], tb.Bootstrap_erro(Odd_im_b[h, :, i, :], 1), linestyle='',
                  marker='s', ms=4)
    for j in range(cutdata, Ntau):
        y_fit = MR_im_odd[h][j]
        y_err = tb.Bootstrap_erro(BR_im_odd[h][j], 0)
        ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='red', alpha=0.3)
       
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.title(f"$R_{{odd,Re}}$ for kappa_h={kappa_h}")
    for i in range(len(tau_es)):        
        ax.errorbar(np.array(range(Ntau)) + 0.1*i, Odd_re[h, :, i], tb.Bootstrap_erro(Odd_re_b[h, :, i, :], 1), linestyle='',
                  marker='s', ms=4)
    for j in range(cutdata, Ntau):
        y_fit = MR_re_odd[h][j]
        y_err = tb.Bootstrap_erro(BR_re_odd[h][j], 0)
        ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='red', alpha=0.3)
#%%
'''Fourier Transform'''
def DFTcos(data,k_space,t_space):
    Fdata = []
    for k in k_space:
        fdata = 0 + 0*1j
        for t in t_space:
            fdata += 2*1j*np.array(data[int(t)])*np.array(np.cos(k*t)) # set data is pure imag & have been /2
        Fdata.append(fdata)
    return np.array(Fdata)

def DFTsin(data,k_space,t_space):
    Fdata = []
    for k in k_space:
        fdata = 0 + 0*1j
        for t in t_space:
            fdata += 2*1j*np.array(data[int(t)])*np.array(np.sin(k*t))
        Fdata.append(fdata)
    return np.array(Fdata)*1j

def IDFTcos(fdata,k_space,t_space):
    Data = []
    for t in t_space:
        data = 0 + 0*1j
        for ik, k in enumerate(k_space):
            data += 1j*np.array(fdata[ik])*np.array(np.cos(k*t)) # set data is pure imag & have been /2
        Data.append(data)
    return np.array(Data)

def subtract(fdata, data, t_space):
    subtraction = 0
    for t in t_space:
        subtraction += 2*1j*np.array(data[int(t)])*np.array(np.cos(np.pi*t))
    return fdata - subtraction # - Fdata(k=pi)

MV_im_odd, MV_re_odd, MV_im_eve, MV_re_eve = [np.zeros((len(kappa_hs), 2*r+1)).tolist() for _ in range(4)]
BV_im_odd, BV_re_odd, BV_im_eve, BV_re_eve = [np.zeros((len(kappa_hs), 2*r+1)).tolist() for _ in range(4)]

for h, kappa_h in enumerate(kappa_hs):
    MV_im_odd[h] = DFTcos(MR_im_odd[h] , k_space, t_space)
    MV_im_eve[h] = DFTcos(MR_im_eve[h] , k_space, t_space)
    MV_re_odd[h] = DFTsin(MR_re_odd[h] , k_space, t_space)
    MV_re_eve[h] = DFTsin(MR_re_eve[h] , k_space, t_space)
    
    BV_im_odd[h] = DFTcos(BR_im_odd[h] , k_space, t_space)
    BV_im_eve[h] = DFTcos(BR_im_eve[h] , k_space, t_space)
    BV_re_odd[h] = DFTsin(BR_re_odd[h] , k_space, t_space)
    BV_re_eve[h] = DFTsin(BR_re_eve[h] , k_space, t_space)
#%%
'''One Loop Coif'''

def Cw0(Q2, mu, tau, omega2, alpha_s, CF):
    term1 = 1/2
    term2 = (alpha_s * CF) / (4 * pi) * (144 * (tau - 1) - 5 * omega2) / 48 * np.log(mu**2 / Q2)
    term3 = (alpha_s * CF) / (4 * pi) * (1 / (96 * tau**4)) * (
        48 * (8 * tau - 11) * tau**4
        - 2 * (tau - 1)**2 * (144 * tau**3 + (5 * tau * (tau + 2) - 9) * omega2) * np.log(1 - tau)
        + (tau * (2 * (13 - 6 * tau) * tau - 47) + 18) * tau * omega2
    )
    return term1 + term2 + term3

def Cw1(Q2, mu, tau, omega2, alpha_s, CF):
    term1 = 1/4
    term2 = (alpha_s * CF) / (4 * pi) * (160 * (9 * tau - 7) - 33 * omega2) / 480 * np.log(mu**2 / Q2)
    term3 = (alpha_s * CF) / (4 * pi) * (1 / (960 * tau**5)) * (
        80 * (tau * (tau * (48 * tau - 43) - 7) - 2) * tau**3
        - 2 * (tau - 1)**2 * (80 * (tau * (18 * tau + 5) + 1) * tau**2 + 
           3 * (tau * (11 * tau * (tau + 2) - 7) - 16) * omega2) * np.log(1 - tau)
        + (96 - tau * (tau * (2 * tau * (48 * tau - 77) + 163) + 102)) * tau * omega2
    )
    return term1 + term2 + term3

def Cw2(Q2, mu, tau, alpha_s, CF):
    term1 = 1/8
    term2 = (alpha_s * CF) / (4 * pi) * (108 * tau - 83) / 48 * np.log(mu**2 / Q2)
    term3 = (alpha_s * CF) / (4 * pi) * (1 / (96 * tau**4)) * (
        tau * (tau * (2 * tau * (6 * tau * (24 * tau - 17) - 17) - 17) - 18)
        - 2 * (tau - 1)**2 * (tau * (tau * (108 * tau + 47) + 22) + 9) * np.log(1 - tau)
    )
    return term1 + term2 + term3

def Cw3(Q2, mu, tau, alpha_s, CF):
    term1 = 1/16
    term2 = (alpha_s * CF) / (4 * pi) * (720 * tau - 563) / 480 * np.log(mu**2 / Q2)
    term3 = (alpha_s * CF) / (4 * pi) * (1 / (960 * tau**5)) * (
        tau * (tau * (tau * (2 * tau * (32 * tau * (30 * tau - 19) - 93) - 113) - 42) - 144)
        - 2 * (tau - 1)**2 * (tau * (tau * (tau * (720 * tau + 383) + 226) + 129) + 72) * np.log(1 - tau)
    )
    return term1 + term2 + term3

def One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF):    
    return 2*1j*Ek*q3*f_M /Q2 *(Cw0(Q2, mu, tau, omega2, alpha_s, CF) + mom2* omega2* Cw2(Q2, mu, tau, alpha_s, CF)* (1 - p_square*q_square /(6*pq**2)) )

def One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF):    
    return 2*1j*Ek*q3*f_M /Q2 *(mom1* omega1* Cw1(Q2, mu, tau, omega2, alpha_s, CF) + mom3* omega3* Cw3(Q2, mu, tau, alpha_s, CF)* (1 - 3*p_square*q_square /(8*pq**2)) )

def Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF):    
    return 2*1j*Ek*q3*f_M /Q2 *(1 + (1/2)**2 * mom2* omega2) 

def Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF):    
    return 2*1j*Ek*q3*f_M /Q2 *((1/2)* mom1* omega1 + (1/2)**3 * mom3* omega3)

def replacing_omega(order_n, pq, p_square, q_square, Q2, replacing=True):
    #C2n = special.gegenbauer(order_n, 2) # n:int, alpha:float
    #special.eval_gegenbauer
    if replacing == True:
        #return (sqrt(p_square*q_square) /Q2)**order_n *C2n(pq /sqrt(p_square*q_square)) /(order_n + 1) #/Q2
        return (sqrt(p_square*q_square) /Q2)**order_n *special.eval_gegenbauer(order_n, 2, pq /sqrt(p_square*q_square)) /(order_n + 1) #/Q2
    else:
        return (2*pq/Q2)**order_n
    
def runing_alpha_s(mu=2, Lambda_QCD=0.3, nf=4, order='LO'):
    """
    - mu: Renormalization scale in GeV 
    - Lambda_QCD: QCD scale parameter in GeV (default: 0.3 GeV)
    - nf: Number of active quark flavors (mu=2GeV: nf=4)
    - order: 'LO' or 'NLO' for leading or next-to-leading order
    """
    mu = mu *fminv_to_GEV /latt_space # to GeV
    beta0 = 11 - 2/3 * nf
    if order == 'LO':
        return 2 * np.pi / (beta0 * np.log(mu**2 / Lambda_QCD**2))
    
    elif order == 'NLO':
        beta1 = 102 - 38/3 * nf
        L = np.log(mu**2 / Lambda_QCD**2)
        return (2 * np.pi / (beta0 * L)) * (1 - (beta1 / beta0**2) * np.log(L) / L)
#%%
'''Momentum Fitting'''
CF = 4/3 # Nc**2 - 1 / 2*Nc
mu = 2 /fminv_to_GEV *latt_space # renormalization scale 2GeV
#alpha_s = runing_alpha_s(mu,order='NLO')# coupling constant 0.3
alpha_s = 0.3

Ek = OPfitting_exponential[1].best_fit.res[0]
p_square_3d = 1 *2 *np.pi /Lx *2 *np.pi /Lx
q_square_3d = (1 + 1/4) *2 *np.pi /Lx *2 *np.pi /Lx

q3 = 1 *2 *np.pi /Lx  # pe=(0,0,1) pm=(1,0,-1) -> p=(1,0,0) q=1/2(-1,0,2)
pq_3d = -0.5 *2 *np.pi /Lx *2 *np.pi /Lx # will flip sign 

def OL_fitting_V_even_im(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub

def OL_fitting_R_even_im(tau, *a):
    return DFTcos(OL_fitting_V_even_im(k_spaces, *a)*-1j, list(tau), k_spaces)
    
def OL_fitting_V_even_im_sub(q4, *a):
    f_M, M_phi, mom2, sub ,sub2 = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub + sub2*np.cos(q4)

def OL_fitting_V_even_re(q4, *a):
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real

def OL_fitting_V_even_re_sub(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real + sub*np.sin(q4)

def OL_fitting_V_odd_im(q4, *a):
    f_M, M_phi, mom1, mom3, sub = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag + sub

def OL_fitting_V_odd_im_sub(q4, *a):
    f_M, M_phi, mom1, mom3, sub, sub2 = a[0], a[1], a[2], a[3], a[4], a[5]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag + sub + sub2*np.cos(q4)

def OL_fitting_V_odd_re(q4, *a):
    f_M, M_phi, mom1, mom3 = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).real

def OL_fitting_V_odd_re_sub(q4, *a):
    f_M, M_phi, mom1, mom3, sub = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).real + sub*np.sin(q4)

def TR_fitting_V_even_im(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub

def TR_fitting_V_even_im_sub(q4, *a):
    f_M, M_phi, mom2, sub, sub2 = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub + sub2*np.cos(q4)

def TR_fitting_V_even_re(q4, *a):
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real

def TR_fitting_V_even_re_sub(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real + sub*np.sin(q4)

def TR_fitting_V_odd_im(q4, *a):
    f_M, M_phi, mom1, mom3, sub = a[0], a[1], a[2], a[3] ,a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag + sub

def TR_fitting_V_odd_im_sub(q4, *a):
    f_M, M_phi, mom1, mom3, sub, sub2 = a[0], a[1], a[2], a[3] ,a[4], a[5]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag + sub + sub2*np.cos(q4)

def TR_fitting_V_odd_re(q4, *a):
    f_M, M_phi, mom1, mom3 = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).real

def TR_fitting_V_odd_re_sub(q4, *a):
    f_M, M_phi, mom1, mom3, sub = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).real + sub*np.sin(q4)
#%%
from sklearn.covariance import LedoitWolf, OAS
from scipy.linalg import pinv
h=0
plt.rcdefaults()
fit_data_b = np.array(BV_im_eve[h]).imag
#fit_data_b = np.array(BV_re_eve[h]).real
data = fit_data_b.T  # shape: (n_samples, n_features)

# Ledoit-Wolf shrinkage
lw = LedoitWolf().fit(data)
cov_lw = lw.covariance_
lambda_lw = lw.shrinkage_

# Oracle Approximating Shrinkage
oas = OAS().fit(data)
cov_oas = oas.covariance_
lambda_oas = oas.shrinkage_

# Manual shrinkage for reference
sample_cov = np.cov(data, rowvar=False)
target = np.trace(sample_cov) / sample_cov.shape[0] * np.eye(sample_cov.shape[0])
lambda_manual = 1e-13
cov_manual = (1 - lambda_manual) * sample_cov + lambda_manual * target

eigenvalues = np.linalg.eigvalsh(sample_cov)
condition_number = np.max(eigenvalues) / np.min(eigenvalues)
print("Condition number:", condition_number)
plt.figure()
plt.title(f"Eigenvalues of Covariance Matrix \n(Condition number:{condition_number})")
plt.plot(eigenvalues, marker='o')
plt.ylabel("Eigenvalue")
plt.xlabel("Index")
plt.grid(True)
stddev = np.sqrt(np.diag(sample_cov))
corr = sample_cov / np.outer(stddev, stddev)
plt.figure()
plt.title("Correlation Matrix")
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(corr)), [f'{i}' for i in range(len(corr))])
plt.yticks(range(len(corr)), [f'{i}' for i in range(len(corr))])
plt.tight_layout()
plt.show()
inv = 1
if inv == 1:
    lambda_reg = 0# or auto-tuned
    reg_sample_cov = sample_cov + lambda_reg * np.eye(sample_cov.shape[0])
    sample_cov = pinv(reg_sample_cov)
    cov_oas = pinv(oas.covariance_)
    cov_lw = pinv(lw.covariance_)
    cov_manual = pinv(cov_manual)
    
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Sample covariance plot
im0 = axs[0, 0].imshow(sample_cov, cmap='viridis')
axs[0, 0].set_title('Sample Covariance')
axs[0, 0].set_xlabel("Index")
axs[0, 0].set_ylabel("Index")

# Ledoit-Wolf
im1 = axs[0, 1].imshow(cov_lw, cmap='viridis')
axs[0, 1].set_title(f'Ledoit-Wolf\n(shrinkage parameter = {lambda_lw:.3e})')
axs[0, 1].set_xlabel("Index")
axs[0, 1].set_ylabel("Index")

# OAS
im2 = axs[1, 0].imshow(cov_oas, cmap='viridis')
axs[1, 0].set_title(f'OAS\n(shrinkage parameter = {lambda_oas:.3e})')
axs[1, 0].set_xlabel("Index")
axs[1, 0].set_ylabel("Index")

# Manual shrinkage
im3 = axs[1, 1].imshow(cov_manual, cmap='viridis')
axs[1, 1].set_title(f'Manual Shrinkage\n(shrinkage parameter = {lambda_manual:.3e})')
axs[1, 1].set_xlabel("Index")
axs[1, 1].set_ylabel("Index")

# Shared colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im0, cax=cbar_ax, label='Value')

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.show()

# Avoid divide-by-zero with np.where
diff_lw = np.where(sample_cov != 0, (sample_cov - cov_lw) / sample_cov * 100, 0)
diff_oas = np.where(sample_cov != 0, (sample_cov - cov_oas) / sample_cov * 100, 0)
diff_manual = np.where(sample_cov != 0, (sample_cov - cov_manual) / sample_cov * 100, 0)

# Optional: Clip extreme values for display (e.g., -100% to +100%)
vmin, vmax = -0.2, 0.2

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Ledoit-Wolf %
im0 = axs[0].imshow(diff_lw, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[0].set_title(f'% Diff: Sample - Ledoit-Wolf\n(shrinkage parameter = {lambda_lw:.3e})')
axs[0].set_xlabel("Index")
axs[0].set_ylabel("Index")

# OAS %
im1 = axs[1].imshow(diff_oas, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[1].set_title(f'% Diff: Sample - OAS\n(shrinkage parameter = {lambda_oas:.3e})')
axs[1].set_xlabel("Index")
axs[1].set_ylabel("Index")

# Manual %
im2 = axs[2].imshow(diff_manual, cmap='coolwarm', vmin=vmin, vmax=vmax)
axs[2].set_title(f'% Diff: Sample - Manual\n(shrinkage parameter = {lambda_manual:.3e})')
axs[2].set_xlabel("Index")
axs[2].set_ylabel("Index")

# Shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im0, cax=cbar_ax, label='Relative Difference (%)')

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.show()
#%%
p0 = [2.91859588e-02, 1.28670149e+00, 1.01291219e-17, 6.18200958e-04]
bounds= [(1e-7,100),(0,100),(0,100),(0,100)]
half = int((len(k_space)+1)/2) 
for h, kappa_h in enumerate(kappa_hs[:1]):
    fit_data = np.array(MV_im_eve[h]).imag
    fit_data_b = np.array(BV_im_eve[h]).imag
    OL_fit = tb.Fit_cov(OL_fitting_V_even_im, k_space[:], fit_data[:], np.array(fit_data_b)[:,:], p0 = p0, bounds= bounds, maxfev=100000000, shrinkage=4e-5,mini=False)
    # OL_fit = tb.Fit_cov_AIC(OL_fitting_R_even_im, t_space[:],np.array(MR_im_eve[0])[:], np.array(BR_im_eve[0])[:,:], p0 = p0, bounds= bounds, maxfev=100000)
    M_psi = OL_fit.res[1]
    M_psi_b = OL_fit.boots_res[:,1]
    # np.savetxt("m_phi/" + conf_name + f"-{kappa_h}.dat", np.array([M_psi] + list(M_psi_b)))
    fit_data = np.array(MV_im_eve[h]).imag
    fit_data_b = np.array(BV_im_eve[h]).imag
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.title(f"$V_{{even,Im}}$ for {conf_name} kappa_h={kappa_hs[h]}")
    ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
    ax.plot(k_space, OL_fitting_V_even_im(k_space,*OL_fit.res), linestyle='', marker='x', ms=4,
            label=f"one loop\n$\chi^2$: {OL_fit.chi:.2e}" + 
            "\n$f_k$: " + f"{OL_fit.res[0]:.3e} +- {tb.Bootstrap_erro(OL_fit.boots_res[:,0]):.3e}" + 
            "\n$m_{\Psi}$: " + f"{OL_fit.res[1]:.3e} +- {tb.Bootstrap_erro(OL_fit.boots_res[:,1]):.3e}")
    plt.legend(loc=0)
    # One-loop band
    one_loop_vals = np.array([OL_fitting_V_even_im(k_spaces, *params) for params in OL_fit.boots_res])
    one_loop_mean = OL_fitting_V_even_im(k_spaces,*OL_fit.res)
    one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
    ax.plot(k_spaces, one_loop_mean, color='C1')
    ax.fill_between(k_spaces, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C1', alpha=0.2)
    plt.show()
#%%
fit_results = {}
conf_names = ["B451", "B452", "N450", "N304", "N305"]
for filepath in glob.glob("m_phi/*.dat"):
    filename = os.path.basename(filepath)
    conf_name, kappa_str = filename.replace(".dat", "").split("-")
    

    data = np.loadtxt(filepath)
    m_Psi = data[0]
    m_Psi_b = data[1:]  # Bootstrap error

    fit_results[(kappa_str, conf_name)] = {
        "m_Psi": m_Psi,
        "m_Psi_b": m_Psi_b,
    }
    
plt.rcParams.update({'font.size': 24})
# Define a color map for the configurations
conf_colors = {
    "B451": "tab:blue",
    "B452": "tab:orange",
    "N450": "tab:green",
    "N304": "tab:red",
    "N305": "tab:gray",
}

conf_markers = {
    "B451": "o",
    "B452": "s",
    "N450": "^",
    "N304": "D",
    "N305": "x",
}

shift = 2e-4
shift_back = 0.0004
# Function to adjust color based on kappa_h
def adjust_color(base_color, kappa_h, adjustment_factor=0.5):
    # Convert base color to RGB
    rgb_color = mcolors.to_rgb(base_color)
    
    # Create a slight adjustment to the color based on kappa_h
    adjustment = (kappa_hs.index(kappa_h) / len(kappa_hs)) * adjustment_factor
    
    # Adjust the RGB values
    adjusted_rgb = [min(1, max(0, c + adjustment)) for c in rgb_color]
    
    return adjusted_rgb

# Create figure
plt.figure(figsize=(16, 9))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data
        # Extract values
        m_Psi = fit_results[key]["m_Psi"]
        m_Psi_b = fit_results[key]["m_Psi_b"] 
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_m_Psi_sq = (m_Psi) ** 2  # Compute (a m_Psi)^2
        a_m_Psi_sq_err = tb.Bootstrap_erro((m_Psi_b**2))
        m_Psi_inv = 1 / m_Psi  * a / fminv_to_GEV# Compute m_Psi^{-1}
        m_Psi_inv_err = tb.Bootstrap_erro((1 / m_Psi_b) * a / fminv_to_GEV)# Error propagation
        base_color = conf_colors[conf_name]  # Base color for configuration
        color = adjust_color(base_color, kappa_h)
        # Plot
        plt.errorbar(a_m_Psi_sq, m_Psi_inv, xerr=a_m_Psi_sq_err, yerr=m_Psi_inv_err, fmt="o",
                     label=f"{conf_name},$\kappa _h$={kappa_h}",color=color, capsize=5)

# Labels and formatting
plt.xlabel("${(am_{\Psi})}^2$")
plt.ylabel("${m_{\Psi}}^{-1}$ (${GeV}^{-1}$)")
plt.title("${m_{\Psi}}^{-1}$ vs. ${(am_{\Psi})}^2$ for Different Ensemble")
# Move legend outside the plot (upper right)
plt.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout to fit the legend
plt.tight_layout()
plt.grid()
# Show plot
plt.show()
#%%
plt.rcParams.update({'font.size': 24})
# Create figure
plt.figure(figsize=(16, 9))
pa = []
# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data
        # Extract values
        m_Psi = fit_results[key]["m_Psi"]
        m_Psi_b = fit_results[key]["m_Psi_b"] 
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        m_Psi = m_Psi / a * fminv_to_GEV  # Compute (a m_Psi)^2
        m_Psi_err = tb.Bootstrap_erro((m_Psi_b)/ a * fminv_to_GEV)
        m_Psi_inv = 1 / m_Psi  * a / fminv_to_GEV# Compute m_Psi^{-1}
        m_Psi_inv_err = tb.Bootstrap_erro((1 / m_Psi_b) * a / fminv_to_GEV)# Error propagation
        base_color = conf_colors[conf_name]  # Base color for configuration
        color = adjust_color(base_color, kappa_h)
        # Plot
        plt.errorbar(float(kappa_h)**-1, m_Psi, yerr= m_Psi_err, fmt="o",
                     label=f"{conf_name},$\kappa _h$={kappa_h}",color=color, capsize=5)
        if a not in pa:
            y_val = 1 / a * fminv_to_GEV  # Convert to GeV
            plt.axhline(y=y_val, color=color, linestyle='--', linewidth=1.5)
            plt.text(x=plt.xlim()[0], y=y_val + 0.05, s=f'$a m_\\Psi = 1$ (a = {a} fm)', fontsize=20, color=color)
            pa.append(a)
# Labels and formatting
plt.xlabel("$\kappa_h ^{-1}$")
plt.ylabel("$ m_{\Psi}(GeV)$")
plt.title("$m_{\Psi}(GeV)$ vs. $\kappa_h ^{-1}$ for Different Ensemble")
# Move legend outside the plot (upper right)
# plt.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Adjust layout to fit the legend
plt.tight_layout()
plt.grid()
# Store points per conf_name for extrapolation
fit_points = {conf_name: {'x': [], 'y': [], 'color': conf_colors[conf_name]} for conf_name in conf_names}

# Collect data
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue
        a = List[conf_names.index(conf_name)][6]
        m_Psi = fit_results[key]["m_Psi"] / a * fminv_to_GEV
        x = float(kappa_h) ** -1
        fit_points[conf_name]['x'].append(x)
        fit_points[conf_name]['y'].append(m_Psi)

# Linear fit and plot
for conf_name in conf_names:
    xvals = np.array(fit_points[conf_name]['x'])
    yvals = np.array(fit_points[conf_name]['y'])
    a = List[conf_names.index(conf_name)][6]
    coeffs = np.polyfit(xvals[1:], yvals[1:], 1)
    fit_fn = np.poly1d(coeffs)
    x_fit = np.linspace(min(xvals) - 0.8, max(xvals) + 0.2, 100)
    y_fit = fit_fn(x_fit)
    plt.plot(x_fit, y_fit, color=fit_points[conf_name]['color'], linewidth=2)
    plt.text(x=x_fit[-10], y=yvals[0] + 0.5, s=f'a = {a} fm', fontsize=20, color="black")

kappa_h1 = (0.115 + 0.1245)/2
plt.axvline(x=kappa_h1**-1, color="C0", linestyle='-', linewidth=1.5, label=f'$\kappa_h ^{-1}$ = {kappa_h1**-1})')
plt.text(x=kappa_h1**-1 + 0.05, y=plt.ylim()[0] + 0.1, s=f'$\kappa_h$ = {kappa_h1: .2e}', fontsize=20, color="C0")
kappa_h2 = 0.1245 - (0.115 - 0.1245)
plt.axvline(x=kappa_h2**-1, color="C3", linestyle='-', linewidth=1.5, label=f'$\kappa_h ^{-1}$ = {kappa_h2**-1})')
plt.text(x=kappa_h2**-1 + 0.05, y=plt.ylim()[0] + 0.1, s=f'$\kappa_h$ = {kappa_h2: .2e}', fontsize=20, color="C3")
plt.show()