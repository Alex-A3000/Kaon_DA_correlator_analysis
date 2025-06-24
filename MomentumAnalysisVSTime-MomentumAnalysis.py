# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:32:06 2025

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
import scipy.optimize as opt
from scipy import linalg
import sys
from sklearn.covariance import LedoitWolf
from scipy.linalg import pinv

# fitting model
def con_fun(x, *a):
    return a[0]

def exp_fun(x,*a):
    return a[0] + a[1]*np.exp(-1*a[2]*x)

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14, 16], 0.075, 32],
    ["B452", 400, 0.137045, 0.136378, 64,  [10, 12, 14, 16], 0.075, 32],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16], 0.075, 48],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24], 0.049, 48],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24], 0.049, 48],
]
fminv_to_GEV = 1/5.068
i = 0
latt_space = List[i][6]
Lx = List[i][7]
Ntau = 30
cutdata = 0
bounds = [(-np.inf,np.inf), (0,np.inf), (0,np.inf)]
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
t_spaces = np.linspace(Itau, Ntau-1, 2*r)

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
'''One Loop Coif MomentumSpace'''
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
    
'''Momentum Space Fitting form'''
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
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag
    
def OL_fitting_V_even_im_sub(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub

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
    f_M, M_phi, mom1, mom3 = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag

def OL_fitting_V_odd_im_sub(q4, *a):
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
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag

def TR_fitting_V_even_im_sub(q4, *a):
    f_M, M_phi, mom2, sub = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag + sub

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
    f_M, M_phi, mom1, mom3 = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF).imag

def TR_fitting_V_odd_im_sub(q4, *a):
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

"""Time Momentum Space Fitting form (integrated ver)"""
from scipy.integrate import quad
from joblib import Parallel, delayed
n_jobs = 10 # n nodes
int_limit = np.pi
quad_limit = 100

def OL_fitting_R_even_im_scalar(tau, *a): 
    integrand = lambda q4: np.cos(q4 * tau) * OL_fitting_V_even_im(q4, *a)
    result, error = quad(integrand, -int_limit, int_limit, limit=quad_limit)
    return result / (2 * np.pi)

def OL_fitting_R_even_im(taus, *a,):
    results = Parallel(n_jobs)(
        delayed(OL_fitting_R_even_im_scalar)(tau, *a) for tau in taus #speed up
    )
    return np.array(results)

def TR_fitting_R_even_im_scalar(tau, *a): 
    integrand = lambda q4: np.cos(q4 * tau) * TR_fitting_V_even_im(q4, *a)
    result, error = quad(integrand, -int_limit, int_limit, limit=quad_limit)
    return result / (2 * np.pi)

def TR_fitting_R_even_im(taus, *a,):
    results = Parallel(n_jobs)(
        delayed(TR_fitting_R_even_im_scalar)(tau, *a) for tau in taus #speed up
    )
    return np.array(results)

"""Time Momentum Space Fitting form"""

def TR_fourier_R_even_im(t, *a):
    f_M, M_phi = a[0], a[1]
    A = 2*1j*Ek*q3*f_M
    phi = q_square_3d + M_phi**2
    C0 = (A/2/phi)*np.exp(-phi*t)
    return C0.imag

#%%
reset = 0
if reset==0:
    TR_fit = list(np.zeros(4))
    OL_fit = list(np.zeros(4))
    TR_fit_R = list(np.zeros(4))
    OL_fit_R = list(np.zeros(4))
#%%
h = 0
p0 = [0.06239345, 0.72388352, 0.40647488, 0.00169941]
bounds= [(0,100),(0,100),(0,100),(0,100)]
fit_data = np.array(MV_im_eve[h]).imag
fit_data_b = np.array(BV_im_eve[h]).imag
TR_fit[0] = tb.Fit_cov(TR_fitting_V_even_im_sub, k_space, fit_data, fit_data_b, shrinkage="LW", tol=1e-27, p0=p0, bounds=bounds, maxfev=100000)
OL_fit[0] = tb.Fit_cov(OL_fitting_V_even_im_sub, k_space, fit_data, fit_data_b, shrinkage="LW", tol=1e-27, p0=p0, bounds=bounds, maxfev=100000)
#%%
plt.rcParams.update({"text.usetex": False})
fit_data = np.array(MV_im_eve[h]).imag
fit_data_b = np.array(BV_im_eve[h]).imag
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{even,Im}}$ for {conf_name} kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_even_im_sub(k_space,*TR_fit[0].res), linestyle='', marker='o', ms=4,
        label=f"tree level\n$\chi^2$: {TR_fit[0].chi:.2e}" + 
        "\n$f_k$: " + f"{TR_fit[0].res[0]:.3e} +- {tb.Bootstrap_erro(TR_fit[0].boots_res[:,0]):.3e}" + 
        "\n$m_{\Psi}$: " + f"{TR_fit[0].res[1]:.3e} +- {tb.Bootstrap_erro(TR_fit[0].boots_res[:,1]):.3e}")
ax.plot(k_space, OL_fitting_V_even_im_sub(k_space,*OL_fit[0].res), linestyle='', marker='x', ms=4,
        label=f"one loop\n$\chi^2$: {OL_fit[0].chi:.2e}" + 
        "\n$f_k$: " + f"{OL_fit[0].res[0]:.3e} +- {tb.Bootstrap_erro(OL_fit[0].boots_res[:,0]):.3e}" + 
        "\n$m_{\Psi}$: " + f"{OL_fit[0].res[1]:.3e} +- {tb.Bootstrap_erro(OL_fit[0].boots_res[:,1]):.3e}")
plt.legend(loc=0)
# Tree-level band
tree_vals = np.array([TR_fitting_V_even_im_sub(k_spaces, *params) for params in TR_fit[0].boots_res])
tree_mean = TR_fitting_V_even_im_sub(k_spaces,*TR_fit[0].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.plot(k_spaces, tree_mean, color='C1')
ax.fill_between(k_spaces, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2)
# One-loop band
one_loop_vals = np.array([OL_fitting_V_even_im_sub(k_spaces, *params) for params in OL_fit[0].boots_res])
one_loop_mean = OL_fitting_V_even_im_sub(k_spaces,*OL_fit[0].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.plot(k_spaces, one_loop_mean, color='C2')
ax.fill_between(k_spaces, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2)
plt.show()    
#%%
TR_fit_R[0] = TR_fit[0]
OL_fit_R[0] = OL_fit[0]
fit_data = np.array(MR_im_eve[h])
fit_data_b = np.array(BR_im_eve[h])
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$R_{{even,Im}}$ for {conf_name} kappa_h={kappa_hs[h]}")
ax.errorbar(t_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='-', marker='s', ms=4, label="data")
ax.plot(t_space, TR_fitting_R_even_im(t_space,*TR_fit[0].res), linestyle='', marker='o', ms=4,
        label=f"tree level\n$\chi^2$: {TR_fit[0].chi:.2e}" + 
        "\n$f_k$: " + f"{TR_fit[0].res[0]:.3e} +- {tb.Bootstrap_erro(TR_fit[0].boots_res[:,0]):.3e}" + 
        "\n$m_{\Psi}$: " + f"{TR_fit[0].res[1]:.3e} +- {tb.Bootstrap_erro(TR_fit[0].boots_res[:,1]):.3e}")
ax.plot(t_space, OL_fitting_R_even_im(t_space,*OL_fit[0].res), linestyle='', marker='x', ms=4,
        label=f"one loop\n$\chi^2$: {OL_fit[0].chi:.2e}" + 
        "\n$f_k$: " + f"{OL_fit[0].res[0]:.3e} +- {tb.Bootstrap_erro(OL_fit[0].boots_res[:,0]):.3e}" + 
        "\n$m_{\Psi}$: " + f"{OL_fit[0].res[1]:.3e} +- {tb.Bootstrap_erro(OL_fit[0].boots_res[:,1]):.3e}")
plt.legend(loc=0)
# Tree-level band
tree_vals = np.array([TR_fitting_R_even_im(t_spaces, *params) for params in TR_fit[0].boots_res])
tree_mean = TR_fitting_R_even_im(t_spaces,*TR_fit[0].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.plot(t_spaces, tree_mean, color='C1')
ax.fill_between(t_spaces, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2)
# One-loop band
one_loop_vals = np.array([OL_fitting_R_even_im(t_spaces, *params) for params in OL_fit[0].boots_res])
one_loop_mean = OL_fitting_R_even_im(t_spaces,*OL_fit[0].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.plot(t_spaces, one_loop_mean, color='C2')
ax.fill_between(t_spaces, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2)
plt.show() 