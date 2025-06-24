# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:42:15 2025

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
import sys

# fitting model
def con_fun(x, *a):
    return a[0]

def exp_fun(x,*a):
    return a[0] + a[1]*np.exp(-1*a[2]*x)

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14,],    0.075, 32],
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
bounds = [(0,np.inf), (0,np.inf), (0,np.inf)]
p0 = [1e-5,1e-5,1e-5]
bounds = [(0,np.inf)]
p0 = [1e-5]
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])
# load the fitting result
dir = "kaon_result/"
name = conf_name + "GEVP_OP_fitting_result.pickle"

with open(dir + name, 'rb') as f:
    fitting_result = pickle.load(f)

eigenvectors = fitting_result["eigenvector"]
OPfitting_exponential = fitting_result["exponential"]
#define the directory of input data
three_pt_dir = conf_name + "/"
kappa_hs = ["0.104000","0.115000","0.124500"]
kappa_hs = ["0.124500"]
tau_es = List[i][5]
R_im_odd, R_re_odd, R_im_eve, R_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
MR_im_odd, MR_re_odd, MR_im_eve, MR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
BR_im_odd, BR_re_odd, BR_im_eve, BR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
Odd_im, Even_im, Odd_re, Even_re = [np.zeros((len(kappa_hs), Ntau, len(tau_es))) for _ in range(4)]
Odd_im_b, Even_im_b, Odd_re_b, Even_re_b = [np.zeros((len(kappa_hs), Ntau, len(tau_es), 4*Nconf)) for _ in range(4)]
R = np.zeros((len(kappa_hs), Ntau, len(tau_es), 4))
R_b = np.zeros((len(kappa_hs), Ntau, len(tau_es), 4, 4*Nconf))
# exc = np.zeros(len(kappa_hs)).tolist()
for i, kappa_h in enumerate(kappa_hs):
    three_pt_data = tb.hadronic_tensor(three_pt_dir, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, fitting_result)
    # exc[i] = tb.hadronic_tensor_exc(con_fun, Ntau, three_pt_dir, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, fitting_result)
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
            # R_re_odd[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Odd_re[i][tau][:], Odd_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_re_eve[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Even_re[i][tau][:], Even_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_im_odd[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Odd_im[i][tau][:], Odd_im_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_im_eve[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Even_im[i][tau][:], Even_im_b[i][tau][:][:], bounds=bounds, p0=p0)
            
            # R_re_odd[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Odd_re[i][tau][1:], Odd_re_b[i][tau][1:][:], bounds=bounds, p0=p0)
            # R_re_eve[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Even_re[i][tau][1:], Even_re_b[i][tau][1:][:], bounds=bounds, p0=p0)
            R_im_odd[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Odd_im[i][tau][1:], Odd_im_b[i][tau][1:][:], bounds=bounds, p0=p0)
            R_im_eve[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Even_im[i][tau][1:], Even_im_b[i][tau][1:][:], bounds=bounds, p0=p0)

            # MR_re_odd[i][tau] = R_re_odd[i][tau].res[0]
            # MR_re_eve[i][tau] = R_re_eve[i][tau].res[0]
            MR_im_odd[i][tau] = R_im_odd[i][tau].res[0]
            MR_im_eve[i][tau] = R_im_eve[i][tau].res[0]
            
            MR_re_odd[i][tau] = (Odd_re[i][tau][-1] + Odd_re[i][tau][-2])/2
            MR_re_eve[i][tau] = (Even_re[i][tau][-1] + Even_re[i][tau][-2])/2
            # MR_im_odd[i][tau] = Odd_im[i][tau][-1]
            # MR_im_eve[i][tau] = Even_im[i][tau][-1]
            
            # BR_re_odd[i][tau] = np.array(R_re_odd[i][tau].boots_res)[:,0]
            # BR_re_eve[i][tau] = np.array(R_re_eve[i][tau].boots_res)[:,0]
            BR_im_odd[i][tau] = np.array(R_im_odd[i][tau].boots_res)[:,0]
            BR_im_eve[i][tau] = np.array(R_im_eve[i][tau].boots_res)[:,0]
            
            BR_re_odd[i][tau] = (Odd_re_b[i][tau][-1] + Odd_re_b[i][tau][-2])/2
            BR_re_eve[i][tau] = (Even_re_b[i][tau][-1] + Even_re_b[i][tau][-1])/2
            # BR_im_odd[i][tau] = Odd_im_b[i][tau][-1]
            # BR_im_eve[i][tau] = Even_im_b[i][tau][-1]
#%%         
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
        # y_fit = exc[h].im_even[j]
        # y_err = tb.Bootstrap_erro(exc[h].im_even_b, 0)[j]
        # ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='blue', alpha=0.3)
        
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
        # y_fit = -1*exc[h].re_even[j]
        # y_err = tb.Bootstrap_erro(exc[h].re_even_b, 0)[j]
        # ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='blue', alpha=0.3)
        
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
        # y_fit = exc[h].im_odd[j]
        # y_err = tb.Bootstrap_erro(exc[h].im_odd_b, 0)[j]
        # ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='blue', alpha=0.3)
        
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
        # y_fit = -1*exc[h].re_odd[j]
        # y_err = tb.Bootstrap_erro(exc[h].re_odd_b, 0)[j]
        # ax.fill_between(np.linspace(j, j+1, 10), y_fit - y_err, y_fit + y_err, color='blue', alpha=0.3)
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

Itau = 0#cutdata
r = Ntau - Itau
q_max = np.pi
#q_max = 3/fminv_to_GEV * latt_space
t_space = np.linspace(Itau, Ntau-1, r)
k_space = np.linspace(-q_max, q_max, 2*r-1)
fit_range = 9
k_space = k_space[int(len(k_space)/2+1/2)-1-fit_range:int(len(k_space)/2+1/2)+fit_range]

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
    
tb.Plot(np.array(MV_im_odd).imag,tb.Bootstrap_erro(np.array(BV_im_odd).imag,2),x=k_space*fminv_to_GEV/latt_space,Title="Im Odd",Label=kappa_hs,Xlabel="q4(GeV)")
tb.Plot(np.array(MV_im_eve).imag,tb.Bootstrap_erro(np.array(BV_im_eve).imag,2),x=k_space*fminv_to_GEV/latt_space,Title="Im Even",Label=kappa_hs,Xlabel="q4(GeV)")
tb.Plot(np.array(MV_re_odd).real,tb.Bootstrap_erro(np.array(BV_re_odd).real,2).real,x=k_space*fminv_to_GEV/latt_space,Title="Re Odd",Label=kappa_hs,Xlabel="q4(GeV)")
tb.Plot(np.array(MV_re_eve).real,tb.Bootstrap_erro(np.array(BV_re_eve).real,2).real,x=k_space*fminv_to_GEV/latt_space,Title="Re Even",Label=kappa_hs,Xlabel="q4(GeV)")
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
#%%
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

def OL_fitting_V_even_re_sub2(q4, *a):
    f_M, M_phi, mom2, sub, sub2 = a[0], a[1], a[2], a[3], a[4]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real + sub*np.sin(q4) + sub2*np.sin(2*q4)

def OL_fitting_V_even_re_sub3(q4, *a):
    f_M, M_phi, mom2, sub, sub2, sub3 = a[0], a[1], a[2], a[3], a[4], a[5]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).real + sub*np.sin(q4) + sub2*np.sin(2*q4) + sub3*np.sin(3*q4)

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

def compare_plot(q4, f_M, M_phi, mom1, mom2, mom3):
    tb.Plot([TR_fitting_V_even_im(q4, f_M, M_phi, mom2, 0), OL_fitting_V_even_im(q4, f_M, M_phi, mom2, 0)], x=q4, Line='-', Title="Im_even", Label=["tree", "one-loop"])
    tb.Plot([TR_fitting_V_even_re(q4, f_M, M_phi, mom2), OL_fitting_V_even_re(q4, f_M, M_phi, mom2)], x=q4, Line='-', Title="Re_even", Label=["tree", "one-loop"])
    tb.Plot([TR_fitting_V_odd_im(q4, f_M, M_phi, mom1, mom3, 0), OL_fitting_V_odd_im(q4, f_M, M_phi, mom2, mom3, 0)], x=q4, Line='-', Title="Im_odd", Label=["tree", "one-loop"])
    tb.Plot([TR_fitting_V_odd_re(q4, f_M, M_phi, mom2, mom3), OL_fitting_V_odd_re(q4, f_M, M_phi, mom2, mom3)], x=q4, Line='-', Title="Re_odd", Label=["tree", "one-loop"])
#%%
reset = 0
if reset==0:
    TR_fit = list(np.zeros(4))
    OL_fit = list(np.zeros(4))
    TR_fit_f = list(np.zeros(3))
    OL_fit_f = list(np.zeros(3))
    TR_fit_fs = list(np.zeros(3))
    OL_fit_fs = list(np.zeros(3))
#%%
h = 0
p0 = [1e-5,1e-5,0.1,1e-5]
bounds= [(0,100),(0,100),(0,100),(0,1)]
fit_data = np.array(MV_im_eve[h]).imag
fit_data_b = np.array(BV_im_eve[h]).imag
TR_fit[0] = tb.Fit_min(TR_fitting_V_even_im, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1000)
OL_fit[0] = tb.Fit_min(OL_fitting_V_even_im, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1000)
#%%
fit_data = np.array(MV_im_eve[h]).imag
fit_data_b = np.array(BV_im_eve[h]).imag
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{even,Im}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_even_im(k_space,*TR_fit[0].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_even_im(k_space,*OL_fit[0].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_even_im(k_space, *params) for params in TR_fit[0].boots_res])
tree_mean = TR_fitting_V_even_im(k_space,*TR_fit[0].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_even_im(k_space, *params) for params in OL_fit[0].boots_res])
one_loop_mean = OL_fitting_V_even_im(k_space,*OL_fit[0].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
# #plt.hist(np.array(TR_fit[0].boots_res)[:,2], 1000, label="tree: second moment")
# plt.hist(np.array(OL_fit[0].boots_res)[:,2], 100, label="one-loop: second moment")
# plt.legend(loc=0)
# plt.show()
#%%
p0 = list(OL_fit[0].res)[:2] + [1e-5]
bounds= [(0,100),(0,0.73),(0,100)]
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
TR_fit[1] = tb.Fit_min(TR_fitting_V_even_re, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
OL_fit[1] = tb.Fit_min(OL_fitting_V_even_re, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
#%% 
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
k_space_s = np.linspace(-q_max, q_max, 20*r+1)  
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{even,Re}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space_s, TR_fitting_V_even_re(k_space_s,*TR_fit[1].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space_s, OL_fitting_V_even_re(k_space_s,*OL_fit[1].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_even_re(k_space_s, *params) for params in TR_fit[1].boots_res])
tree_mean = TR_fitting_V_even_re(k_space_s,*TR_fit[1].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space_s, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_even_re(k_space_s, *params) for params in OL_fit[1].boots_res])
one_loop_mean = OL_fitting_V_even_re(k_space_s,*OL_fit[1].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space_s, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = list(OL_fit[0].res)[:2] + [1e-2,1e-2,1e-7]
bounds= [(0,0.05),(0.5,0.75),(0,0.1),(0,0.1),(0,1)]
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
TR_fit[2] = tb.Fit_opt(TR_fitting_V_odd_im, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1000)
OL_fit[2] = tb.Fit_opt(OL_fitting_V_odd_im, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=100000)
#%%   
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Im}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_im(k_space,*TR_fit[2].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_im(k_space,*OL_fit[2].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_im(k_space, *params) for params in TR_fit[2].boots_res])
tree_mean = TR_fitting_V_odd_im(k_space,*TR_fit[2].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_im(k_space, *params) for params in OL_fit[2].boots_res])
one_loop_mean = OL_fitting_V_odd_im(k_space,*OL_fit[2].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = list(OL_fit[0].res)[:2] + [1e-5,1e-5]
bounds= [(0,0.05),(0,0.7),(0,1),(0,1)]
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real
TR_fit[3] = tb.Fit_opt(TR_fitting_V_odd_re, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e5)
OL_fit[3] = tb.Fit_opt(OL_fitting_V_odd_re, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e5)
#%% 
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real  
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Re}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_re(k_space,*TR_fit[3].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_re(k_space,*OL_fit[3].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_re(k_space, *params) for params in TR_fit[3].boots_res])
tree_mean = TR_fitting_V_odd_re(k_space,*TR_fit[3].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_re(k_space, *params) for params in OL_fit[3].boots_res])
one_loop_mean = OL_fitting_V_odd_re(k_space,*OL_fit[3].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-5,1e-5,0.1,1e-5,1e-5]
bounds= [(0,100),(0,100),(0,100),(0,1),(-1,1)]
fit_data = np.array(MV_im_eve[h]).imag
fit_data_b = np.array(BV_im_eve[h]).imag
TR_fit_ss = tb.Fit_min(TR_fitting_V_even_im_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1000)
OL_fit_ss = tb.Fit_min(OL_fitting_V_even_im_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1000)
#%%
"""fixing fk & mpsi"""
def TR_fitting_V_even_re_fix(q4,*a):
    return TR_fitting_V_even_re(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_even_re_fix(q4,*a):
    return OL_fitting_V_even_re(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def TR_fitting_V_even_re_fix_sub(q4,*a):
    return TR_fitting_V_even_re_sub(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_even_re_fix_sub(q4,*a):
    return OL_fitting_V_even_re_sub(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def OL_fitting_V_even_re_fix_sub2(q4,*a):
    return OL_fitting_V_even_re_sub2(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def OL_fitting_V_even_re_fix_sub3(q4,*a):
    return OL_fitting_V_even_re_sub3(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def TR_fitting_V_odd_im_fix(q4,*a):
    return TR_fitting_V_odd_im(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_odd_im_fix(q4,*a):
    return OL_fitting_V_odd_im(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def TR_fitting_V_odd_im_fix_sub(q4,*a):
    return TR_fitting_V_odd_im_sub(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_odd_im_fix_sub(q4,*a):
    return OL_fitting_V_odd_im_sub(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def TR_fitting_V_odd_re_fix(q4,*a):
    return TR_fitting_V_odd_re(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_odd_re_fix(q4,*a):
    return OL_fitting_V_odd_re(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
def TR_fitting_V_odd_re_fix_sub(q4,*a):
    return TR_fitting_V_odd_re_sub(q4,TR_fit[0].res[0],TR_fit[0].res[1],*a)
def OL_fitting_V_odd_re_fix_sub(q4,*a):
    return OL_fitting_V_odd_re_sub(q4,OL_fit[0].res[0],OL_fit[0].res[1],*a)
#%%
"""by fixing fk & M_phi"""
p0 = [1e-5]
bounds= [(0,100)]
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
TR_fit_f[0] = tb.Fit_min(TR_fitting_V_even_re_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e5)
OL_fit_f[0] = tb.Fit_min(OL_fitting_V_even_re_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
#%% 
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real  
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{even,Re}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_even_re_fix(k_space,*TR_fit_f[0].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_even_re_fix(k_space,*OL_fit_f[0].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_even_re_fix(k_space, *params) for params in TR_fit_f[0].boots_res])
tree_mean = TR_fitting_V_even_re_fix(k_space,*TR_fit_f[0].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_even_re_fix(k_space, *params) for params in OL_fit_f[0].boots_res])
one_loop_mean = OL_fitting_V_even_re_fix(k_space,*OL_fit_f[0].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-10,1e-10]
bounds= [(0,100),(-1e2,1e2)]
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
TR_fit_fs[0] = tb.Fit_min(TR_fitting_V_even_re_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e5)
OL_fit_fs[0] = tb.Fit_min(OL_fitting_V_even_re_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
#%%
p0 = [1e-10,1e-10,1e-10]
bounds= [(0,100),(-1e2,1e2),(-1e2,1e2)]
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
OL_sub2 = tb.Fit_min(OL_fitting_V_even_re_fix_sub2, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
#%%
p0 = [1e-10,1e-10,1e-10,1e-10]
bounds= [(0,100),(-1e-7,1e-7),(-1e-7,1e-7),(-1e-7,1e-7)]
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real
OL_sub3 = tb.Fit_min(OL_fitting_V_even_re_fix_sub3, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=1e15)
#%% 
fit_data = np.array(MV_re_eve[h]).real
fit_data_b = np.array(BV_re_eve[h]).real  
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{even,Re,sub}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_even_re_fix_sub(k_space,*TR_fit_fs[0].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_even_re_fix_sub(k_space,*OL_fit_fs[0].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_even_re_fix_sub(k_space, *params) for params in TR_fit_fs[0].boots_res])
tree_mean = TR_fitting_V_even_re_fix_sub(k_space,*TR_fit_fs[0].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_even_re_fix_sub(k_space, *params) for params in OL_fit_fs[0].boots_res])
one_loop_mean = OL_fitting_V_even_re_fix_sub(k_space,*OL_fit_fs[0].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-5,1e-5,1e-5]
bounds= [(0,10),(0,100),(0,100)]
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
TR_fit_f[1] = tb.Fit_min(TR_fitting_V_odd_im_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
OL_fit_f[1] = tb.Fit_min(OL_fitting_V_odd_im_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
#%%   
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Im}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_im_fix(k_space,*TR_fit_f[1].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_im_fix(k_space,*OL_fit_f[1].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_im_fix(k_space, *params) for params in TR_fit_f[1].boots_res])
tree_mean = TR_fitting_V_odd_im_fix(k_space,*TR_fit_f[1].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_im_fix(k_space, *params) for params in OL_fit_f[1].boots_res])
one_loop_mean = OL_fitting_V_odd_im_fix(k_space,*OL_fit_f[1].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-5,1e-5,1e-5,1e-5]
bounds= [(0,100),(0,100),(0,100),(-100,100)]
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
TR_fit_fs[1] = tb.Fit_min(TR_fitting_V_odd_im_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
OL_fit_fs[1] = tb.Fit_min(OL_fitting_V_odd_im_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
#%%   
fit_data = np.array(MV_im_odd[h]).imag
fit_data_b = np.array(BV_im_odd[h]).imag
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Im,sub}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_im_fix_sub(k_space,*TR_fit_fs[1].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_im_fix_sub(k_space,*OL_fit_fs[1].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_im_fix_sub(k_space, *params) for params in TR_fit_fs[1].boots_res])
tree_mean = TR_fitting_V_odd_im_fix_sub(k_space,*TR_fit_fs[1].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_im_fix_sub(k_space, *params) for params in OL_fit_fs[1].boots_res])
one_loop_mean = OL_fitting_V_odd_im_fix_sub(k_space,*OL_fit_fs[1].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-5,1e-5]
bounds= [(0,100),(0,100)]
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real
TR_fit_f[2] = tb.Fit_min(TR_fitting_V_odd_re_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
OL_fit_f[2] = tb.Fit_min(OL_fitting_V_odd_re_fix, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
#%%   
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Re}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_re_fix(k_space,*TR_fit_f[2].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_re_fix(k_space,*OL_fit_f[2].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_re_fix(k_space, *params) for params in TR_fit_f[2].boots_res])
tree_mean = TR_fitting_V_odd_re_fix(k_space,*TR_fit_f[2].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_re_fix(k_space, *params) for params in OL_fit_f[2].boots_res])
one_loop_mean = OL_fitting_V_odd_re_fix(k_space,*OL_fit_f[2].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
p0 = [1e-5,1e-5,1e-5]
bounds= [(0,100),(0,100),(-100,100)]
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real
TR_fit_fs[2] = tb.Fit_min(TR_fitting_V_odd_re_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
OL_fit_fs[2] = tb.Fit_min(OL_fitting_V_odd_re_fix_sub, k_space, fit_data, fit_data_b, p0 = p0, bounds= bounds, maxfev=10000000)
#%%   
fit_data = np.array(MV_re_odd[h]).real
fit_data_b = np.array(BV_re_odd[h]).real
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$V_{{odd,Re,sub}}$ for kappa_h={kappa_hs[h]}")
ax.errorbar(k_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='', marker='s', ms=4, label="data")
ax.plot(k_space, TR_fitting_V_odd_re_fix_sub(k_space,*TR_fit_fs[2].res), linestyle='-', marker='o', ms=4, label="tree")
ax.plot(k_space, OL_fitting_V_odd_re_fix_sub(k_space,*OL_fit_fs[2].res), linestyle='-', marker='x', ms=4, label="one-loop")
# Tree-level band
tree_vals = np.array([TR_fitting_V_odd_re_fix_sub(k_space, *params) for params in TR_fit_fs[2].boots_res])
tree_mean = TR_fitting_V_odd_re_fix_sub(k_space,*TR_fit_fs[2].res)
tree_std = tb.Bootstrap_erro(tree_vals, 0)
ax.fill_between(k_space, tree_mean - tree_std, tree_mean + tree_std, color='C1', alpha=0.2, label="tree band")
# One-loop band
one_loop_vals = np.array([OL_fitting_V_odd_re_fix_sub(k_space, *params) for params in OL_fit_fs[2].boots_res])
one_loop_mean = OL_fitting_V_odd_re_fix_sub(k_space,*OL_fit_fs[2].res)
one_loop_std = tb.Bootstrap_erro(one_loop_vals, 0)
ax.fill_between(k_space, one_loop_mean - one_loop_std, one_loop_mean + one_loop_std, color='C2', alpha=0.2, label="one-loop band")
plt.legend(loc=0)
plt.show()
#%%
h1, h2 = 0, 0
# Extract values
chi_ol = OL_fit_f[h1].chi
res_ol = OL_fit_f[h1].res  # assuming res_ol = [A, B]
err_ol = tb.Bootstrap_erro(np.array(OL_fit_f[h1].boots_res), 0)  # [a, b]

chi_tr = TR_fit_f[h1].chi
res_tr = TR_fit_f[h1].res  # [A, B]
err_tr = tb.Bootstrap_erro(np.array(TR_fit_f[h1].boots_res), 0)  # [a, b]

# Format string
output = (
    f"one loop fitting_{h1}: chi2 = {chi_ol:.4f}, "
    f"A_{h2} = {res_ol[h2]:.4f} ± {err_ol[h2]:.4f}, \n"
    
    f"tree-level fitting_{h1}: chi2 = {chi_tr:.4f}, "
    f"A_{h2} = {res_tr[h2]:.4f} ± {err_tr[h2]:.4f}, "
)

print(output)
#%%
#h1, h2 = 0, 0
# Extract values
chi_ol = OL_fit_fs[h1].chi
res_ol = OL_fit_fs[h1].res  # assuming res_ol = [A, B]
err_ol = tb.Bootstrap_erro(np.array(OL_fit_fs[h1].boots_res), 0)  # [a, b]

chi_tr = TR_fit_fs[h1].chi
res_tr = TR_fit_fs[h1].res  # [A, B]
err_tr = tb.Bootstrap_erro(np.array(TR_fit_fs[h1].boots_res), 0)  # [a, b]

# Format string
output = (
    f"one loop sub fitting_{h1}: chi2 = {chi_ol:.4f}, "
    f"A_{h2} = {res_ol[h2]:.4f} ± {err_ol[h2]:.4f}, \n"
    
    f"tree-level sub fitting_{h1}: chi2 = {chi_tr:.4f}, "
    f"A_{h2} = {res_tr[h2]:.4f} ± {err_tr[h2]:.4f}, "
)

print(output)
#%%
"""Global fitting"""
q4_array = k_space
p0 = list(OL_fit[0].res)[:2] + list(OL_fit_f[1].res)[2:3] + list(OL_fit_f[0].res)[2:3] + list(OL_fit_f[1].res)[3:4] + list(OL_fit_f[0].res)[3:4] + list(OL_fit_f[2].res)[4:5]
bounds= [(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100)]
def global_chisq(params):
    f_M, M_phi, mom1, mom2, mom3, sub_odd, sub_even = params

    # Pack arguments for each component
    args_even_im = [f_M, M_phi, mom2, sub_even]
    args_even_re = [f_M, M_phi, mom2]
    args_odd_im  = [f_M, M_phi, mom1, mom3, sub_odd]
    args_odd_re  = [f_M, M_phi, mom1, mom3]

    # Model predictions
    pred_even_im = np.array([OL_fitting_V_even_im(q4, *args_even_im) for q4 in q4_array])
    pred_even_re = np.array([OL_fitting_V_even_re(q4, *args_even_re) for q4 in q4_array])
    pred_odd_im  = np.array([OL_fitting_V_odd_im(q4, *args_odd_im) for q4 in q4_array])
    pred_odd_re  = np.array([OL_fitting_V_odd_re(q4, *args_odd_re) for q4 in q4_array])

    # Compute residuals
    chi2_even_im = np.sum((MV_im_eve[h] - pred_even_im)**2 / (tb.Bootstrap_erro(BV_im_eve[h],1)**2 + sys.float_info.epsilon))
    chi2_even_re = np.sum((MV_re_eve[h] - pred_even_re)**2 / (tb.Bootstrap_erro(BV_re_eve[h],1)**2 + sys.float_info.epsilon))
    chi2_odd_im  = np.sum((MV_im_odd[h] - pred_odd_im)**2 / (tb.Bootstrap_erro(BV_im_odd[h],1)**2 + sys.float_info.epsilon))
    chi2_odd_re  = np.sum((MV_re_odd[h] - pred_odd_re)**2 / (tb.Bootstrap_erro(BV_re_odd[h],1)**2 + sys.float_info.epsilon))

    return chi2_even_im + chi2_even_re + chi2_odd_im + chi2_odd_re
#global_fit_result = opt.minimize(global_chisq, p0, method='Nelder-Mead', tol=1e-7, bounds=bounds)
#%%
"""Check OPE & higher-twist targetmass"""
run_q = 1
M_phi = OL_fit[0].res[1]
q4_a = np.linspace(-1,1,20001)
pq = 1j*q4_a*Ek - run_q*pq_3d
p_square =  Ek**2 - p_square_3d
q_square = -q4_a**2 - run_q*q_square_3d
Q2 = -q_square + M_phi**2
tau = -q_square /Q2
omega1_t = replacing_omega(1, pq, p_square, q_square, Q2)
omega1_f = replacing_omega(1, pq, p_square, q_square, Q2,replacing=False)
omega2_t = replacing_omega(2, pq, p_square, q_square, Q2)
omega2_f = replacing_omega(2, pq, p_square, q_square, Q2,replacing=False)
omega3_t = replacing_omega(3, pq, p_square, q_square, Q2)
omega3_f = replacing_omega(3, pq, p_square, q_square, Q2,replacing=False)
#%%
"""check overall factor 0.5"""
ht2 = (Cw0(Q2, mu, tau, omega2_f, 0, CF)/Q2)
ht2_n = (0.5/Q2)
ht2 = (Cw1(Q2, mu, tau, omega2_f, 0, CF)*omega1_f/Q2)
ht2_n = (0.5*(0.5**1)*omega1_f/Q2)
ht2 = (Cw2(Q2, mu, tau, 0, CF)*omega2_f*(1 - p_square*q_square /(6*pq**2))/Q2)
ht2_n = (0.5*(0.5**2)*omega2_f/Q2)
ht2 = (Cw3(Q2, mu, tau, 0, CF)*omega3_f*(1 - 3*p_square*q_square/(8*pq**2)))
ht2_n = (0.5*(0.5**3)*omega3_f)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title("(Im)")
ax.plot(q4_a, ht2.imag, label="tree")
ax.plot(q4_a, ht2_n.imag, label="one loop")
plt.xlabel("q4")
plt.legend(loc=0)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title("(Re)")
ax.plot(q4_a, ht2.real, label="tree")
ax.plot(q4_a, ht2_n.real, label="one loop")
plt.xlabel("q4")
plt.legend(loc=0)
plt.show()
#%%
# ht3 = 1*(omega3_f* (1 - 3*p_square*q_square /(8*pq**2))/Q2)
# ht3_n = 1*(omega3_f/Q2)
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
# plt.grid(color='gray', linestyle='--', linewidth=1)
# plt.title("higher-twist targetmass effects (3nd moment term & Im)")
# ax.plot(q4_a, ht3.imag, label="higher-twist")
# ax.plot(q4_a, ht3_n.imag, label="without")
# plt.xlabel("q4")
# plt.legend(loc=0)
# plt.show()
# fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
# plt.grid(color='gray', linestyle='--', linewidth=1)
# plt.title("higher-twist targetmass effects (3nd moment term & Re)")
# ax.plot(q4_a, ht3.real, label="higher-twist")
# ax.plot(q4_a, ht3_n.real, label="without")
# plt.xlabel("q4")
# plt.legend(loc=0)
# plt.show()
#%%
# """Check with RA"""
# CF = 4/3 # Nc**2 - 1 / 2*Nc
# mu = 2 /fminv_to_GEV *0.04 # renormalization scale 2GeV
# alpha_s = 0.201
# Ek = 0.286831
# Lx = 48
# p_square_3d = 4 *2 *np.pi /Lx *2 *np.pi /Lx
# q_square_3d = (1 + 1) *2 *np.pi /Lx *2 *np.pi /Lx
# q3 = -1 *2 *np.pi /Lx # q = {-1, 0, -1} * 2*pi/L
# pq_3d = 2 *2 *np.pi /Lx *2 *np.pi /Lx # p = {-2, 0, 0} * 2*pi/L

# q4 = np.linspace(-2*np.pi, 2*np.pi, 100000)
# nt_space = np.array(range(11))
# f_M =1 
# M_phi =1  
# mom2 = 0
# pq = 1j*q4*Ek - pq_3d
# p_square =  Ek**2 - p_square_3d
# q_square = -q4**2 - q_square_3d
# Q2 = -q_square + M_phi**2
# tau = -q_square /Q2
# omega2 = replacing_omega(2, pq, p_square, q_square, Q2)
# OLVE = One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF).imag
# invF_OLVE = IDFTcos(OLVE, q4, nt_space)
# (invF_OLVE/invF_OLVE[1]).real
#%%
# Ek = OPfitting_exponential[1].best_fit.res[0]
# q3 = 1 * 2 *np.pi /32
# qsquare = 1.25 * 2 *np.pi /32 * 2 *np.pi /32
# pq = 0.5 * 2 *np.pi /32 * 2 *np.pi /32

# def R_even_im_fun_tree(t, *a):
#     A = a[0] * 2 * Ek * q3
#     #Ephi = (a[1]**2 - qsquare)**0.5# Epsi**2 = mphi**2 - q**2
#     Ephi = a[1]
#     C0 = (A/2/Ephi)*np.exp(-Ephi*t)
#     return C0

# for h, kappa_h in enumerate(kappa_hs):
#     fit_even_im = tb.Fit_cov(R_even_im_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_im_eve[h][cutdata:]), np.array(BR_im_eve[h][cutdata:]), maxfev=100000, p0 = [1e-5,1e-5], bounds= [(0,100),(0,100)])
#     Ephi = fit_even_im.res[1]
#     Fk = fit_even_im.res[0]

#     def R_even_re_fun_tree(t, *a):
#         A = Fk * 2 * Ek * q3
#         chi2 = a[0]
#         C0 = (A/2/Ephi)*np.exp(-Ephi*t)
#         C2 = 2*t*Ek*pq /8 /Ephi**2 *(abs(t)*Ephi + 1)
#         return C0*(C2*chi2)

#     def R_odd_im_fun_tree(t, *a):
#         A = Fk * 2 * Ek * q3
#         chi = a[0]
#         C = (A/4/Ephi)*np.exp(-Ephi*t)
#         C1 = pq / Ephi**2 * (1 + Ephi *t)
#         return C*(C1*chi)

#     def R_odd_re_fun_tree(t, *a):
#         A = Fk * 2 * Ek * q3
#         chi = a[0]
#         C = (A/4/Ephi)*np.exp(-Ephi*t)
#         C1 = Ek * t
#         return C*(C1*chi)

#     fit_even_re = tb.Fit_cov(R_even_re_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_re_eve[h][cutdata:]), np.array(BR_re_eve[h][cutdata:]), maxfev=100000, p0 = [1e-5], bounds= [(1e-10,100)])
#     fit_odd_im = tb.Fit_cov(R_odd_im_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_im_odd[h][cutdata:]), np.array(BR_im_odd[h][cutdata:]), maxfev=100000, p0 = [1e-5], bounds= [(1e-14,100)])
#     fit_odd_re = tb.Fit_cov(R_odd_re_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_re_odd[h][cutdata:]), np.array(BR_re_odd[h][cutdata:]), maxfev=100000, p0 = [1e-5], bounds= [(1e-14,100)])

#     af = 0.075
#     beta = 5.068

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
#     plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
#     plt.title("$R_{even,Im}$" + " for kappa_h={}".format(kappa_h))
#     ax.plot(np.array(range(Ntau))[cutdata:], R_even_im_fun_tree(np.array(range(Ntau))[cutdata:], *fit_even_im.res), 'r-',
#             label='fit: \n$f_k$=%4f +- %4f MeV\n$m_\Psi$=%4f +- %4f GeV\n$\chi^2$: %4f' %(fit_even_im.res[0]/af/beta*1000,tb.Bootstrap_erro(np.array(fit_even_im.boots_res)[:,0]/af/beta*1000)
#                                                                                     ,(fit_even_im.res[1]**2 + qsquare)**0.5/af/beta,tb.Bootstrap_erro((np.array(fit_even_im.boots_res)[:,1]**2 + qsquare)**0.5/af/beta)
#                                                                                     ,fit_even_im.chi))
#     ax.errorbar(np.array(range(Ntau))[cutdata:], MR_im_eve[h][cutdata:], tb.Bootstrap_erro(BR_im_eve[h][cutdata:],1), linestyle = '', color = 'blue'
#                 , marker = 's', ms = 2, label = 'data')
#     plt.legend(loc = 0)
#     plt.savefig("tree_level_fitting/" + conf_name + "$R_{even,Im}$" + " for kappa_h={}.png".format(kappa_h))

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
#     plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
#     plt.title("$R_{even,Re}$" + " for kappa_h={}".format(kappa_h))
#     ax.plot(np.array(range(Ntau))[cutdata:], R_even_re_fun_tree(np.array(range(Ntau))[cutdata:], *fit_even_re.res), 'r-',
#             label='fit: \n$f_k$=%4f +- %4f MeV\n$m_\Psi$=%4f +- %4f GeV\n$\phi^2$=%4f +- %4f \n$\chi^2$: %4f' 
#                                                                                     %(fit_even_im.res[0]/af/beta*1000,tb.Bootstrap_erro(np.array(fit_even_im.boots_res)[:,0]/af/beta*1000)
#                                                                                     ,(fit_even_im.res[1]**2 + qsquare)**0.5/af/beta,tb.Bootstrap_erro((np.array(fit_even_im.boots_res)[:,1]**2 + qsquare)**0.5/af/beta)
#                                                                                     ,fit_even_re.res[0],tb.Bootstrap_erro(np.array(fit_even_re.boots_res)[:,0])
#                                                                                     ,fit_even_re.chi))
#     ax.errorbar(np.array(range(Ntau))[cutdata:], MR_re_eve[h][cutdata:], tb.Bootstrap_erro(BR_re_eve[h][cutdata:],1), linestyle = '', color = 'blue'
#                 , marker = 's', ms = 2, label = 'data')
#     plt.legend(loc = 0)
#     plt.savefig("tree_level_fitting/" + conf_name + "$R_{even,Re}$" + " for kappa_h={}.png".format(kappa_h))

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
#     plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
#     plt.title("$R_{odd,Im}$" + " for kappa_h={}".format(kappa_h))
#     ax.plot(np.array(range(Ntau))[cutdata:], R_odd_im_fun_tree(np.array(range(Ntau))[cutdata:], *fit_odd_im.res), 'r-',
#             label='fit: \n$f_k$=%4f +- %4f MeV\n$m_\Psi$=%4f +- %4f GeV\n$\phi$=%4f +- %4f \n$\chi^2$: %4f' 
#                                                                                     %(fit_even_im.res[0]/af/beta*1000,tb.Bootstrap_erro(np.array(fit_even_im.boots_res)[:,0]/af/beta*1000)
#                                                                                     ,(fit_even_im.res[1]**2 + qsquare)**0.5/af/beta,tb.Bootstrap_erro((np.array(fit_even_im.boots_res)[:,1]**2 + qsquare)**0.5/af/beta)
#                                                                                     ,fit_odd_im.res[0],tb.Bootstrap_erro(np.array(fit_odd_im.boots_res)[:,0])
#                                                                                     ,fit_odd_im.chi))
#     ax.errorbar(np.array(range(Ntau))[cutdata:], MR_im_odd[h][cutdata:], tb.Bootstrap_erro(BR_im_odd[h][cutdata:],1), linestyle = '', color = 'blue'
#                 , marker = 's', ms = 2, label = 'data')
#     plt.legend(loc = 0)
#     plt.savefig("tree_level_fitting/" + conf_name + "$R_{odd,Im}$" + " for kappa_h={}.png".format(kappa_h))

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi = 150)
#     plt.grid(color = 'gray', linestyle = '--', linewidth = 1)
#     plt.title("$R_{odd,Re}$" + " for kappa_h={}".format(kappa_h))
#     ax.plot(np.array(range(Ntau))[cutdata:], R_odd_re_fun_tree(np.array(range(Ntau))[cutdata:], *fit_odd_re.res), 'r-',
#             label='fit: \n$f_k$=%4f +- %4f MeV\n$m_\Psi$=%4f +- %4f GeV\n$\phi$=%4f +- %4f \n$\chi^2$: %4f' 
#                                                                                     %(fit_even_im.res[0]/af/beta*1000,tb.Bootstrap_erro(np.array(fit_even_im.boots_res)[:,0]/af/beta*1000)
#                                                                                     ,(fit_even_im.res[1]**2 + qsquare)**0.5/af/beta,tb.Bootstrap_erro((np.array(fit_even_im.boots_res)[:,1]**2 + qsquare)**0.5/af/beta)
#                                                                                     ,fit_odd_re.res[0],tb.Bootstrap_erro(np.array(fit_odd_re.boots_res)[:,0])
#                                                                                     ,fit_odd_re.chi))
#     ax.errorbar(np.array(range(Ntau))[cutdata:], MR_re_odd[h][cutdata:], tb.Bootstrap_erro(BR_re_odd[h][cutdata:],1), linestyle = '', color = 'blue'
#                 , marker = 's', ms = 2, label = 'data')
#     plt.legend(loc = 0)
#     plt.savefig("tree_level_fitting/" + conf_name + "$R_{odd,Re}$" + " for kappa_h={}.png".format(kappa_h))
    ##Save the fit results to a file
    # with open(dir + conf_name + "tree_fit_results_kappa_h_{}.txt".format(kappa_h), "w") as f:
    #     f.write("Fit results for kappa_h = {}\n".format(kappa_h))
    #     f.write("f_k = {} +/- {}\n".format(fit_even_im.res[0], tb.Bootstrap_erro(np.array(fit_even_im.boots_res)[:,0])))
    #     f.write("m_Psi = {} +/- {}\n".format((fit_even_im.res[1]**2 + qsquare)**0.5, tb.Bootstrap_erro((np.array(fit_even_im.boots_res)[:,1]**2 + qsquare)**0.5)))
    #     f.write("chi^2 = {}\n".format(fit_even_im.chi))
    #     f.write("phi^2 = {} +/- {}\n".format(fit_even_re.res[0], tb.Bootstrap_erro(np.array(fit_even_re.boots_res)[:,0])))
    #     f.write("chi^2 = {}\n".format(fit_even_re.chi))
    #     f.write("phi = {} +/- {}\n".format(fit_odd_im.res[0], tb.Bootstrap_erro(np.array(fit_odd_im.boots_res)[:,0])))
    #     f.write("chi^2 = {}\n".format(fit_odd_im.chi))
    #     f.write("phi~ = {} +/- {}\n".format(fit_odd_re.res[0], tb.Bootstrap_erro(np.array(fit_odd_re.boots_res)[:,0])))
    #     f.write("chi^2 = {}\n".format(fit_odd_re.chi))