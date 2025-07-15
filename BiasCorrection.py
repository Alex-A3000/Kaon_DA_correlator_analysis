# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 22:29:39 2025

@author: s9503
"""

import numpy as np
import matplotlib.pyplot as plt
import function_kaon as tb
from scipy.optimize import minimize_scalar
import scipy.optimize as opt

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14, 16], 0.075, 32],
    ["B452", 400, 0.137045, 0.136378, 64,  [4, 6, 8, 10, 12, 14, 16], 0.075, 32],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16], 0.075, 48],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24], 0.049, 48],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24], 0.049, 48],
]
fminv_to_GEV = 1/5.068
i = 1
latt_space = List[i][6]
Lx = List[i][7]
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])
tau_es = np.array(List[i][5])
kappa_hs = ["0.104000","0.115000","0.124500"]
kaon = ["l-s","s-l"]
gamma_k = ["15","7"]

p1 = [0, 0, 1]
p2 = [1, 0,-1]

# === Load data ===
dirs = ["all-source-combined", "one-source-combined", "others-source-combined"]
labels = ["All 32", "One source only", "Remaining 31 avg"]
three_pt_data = np.zeros((len(dirs)
                         ,len(gamma_k)
                         ,len(tau_es)
                         ,len(kappa_hs)
                         ,len(kaon)
                         ,2 # swape pe pm
                         ,Nconf
                         ,Lt
                         ,2))

for i_d, d in enumerate(dirs):
    for i_kh, kappa_h in enumerate(kappa_hs):
        for i_te, tau_e in enumerate(tau_es):
            for i_gk, g_k in enumerate(gamma_k):
                for i_sl, sl in enumerate(kaon):
                    for swap in range(2):
                        if swap == 0:
                            p_e = p1
                            p_m = p2
                        if swap == 1:
                            p_e = p2
                            p_m = p1
                        
                        filename = (f"3pt_correlator-kl-{kappa_l}-ks-{kappa_s}-{sl}-kh-{kappa_h}-"
                                    f"{g_k}-tau_e-{tau_e}-ge-14-gm-13-pe-{p_e[0]}{p_e[1]}{p_e[2]}-pm-{p_m[0]}{p_m[1]}{p_m[2]}.dat")
                        path = f"{conf_name}-bias/{d}/{filename}"
                        data = np.loadtxt(path)
                        data = tb.Cut_conf(data, Lt) 
                        three_pt_data[i_d,i_gk,i_te,i_kh,i_sl,swap] = data
#%%
i_d,i_gk,i_te,i_sl,swap = 0,0,0,0,1
tau_m = 4
plt.figure(figsize=(12, 6))
plt.title("Check Correlation (kappa_h 0.104 vs 0.1245)")
plt.scatter(three_pt_data[i_d,i_gk,i_te,0,i_sl,swap,:,tau_m,1],three_pt_data[i_d,i_gk,i_te,-1,i_sl,swap,:,tau_m,1])
#%%
def Bias_Correction(data_bias_L,data_bias_S,data_true,A): 
    return A*data_bias_L + data_true - A*data_bias_S

def covariance(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    mean_A = np.mean(A,0)
    mean_B = np.mean(B,0)
    return np.mean((A - mean_A) * (B - mean_B),0)

def compute_optimal_A(O1,O1_bar,O2):
    #O1 bias O1_bar average source 
    O1 = np.asarray(O1)
    O2 = np.asarray(O2)
    delta = -O1_bar + O1
    return (covariance(delta, delta) / covariance(O2, delta))**-1

Boot = tb.Bootstrap
Berr = tb.Bootstrap_erro

t_space = np.linspace(0,Lt-1,Lt)

data_A = three_pt_data[0,i_gk,i_te,1,i_sl,swap,:,:,1]
data_B = three_pt_data[0,i_gk,i_te,-1,i_sl,swap,:,:,1]

data_A_s = three_pt_data[1,i_gk,i_te,1,i_sl,swap,:,:,1]
data_A_l = three_pt_data[2,i_gk,i_te,1,i_sl,swap,:,:,1]

data_B_s = three_pt_data[1,i_gk,i_te,-1,i_sl,swap,:,:,1]
data_B_l = three_pt_data[2,i_gk,i_te,-1,i_sl,swap,:,:,1]

AB = compute_optimal_A(data_A_s, data_A_l, data_B_s)
BA = compute_optimal_A(data_B_s, data_B_l, data_A_s)

data_A_b = Bias_Correction(data_B_l, data_B_s, data_A_s, BA)
data_B_b = Bias_Correction(data_A_l, data_A_s, data_B_s, AB)
#%%
opt_AB_list = []
x0=0
for i_t in range(Lt):    
    def var_data_B_b(OA):
        data_b = Bias_Correction(data_A_l[:,i_t], data_A_s[:,i_t], data_B_s[:,i_t], OA)
        return covariance(data_b, data_b)
    opt_AB = minimize_scalar(var_data_B_b,tol=0,method='Brent')
    print(opt_AB.x)
    x0 = opt_AB.x
    opt_AB_list.append(opt_AB.x)
opt_AB_list = np.array(opt_AB_list).ravel()

x0=0
opt_BA_list = []
for i_t in range(Lt):    
    def var_data_A_b(OA):
        data_b = Bias_Correction(data_B_l[:,i_t], data_B_s[:,i_t], data_A_s[:,i_t], OA)
        return covariance(data_b, data_b)
    opt_BA = minimize_scalar(var_data_A_b,tol=0,method='Brent')
    print(opt_BA.x)
    x0 = opt_BA.x
    opt_BA_list.append(opt_BA.x)
opt_BA_list = np.array(opt_BA_list).ravel()     
  
data_A_b_opt = Bias_Correction(data_B_l, data_B_s, data_A_s, opt_BA_list)
data_B_b_opt = Bias_Correction(data_A_l, data_A_s, data_B_s, opt_AB_list) 
#%%    
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.rcParams.update({"text.usetex": False})
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title("BiasCorrection (kappa_h 0.104 vs 0.1245)")
 
 
# ax.errorbar(t_space,np.mean(data_A,0),
#             Berr(Boot(data_A,4)),
#             linestyle='', marker='s', ms=4, label="ALL 0.104") 
# ax.errorbar(t_space+0.1,np.mean(data_A_s,0),
#             Berr(Boot(data_A_s,4)),
#             linestyle='', marker='x', ms=4, label="ONE 0.104")  
# ax.errorbar(t_space+0.3,np.mean(data_A_b_opt,0),
#             Berr(Boot(data_A_b_opt,4)),
#             linestyle='', marker='o', ms=4, label="OPT 0.104")
# ax.errorbar(t_space+0.2,np.mean(data_A_b,0),
#             Berr(Boot(data_A_b,4)),
#             linestyle='', marker='o', ms=4, label="BC 0.104") 
ax.errorbar(t_space,np.mean(data_B,0),
            Berr(Boot(data_B,4)),
            linestyle='', marker='s', ms=4, label="ALL 0.1245")
ax.errorbar(t_space+0.1,np.mean(data_B_s,0),
            Berr(Boot(data_B_s,4)),
            linestyle='', marker='x', ms=4, label="ONE 0.1245")
ax.errorbar(t_space+0.3,np.mean(data_B_b_opt,0),
            Berr(Boot(data_B_b_opt,4)),
            linestyle='', marker='o', ms=4, label="OPT 0.1245")  
ax.errorbar(t_space+0.2,np.mean(data_B_b,0),
            Berr(Boot(data_B_b,4)),
            linestyle='', marker='o', ms=4, label="BC 0.1245") 

plt.legend(loc=0)   
#%%
delta = data_B - data_B_b_opt
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.rcParams.update({"text.usetex": False})
plt.grid(color='gray', linestyle='--', linewidth=1)
ax.errorbar(t_space,np.mean(delta,0)/np.mean(data_B,0),
            Berr(Boot(delta,4)/Boot(data_B,4)),
            linestyle='', marker='o', ms=4)
plt.ylim([-1,1])  
#%%
delta = data_B_b
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.rcParams.update({"text.usetex": False})
plt.grid(color='gray', linestyle='--', linewidth=1)
ax.plot(t_space,(Berr(Boot(data_B_b,4))/Berr(Boot(data_B,4)))**2,
            linestyle='', marker='o', ms=4)
#%%
# def General_Bias_Correction(data_true, bias_deltas, *A):
#     corrected = np.copy(data_true)
#     for i in range(len(bias_deltas)):
#         corrected += A[0][i] * bias_deltas[i]
#     return corrected

def General_Bias_Correction(data_true, bias_deltas, A):
    corrected = np.copy(data_true)
    corrected += A[0] * bias_deltas[0] + A[1] * bias_deltas[1]
    return corrected

data_true = three_pt_data[0,i_gk,i_te,-1,i_sl,swap,:,:,1]
data_true_s = three_pt_data[1,i_gk,i_te,-1,i_sl,swap,:,:,1]
data_true_l = three_pt_data[2,i_gk,i_te,-1,i_sl,swap,:,:,1]

bias_deltas = []
bias_deltas.append(three_pt_data[2,i_gk,i_te,0,i_sl,swap,:,:,1]
                        - three_pt_data[1,i_gk,i_te,0,i_sl,swap,:,:,1])
bias_deltas.append(three_pt_data[2,i_gk,i_te,1,i_sl,swap,:,:,1]
                        - three_pt_data[1,i_gk,i_te,1,i_sl,swap,:,:,1])
bias_deltas = np.array(bias_deltas)    
opt_A = []
A0 = 0, 0
for i_t in range(Lt):
    def variance_of_correction(A):
        corrected = General_Bias_Correction(data_true_s[:,i_t], bias_deltas[:,:,i_t], A)
        return covariance(corrected,corrected)
    result = opt.minimize(variance_of_correction, x0=A0, tol=0, method='Powell')
    print(result.x)
    opt_A.append(result.x)
    A0 = result.x
opt_A = np.array(opt_A).ravel()
data_true_opt = General_Bias_Correction(data_true_s,bias_deltas,opt_A)
#%%    
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.rcParams.update({"text.usetex": False})
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title("BiasCorrection")
 
 
ax.errorbar(t_space,np.mean(data_true,0),
            Berr(Boot(data_true,4)),
            linestyle='', marker='s', ms=4, label="ALL 0.1245")
ax.errorbar(t_space+0.1,np.mean(data_B_s,0),
            Berr(Boot(data_true_s,4)),
            linestyle='', marker='x', ms=4, label="ONE 0.1245")
ax.errorbar(t_space+0.2,np.mean(data_B_b_opt,0),
            Berr(Boot(data_B_b_opt,4)),
            linestyle='', marker='o', ms=4, label="BC 0.1245")  
ax.errorbar(t_space+0.3,np.mean(data_true_opt,0),
            Berr(Boot(data_true_opt,4)),
            linestyle='', marker='o', ms=4, label="BC2 0.1245") 
    
plt.legend(loc=0) 