# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:42:19 2025

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
input("Press Enter to continue...")
# fitting model
def con_fun(x, *a):
    return a[0]

def exp_fun(x,*a):
    return a[0] + a[1]*np.exp(-1*a[2]*x)

def ExcitedStateContamination(x,data,data_b,tau):
    bounds = [(0,np.inf)]
    p0 = [1e-5]
    return tb.Fit_cov_AIC(con_fun, x, data, data_b, bounds=bounds, p0=p0, min_subset_size=2)

def runing_alpha_s(mu=2, Lambda_QCD=0.3, nf=4, order='NNLO'):
    """
    - mu: Renormalization scale in GeV 
    - Lambda_QCD (default: 0.3 GeV)
    - nf: Number of active quark flavors (mu=2GeV: nf=4)
    """
    L = np.log(mu**2 / Lambda_QCD**2)
    
    beta0 = 11 - 2/3 * nf
    beta1 = 102 - 38/3 * nf
    beta2 = (2857/2) - (5033/18)*nf + (325/54)*nf**2
    
    if order == 'LO':
        return 2 * np.pi / (beta0 * L)
    
    elif order == 'NLO':
        return (2 * np.pi / (beta0 * L)) * (1 - (beta1 / beta0**2) * np.log(L) / L)
    
    elif order == 'NNLO':
        term1 = 1
        term2 = -(beta1 / beta0**2) * np.log(L) / L
        term3 = (1 / (beta0**2 * L**2)) * (
            (beta1 / beta0)**2 * (np.log(L)**2 - np.log(L) - 1) + beta2 / beta0
        )
        return (2 * np.pi / (beta0 * L)) * (term1 + term2 + term3)
    
List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14, 16], 0.075, 32],
    ["B452", 400, 0.137045, 0.136378, 64,  [4, 6, 8, 10, 12, 14, 16], 0.075, 32],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16], 0.075, 48],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24], 0.049, 48],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24], 0.049, 48],
]
i = 1
Ntau=21
kappa_hs = ["0.124500"]
fminv_to_GEV = 1/5.068
latt_space = List[i][6]
Lx = List[i][7]
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])
t_space = np.linspace(0,Ntau-1,Ntau)
t_spaces = np.linspace(0,Ntau-1,10*Ntau)
tau_es = np.array(List[i][5])

dir = "kaon_result/"
name = conf_name + "GEVP_OP_fitting_result.pickle"

with open(dir + name, 'rb') as f:
    fitting_result = pickle.load(f)

eigenvectors = fitting_result["eigenvector"]
OPfitting_exponential = fitting_result["exponential"]
three_pt_dir = conf_name + "/"

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
            Even_im[i][tau][t_e] = 1*three_pt_data.im_even[t_e][0][-tau]
            Even_im_b[i][tau][t_e] = 1*three_pt_data.im_even_b[t_e][0][:,-tau]
        R_im_eve[i][tau] = ExcitedStateContamination(tau_es, Even_im[i][tau][:], Even_im_b[i][tau][:][:], tau)
        MR_im_eve[i][tau] = R_im_eve[i][tau].res[0]
        BR_im_eve[i][tau] = np.array(R_im_eve[i][tau].boots_res)[:,0]
            
#%%
from scipy.integrate import quad
from joblib import Parallel, delayed
'''Momentum Space Fitting form'''
CF = 4/3 # Nc**2 - 1 / 2*Nc
mu = 2 /fminv_to_GEV *latt_space # renormalization scale 2GeV
alpha_s = runing_alpha_s(2.5,order='NNLO')
alpha_s = 0.3
n_jobs = 4

Ek = OPfitting_exponential[1].best_fit.res[0]
p_square_3d = 1 *2 *np.pi /Lx *2 *np.pi /Lx
q_square_3d = (1 + 1/4) *2 *np.pi /Lx *2 *np.pi /Lx

q3 = 1 *2 *np.pi /Lx  # pe=(0,0,1) pm=(1,0,-1) -> p=(1,0,0) q=1/2(-1,0,2)
pq_3d = -0.5 *2 *np.pi /Lx *2 *np.pi /Lx # will flip sign 

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
    
def runing_alpha_s(mu=2, Lambda_QCD=0.3, nf=4, order='NNLO'):
    """
    - mu: Renormalization scale in GeV 
    - Lambda_QCD (default: 0.3 GeV)
    - nf: Number of active quark flavors (mu=2GeV: nf=4)
    """
    L = np.log(mu**2 / Lambda_QCD**2)
    
    beta0 = 11 - 2/3 * nf
    beta1 = 102 - 38/3 * nf
    beta2 = (2857/2) - (5033/18)*nf + (325/54)*nf**2
    
    if order == 'LO':
        return 2 * np.pi / (beta0 * L)
    
    elif order == 'NLO':
        return (2 * np.pi / (beta0 * L)) * (1 - (beta1 / beta0**2) * np.log(L) / L)
    
    elif order == 'NNLO':
        term1 = 1
        term2 = -(beta1 / beta0**2) * np.log(L) / L
        term3 = (1 / (beta0**2 * L**2)) * (
            (beta1 / beta0)**2 * (np.log(L)**2 - np.log(L) - 1) + beta2 / beta0
        )
        return (2 * np.pi / (beta0 * L)) * (term1 + term2 + term3)
    
"""Time Momentum Space Fitting form"""

def TR_fourier_R_even_im(t, *a):
    f_M, M_phi = a[0], a[1]
    A = 2*Ek*q3*f_M
    phi = (q_square_3d + M_phi**2)**0.5
    C0 = (A/2/phi)*np.exp(-phi*t)
    return C0

def OL_fitting_V_q_even(q4, *a):
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF)

def OL_fitting_V_q_even_q4_even(q4, *a):
    return (OL_fitting_V_q_even(q4, *a) + OL_fitting_V_q_even(-q4, *a)) / 2

def OL_fitting_V_q_even_q4_odd(q4, *a):
    return (OL_fitting_V_q_even(q4, *a) - OL_fitting_V_q_even(-q4, *a)) / 2

def OL_fitting_V_q_odd(q4, *a):
    f_M, M_phi, mom1, mom3 = a[0], a[1], a[2], a[3]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega1 = replacing_omega(1, pq, p_square, q_square, Q2, replacing=False)
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    omega3 = replacing_omega(3, pq, p_square, q_square, Q2, replacing=False)
    return One_Loop_V_odd(f_M, mom1, mom3, q3, pq, p_square, q_square, Q2, mu, tau, omega1, omega2, omega3, alpha_s, CF)

def OL_fitting_V_q_odd_q4_even(q4, *a):
    return (OL_fitting_V_q_odd(q4, *a) + OL_fitting_V_q_odd(-q4, *a)) / 2

def OL_fitting_V_q_odd_q4_odd(q4, *a):
    return (OL_fitting_V_q_odd(q4, *a) - OL_fitting_V_q_odd(-q4, *a)) / 2

def TR_fitting_V_q_even(q4, *a):
    f_M, M_phi, mom2 = a[0], a[1], a[2]
    pq = 1j*q4*Ek - pq_3d
    p_square =  Ek**2 - p_square_3d
    q_square = -q4**2 - q_square_3d
    Q2 = -q_square + M_phi**2
    tau = -q_square /Q2
    omega2 = replacing_omega(2, pq, p_square, q_square, Q2, replacing=False)
    return Tree_Level_V_even(f_M, mom2, q3, pq, p_square, q_square, Q2, mu, tau, omega2, alpha_s, CF)

def TR_fitting_V_q_even_q4_even(q4, *a):
    return (TR_fitting_V_q_even(q4, *a) + TR_fitting_V_q_even(-q4, *a)) / 2

quad_limit = 10000

def complex_integral(f, a, b, **kwargs):
    real_part = quad(lambda x: np.real(f(x)), a, b, **kwargs)[0]
    imag_part = quad(lambda x: np.imag(f(x)), a, b, **kwargs)[0]
    return real_part + 1j * imag_part

# === FULL scalar integrand ===
def C_Integrand_scalar(tau, Q, fun, *a):
    int_limit_1 = Q
    int_limit_2 = np.inf

    def integrand_1(q4):
        return np.exp(1j * q4 * tau) * fun(q4, *a)

    result_1 = complex_integral(integrand_1, -int_limit_1, int_limit_1, limit=quad_limit)

    def integrand_2(q4):
        return np.exp(-q4 * tau) * fun(Q + 1j * q4, *a)

    def integrand_3(q4):
        return np.exp(-q4 * tau) * fun(-Q + 1j * q4, *a)

    result_2 = complex_integral(integrand_2, 0, int_limit_2, limit=quad_limit)
    result_3 = complex_integral(integrand_3, 0, int_limit_2, limit=quad_limit)

    return (result_1 + 1j * np.exp(1j * Q * tau) * result_2 - 1j * np.exp(-1j * Q * tau) * result_3) / (2 * np.pi)

def C_Integrand(taus, Q, fun, *a):
    results = Parallel(n_jobs=n_jobs)(
        delayed(C_Integrand_scalar)(tau, Q, fun, *a) for tau in taus
    )
    return np.array(results)

#%%
h=0
fit_data = np.array(MR_im_eve[h])
fit_data_b = np.array(BR_im_eve[h])
p0 = [1e-5, 1e-5]
bounds= [(0,100),(0,100)]
tree_level_res = tb.Fit_cov_AIC(TR_fourier_R_even_im, t_space, fit_data, fit_data_b, p0=p0, bounds=bounds)
#%%
h=0
#input("Press Enter to continue...")
ini, fin = 1, 20
fit_data = np.array(MR_im_eve[h])
fit_data_b = np.array(BR_im_eve[h])
def R_q_even_q4_even(taus,*a):
    return C_Integrand(taus, 0.25, OL_fitting_V_q_even_q4_even, *a).imag
p0 = [0.01, 0.01, 1e-7,]
bounds= [(0,100),(0,100),(0,0.25),]
one_loop_res = tb.Fit_cov_AIC(R_q_even_q4_even, t_space[ini:fin], fit_data[ini:fin], fit_data_b[ini:fin], p0=p0, bounds=bounds,boots=False)
with open(f"one_loop_Ruv_fit/one_loop_Ruv_q-even_q4-even_fitting-AIC_{conf_name}_{kappa_hs[h]}_mu{mu}_as{alpha_s}.pkl", "wb") as f:
    pickle.dump(one_loop_res, f)
#%%
# Example data (replace with your actual data)
with open(f"one_loop_Ruv_fit/one_loop_Ruv_q-even_q4-even_fitting-AIC_{conf_name}_{kappa_hs[h]}_mu{mu}_as{alpha_s}.pkl", "rb") as f:
    fit = pickle.load(f)
subset_index = list(range(np.array(fit.all_results)[:,0].shape[0]))
fit_result = [np.array(fit.all_results)[:,0]]  # your fit result data, len=400
chi_squared = [fit.all_chi_square]  # your chi^2 data, len=400

fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis (left): fit result
ax1.scatter(subset_index, fit_result, marker='s', s=10, color='tab:blue', label='Fit Result (Time-Momentum space)')
ax1.set_ylabel('Fit Result', color='tab:blue', fontsize=18)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)

plt.legend(loc=0)
# Second y-axis (right): chi^2
ax2 = ax1.twinx()
#ax2.scatter(subset_index, chi_squared, marker='x', s=10, color='tab:red', label='Chi²')
ax2.bar(subset_index, np.array(chi_squared)[0], width=1.0, color='tab:red', alpha=0.2)
ax2.set_ylabel('Chi²', color='tab:red', fontsize=18)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylim(0, 2)
# Shared x-axis
ax1.set_xlabel('Subset Index', fontsize=18)
plt.title('Fit Result and Chi² vs Subset Index', fontsize=18)
ax1.tick_params(axis='x', labelsize=14)
fig.tight_layout()

plt.show()
#%%
fit_data = np.array(MR_im_eve[h])
fit_data_b = np.array(BR_im_eve[h])
fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)
plt.grid(color='gray', linestyle='--', linewidth=1)
plt.title(f"$R_{{even,Im}}$ for {conf_name} kappa_h={kappa_hs[h]}")
ax.errorbar(t_space, fit_data, tb.Bootstrap_erro(fit_data_b,1), linestyle='-', marker='s', ms=4, label="data")
ax.plot(t_space, R_q_even_q4_even(t_space,*fit.res), linestyle='-', marker='o', ms=4, label="one loop (NUMERICALINTEGRATION)")
plt.legend(loc=0)






