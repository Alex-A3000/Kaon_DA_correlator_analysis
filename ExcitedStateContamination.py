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

# fitting model
def con_fun(x, *a):
    return a[0]

def exp_fun(x,*a):
    return a[0] + a[1]*np.exp(-1*a[2]*x)

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14,]],
    ["B452", 400, 0.137045, 0.136378, 64,  [10, 12, 14, 16]],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16]],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24]],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24]],
]

i = 2
Ntau = 12
cutdata = 2
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
tau_es = List[i][5]
R_im_odd, R_re_odd, R_im_eve, R_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
MR_im_odd, MR_re_odd, MR_im_eve, MR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
BR_im_odd, BR_re_odd, BR_im_eve, BR_re_eve = [np.zeros((len(kappa_hs), Ntau)).tolist() for _ in range(4)]
Odd_im, Even_im, Odd_re, Even_re = [np.zeros((len(kappa_hs), Ntau, len(tau_es))) for _ in range(4)]
Odd_im_b, Even_im_b, Odd_re_b, Even_re_b = [np.zeros((len(kappa_hs), Ntau, len(tau_es), 4*Nconf)) for _ in range(4)]

for i, kappa_h in enumerate(kappa_hs):
    three_pt_data = tb.hadronic_tensor(three_pt_dir, kappa_l, kappa_s, kappa_h, Nconf, Lt, tau_es, fitting_result)
    for tau in range(Ntau):
        for t_e in range(len(tau_es)):
            #flip the sign (according to the results of Excited state analysis)
            Odd_im[i][tau][t_e] = -1*three_pt_data.im_odd[t_e][0][-tau]
            Even_im[i][tau][t_e] = 1*three_pt_data.im_even[t_e][0][-tau]
            Odd_im_b[i][tau][t_e] = -1*three_pt_data.im_odd_b[t_e][0][:,-tau]
            Even_im_b[i][tau][t_e] = 1*three_pt_data.im_even_b[t_e][0][:,-tau]
            Odd_re[i][tau][t_e] = -1*three_pt_data.re_odd[t_e][0][-tau]
            Even_re[i][tau][t_e] = 1*three_pt_data.re_even[t_e][0][-tau]
            Odd_re_b[i][tau][t_e] = -1*three_pt_data.re_odd_b[t_e][0][:,-tau]
            Even_re_b[i][tau][t_e] = 1*three_pt_data.re_even_b[t_e][0][:,-tau]
        if tau >= cutdata-1:
            # R_re_odd[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Odd_re[i][tau][:], Odd_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_re_eve[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Even_re[i][tau][:], Even_re_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_im_odd[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Odd_im[i][tau][:], Odd_im_b[i][tau][:][:], bounds=bounds, p0=p0)
            # R_im_eve[i][tau] = tb.Fit_cov(exp_fun, np.array(tau_es), Even_im[i][tau][:], Even_im_b[i][tau][:][:], bounds=bounds, p0=p0)
            
            # R_re_odd[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Odd_re[i][tau][1:], Odd_re_b[i][tau][1:][:], bounds=bounds, p0=p0)
            # R_re_eve[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Even_re[i][tau][1:], Even_re_b[i][tau][1:][:], bounds=bounds, p0=p0)
            # R_im_odd[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Odd_im[i][tau][1:], Odd_im_b[i][tau][1:][:], bounds=bounds, p0=p0)
            # R_im_eve[i][tau] = tb.Fit_cov(con_fun, np.array(tau_es)[1:], Even_im[i][tau][1:], Even_im_b[i][tau][1:][:], bounds=bounds, p0=p0)

            # MR_re_odd[i][tau] = R_re_odd[i][tau].res[0]
            # MR_re_eve[i][tau] = R_re_eve[i][tau].res[0]
            # MR_im_odd[i][tau] = R_im_odd[i][tau].res[0]
            # MR_im_eve[i][tau] = R_im_eve[i][tau].res[0]
            
            MR_re_odd[i][tau] = Odd_re[i][tau][-1]
            MR_re_eve[i][tau] = Even_re[i][tau][-1]
            MR_im_odd[i][tau] = Odd_im[i][tau][-1]
            MR_im_eve[i][tau] = Even_im[i][tau][-1]
            
            # BR_re_odd[i][tau] = np.array(R_re_odd[i][tau].boots_res)[:,0]
            # BR_re_eve[i][tau] = np.array(R_re_eve[i][tau].boots_res)[:,0]
            # BR_im_odd[i][tau] = np.array(R_im_odd[i][tau].boots_res)[:,0]
            # BR_im_eve[i][tau] = np.array(R_im_eve[i][tau].boots_res)[:,0]
            
            BR_re_odd[i][tau] = Odd_re_b[i][tau][-1]
            BR_re_eve[i][tau] = Even_re_b[i][tau][-1]
            BR_im_odd[i][tau] = Odd_im_b[i][tau][-1]
            BR_im_eve[i][tau] = Even_im_b[i][tau][-1]
            
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

#     fit_even_re = tb.Fit_cov_opt(R_even_re_fun_tree, np.array(range(Ntau))[cutdata:5], np.array(MR_re_eve[h][cutdata:5]), np.array(BR_re_eve[h][cutdata:5]), maxfev=100000, p0 = [1e-1], bounds= [(1e-2,100)])
#     fit_odd_im = tb.Fit_cov_opt(R_odd_im_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_im_odd[h][cutdata:]), np.array(BR_im_odd[h][cutdata:]), maxfev=100000, p0 = [1e-5], bounds= [(1e-14,100)])
#     fit_odd_re = tb.Fit_cov_opt(R_odd_re_fun_tree, np.array(range(Ntau))[cutdata:], np.array(MR_re_odd[h][cutdata:]), np.array(BR_re_odd[h][cutdata:]), maxfev=100000, p0 = [1e-5], bounds= [(1e-14,100)])

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

    # # Save the fit results to a file
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