# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:49:01 2025

@author: s9503
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import function_kaon as tb

List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14,]],
    ["B452", 400, 0.137045, 0.136378, 64,  [10, 12, 14, 16]],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16]],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24]],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24]],
]

i = 1
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])
gamma_k = ["15","7"]
gamma_h = ["ge-14-gm-13","ge-13-gm-14"]
kappa_h = ["0.104000","0.115000","0.124500"]
kaon = ["l-s","s-l"]
tau_es = List[i][5]
indir= conf_name + "/average_data/3pt_data/"
outdir = conf_name + "/"
reweightingfactors = np.loadtxt("reweightingfactors/" + conf_name + ".dat")[:4*Nconf:4,3]

###
all_p = [
    [ 0, 0,-1],
    [-1, 0, 1],
    [ 1, 0, 1],
    [ 0, 1, 1],
    [ 0,-1, 1],
]
###

def three_pt_file_name(conf_name,kappa_l,kappa_s,s_l,k_h,g_k,tau_e,g_h,p_e,p_m):
    return (conf_name + "/average_data/3pt_data/correlator-kl-"
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

three_pt_data = np.zeros((len(tau_es)
                         ,len(kappa_h)
                         ,len(gamma_k)
                         ,len(gamma_h)
                         ,len(kaon)
                         ,4 # all_p
                         ,2 # flip sign
                         ,2 # swape pe pm
                         )).tolist()

uv = ["12","21"]
subsets = np.zeros((len(gamma_h)
                    ,len(kaon)
                    ,4
                    ,2
                    ,2)).tolist()

for i_kh, k_h in enumerate(kappa_h):
    for i_te, tau_e in enumerate(tau_es):
        for i_gk, g_k in enumerate(gamma_k):
            for i_sl, s_l in enumerate(kaon):
                for i_gh, g_h in enumerate(gamma_h):
                    for i_inv, inv in enumerate([1,-1]):
                        p = inv*np.array(all_p)
                        p1 = p[0]
                        for i_p, p2 in enumerate(p[1:]):
                            for swap in range(2):
                                if swap == 0:
                                    p_e = p1
                                    p_m = p2
                                if swap == 1:
                                    p_e = p2
                                    p_m = p1
                                three_pt_data[i_te] [i_kh] [i_gk] [i_gh] [i_sl] [i_p] [i_inv] [swap] = np.loadtxt(three_pt_file_name(conf_name,kappa_l,kappa_s,s_l,k_h,g_k,tau_e,g_h,p_e,p_m))
                                subsets[i_gh] [i_sl] [i_p] [i_inv] [swap] = "uv" + uv[i_gh] + "-" + s_l + "-pe-" + str(p_e) + "-pm-" + str(p_m)

#%%
"""Check Gamma5 Hermiticity"""
def Correct_Swape(conf_name,data,Lt,p):
    # if conf_name in ("N450", "N304", "N305"):
    #     adata = tb.Cut_conf(data, Lt)
    #     data = np.zeros(adata.shape)
    #     data[::2,:,0] = adata[::2,:,0] # correct
    #     data[::2,:,1] = adata[::2,:,1] # correct
    #     data[1::2,:,0] = 1*adata[1::2,:,1]
    #     data[1::2,:,1] =  1*adata[1::2,:,0]
    #     # print(p)
    #     # if sum(p)%2 == 0: # odd cfg flip real and image in some situation ? reason ?
    #     # # situation: flip s-l to l-s, gamma matrix, pe to pm or flip sign of pe and pm ? 
    #     #     data[1::2,:,0] = adata[1::2,:,0] 
    #     #     data[1::2,:,1] = adata[1::2,:,1]
    #     # else:
    #     #     data[1::2,:,0] = -1*adata[1::2,:,1] # flip sign ? depend on p?
    #     #     data[1::2,:,1] =  1*adata[1::2,:,0]
    # else:
    #     data = tb.Cut_conf(data, Lt)
    data = tb.Cut_conf(data, Lt)
    return data
i_te = 0
tau = 0
i_kh = 0
i_gk = 0
tau_e = tau_es[i_te]
tau_m = tau_e - tau
datalist = []
errolist = []
namelist = []
data = []
plt.figure(figsize=(12, 6))
plt.title(conf_name + " Check Gamma5 Hermiticity")
for i_p in range(2):
    for swap in range(2):
        if swap == 0:
            p_e = all_p[0]
        else:    
            p_e = all_p[1+i_p]
        adata = three_pt_data[i_te] [i_kh] [i_gk] [swap] [swap] [i_p] [swap] [swap]
        data.append(Correct_Swape(conf_name, adata, Lt, p_e)[:,tau_m,:])
        datalist.append((-1)**(float(swap))*np.mean(Correct_Swape(conf_name, adata, Lt, p_e)[::]*reweightingfactors.reshape((Nconf,1,1))[::], 0)[tau_m,1])
        errolist.append(tb.Bootstrap_erro(tb.Bootstrap(Correct_Swape(conf_name, adata, Lt, p_e)[::]*reweightingfactors.reshape((Nconf,1,1))[::],4,0),0)[tau_m,1])
        namelist.append(subsets[swap] [swap] [i_p] [swap] [swap])
        # plt.hist((-1)**(float(swap))*Correct_Swape(conf_name, adata, Lt, p_e)[::,tau_m,1]*reweightingfactors, 50,
        #           label=subsets[swap] [swap] [i_p] [swap] [swap],
        #           alpha=0.5)
        plt.plot((-1)**(float(swap))*Correct_Swape(conf_name, adata, Lt, p_e)[::,tau_m,1],
                  label=subsets[swap] [swap] [i_p] [swap] [swap],
                  alpha=0.5)
plt.legend(loc=0)
plt.show()
plt.figure(figsize=(12, 6))
plt.title(conf_name + " Check Gamma5 Hermiticity")
plt.ylabel('3pt Correction Values')
plt.grid(True)
plt.errorbar(namelist, datalist, errolist, marker='o', linestyle='', color='blue')
plt.show()
#%%
"""Average 3pt Data (16 comb)"""
for swap in range(2):
    for i_kh, k_h in enumerate(kappa_h):
        for i_te, tau_e in enumerate(tau_es):
            for i_gk, g_k in enumerate(gamma_k):
                for i_sl, s_l in enumerate(kaon):
                    data = 0
                    for i_gh, g_h in enumerate(gamma_h): #2
                        for i_inv, inv in enumerate([1,-1]): #2
                            for i_p, p2 in enumerate(p[1:]): #4
                                adata = ((-1)**i_gh)*(-1*inv)*three_pt_data[i_te] [i_kh] [i_gk] [i_gh] [i_sl] [i_p] [i_inv] [swap]
                                if conf_name == 'N450':
                                    adata.pop(31)
                                data += adata
                    
                    if swap == 0:
                        np.savetxt(conf_name + "/3pt_correlator-kl-"
                                            + kappa_l + "-ks-"
                                            + kappa_s + "-" + s_l + "-kh-"
                                            + k_h + "-"
                                            + g_k + "-"
                                            + "tau_e-" + str(tau_e) + "-"
                                            + "ge-14-gm-13"
                                            + "-pe-001-pm-10-1.dat"
                                            ,data/16)
                    if swap ==1:
                        np.savetxt(conf_name + "/3pt_correlator-kl-"
                                            + kappa_l + "-ks-"
                                            + kappa_s + "-" + s_l + "-kh-"
                                            + k_h + "-"
                                            + g_k + "-"
                                            + "tau_e-" + str(tau_e) + "-"
                                            + "ge-14-gm-13"
                                            + "-pe-10-1-pm-001.dat"
                                            ,data/16)























