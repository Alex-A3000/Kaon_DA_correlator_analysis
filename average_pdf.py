# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:22:07 2024

@author: s9503
"""

import numpy as np
import Toolbox as tb
import matplotlib.pyplot as plt

def create_name(kappa_l,kappa_s,s_l,k_h,g_k,tau_e,g_h,p_e,p_m):
    return ("B451/average_data/3pt_data/correlator-kl-"
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
                

###
List = [
    ["B451", 500, 0.136981, 0.136409, 64, [4, 6, 8, 10, 12, 14, 16]],
    ["B452", 400, 0.137045, 0.136378, 64, [4, 6, 8, 10, 12, 14, 16]],
    ["N450", 280, 0.137099, 0.136353, 128, [10]],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24]],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24]],
]

i = 2
conf_name = List[i][0]
Lt = List[i][4]
Nconf = List[i][1]
kappa_l = str(List[i][2])
kappa_s = str(List[i][3])
gamma_k = ["15"]#,"7"]
gamma_h = ["ge-14-gm-13","ge-13-gm-14"]
kappa_h = ["0.104000"]#,"0.115000","0.124500"]
kaon = ["l-s"]#,"s-l"]
tau_es = List[i][5]
indir= conf_name + "/average_data/3pt_data/"
outdir = conf_name + "/"
###
all_p = [
    [ 0, 0,-1],
    [-1, 0, 1],
    [ 1, 0, 1],
    [ 0, 1, 1],
    [ 0,-1, 1],
]

all_p = [[ 0, 0,-1],[-1, 0, 1]]


A = []
Ae = []
Al = []
B = []
Be = []
Bl = []
    
for k_h in kappa_h:
    for tau_e in tau_es:
        for g_k in gamma_k:
            for n_l in range(1):
                data = 0
                for n_g in range(2):
                    g_h = gamma_h[n_g]
                    s_l = kaon[n_l]
                    for inv in [-1,1]:
                        p_e = inv*np.array(all_p[0])
                        for p_m in inv*np.array(all_p[1:]):
                            a_data = ((-1)**n_g)*(-1*inv)*np.loadtxt(indir+"correlator-kl-"
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
                                        + str(p_m[2])+".dat"
                                        )
                            # tb.Plot(np.mean(tb.Cut_conf(a_data,64),0)[:,1],tb.Bootstrap_erro(tb.Bootstrap(tb.Cut_conf(a_data,64),4,0),0)[:,1],Title="3pt_data/correlator-kl-"
                            #             + kappa_l + "-ks-"
                            #             + kappa_s + "-"
                            #             + s_l + "-kh-"
                            #             + k_h + "-"
                            #             + g_k + "-"
                            #             + "tau_e-" + str(tau_e) + "-"
                            #             + g_h
                            #             + "-pe-"
                            #             + str(p_e[0])
                            #             + str(p_e[1])
                            #             + str(p_e[2])
                            #             + "-pm-"
                            #             + str(p_m[0])
                            #             + str(p_m[1])
                            #             + str(p_m[2]))
                            A.append(np.mean(tb.Cut_conf(a_data,128),0)[:,1])
                            Ae.append(tb.Bootstrap_erro(tb.Bootstrap(tb.Cut_conf(a_data,128),4,0),0)[:,1])
                            Al.append(
                                s_l
                                + g_h
                                + "-pe-"
                                + str(p_e[0])
                                + str(p_e[1])
                                + str(p_e[2])
                                + "-pm-"
                                + str(p_m[0])
                                + str(p_m[1])
                                + str(p_m[2])+".dat"
                                )
                            data += a_data
                np.savetxt(outdir+"3pt_correlator-kl-"
                                       + kappa_l + "-ks-"
                                       + kappa_s + "-" + s_l + "-kh-"
                                       + k_h + "-"
                                       + g_k + "-"
                                       + "tau_e-" + str(tau_e) + "-"
                                       + "ge-14-gm-13"
                                       + "-pe-001-pm-10-1.dat"
                                       ,data/16)

for k_h in kappa_h:
    for tau_e in tau_es:
        for g_k in gamma_k:
            for n_l in range(1):
                data = 0
                for n_g in range(2):
                    g_h = gamma_h[n_g]
                    s_l = kaon[n_l]
                    for inv in [-1,1]:
                        p_m = inv*np.array(all_p[0])
                        for p_e in inv*np.array(all_p[1:]):
                            a_data = ((-1)**n_g)*(-1*inv)*np.loadtxt(indir+"correlator-kl-"
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
                                        + str(p_m[2])+".dat"
                                        )
                            # tb.Plot(np.mean(tb.Cut_conf(a_data,64),0)[:,1],tb.Bootstrap_erro(tb.Bootstrap(tb.Cut_conf(a_data,64),4,0),0)[:,1],Title="3pt_data/correlator-kl-"
                            #             + kappa_l + "-ks-"
                            #             + kappa_s + "-"
                            #             + s_l + "-kh-"
                            #             + k_h + "-"
                            #             + g_k + "-"
                            #             + "tau_e-" + str(tau_e) + "-"
                            #             + g_h
                            #             + "-pe-"
                            #             + str(p_e[0])
                            #             + str(p_e[1])
                            #             + str(p_e[2])
                            #             + "-pm-"
                            #             + str(p_m[0])
                            #             + str(p_m[1])
                            #             + str(p_m[2]))
                            B.append(np.mean(tb.Cut_conf(a_data,128),0)[:,1])
                            Be.append(tb.Bootstrap_erro(tb.Bootstrap(tb.Cut_conf(a_data,128),4,0),0)[:,1])
                            Bl.append(
                                s_l
                                + g_h
                                + "-pe-"
                                + str(p_e[0])
                                + str(p_e[1])
                                + str(p_e[2])
                                + "-pm-"
                                + str(p_m[0])
                                + str(p_m[1])
                                + str(p_m[2])
                                )
                            data += a_data
                np.savetxt(outdir+"3pt_correlator-kl-"
                                       + kappa_l + "-ks-"
                                       + kappa_s + "-" + s_l + "-kh-"
                                       + k_h + "-"
                                       + g_k + "-"
                                       + "tau_e-" + str(tau_e) + "-"
                                       + "ge-14-gm-13"
                                       + "-pe-10-1-pm-001.dat"
                                       ,data/16)          
tb.Plot(A,Ae,xshift=0.05,Label=Al)
tb.Plot(B,Be,xshift=0.05,Label=Bl)
tb.Plot(list(A) + list(-1*np.array(B)),Ae+Be,xshift=0.05,Label=Al+Bl)