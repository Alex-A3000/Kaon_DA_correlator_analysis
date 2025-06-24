# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 02:59:30 2025

@author: s9503
"""

import numpy as np
import matplotlib.pyplot as plt

# Gegenbauer polynomials C_n^{3/2}(xi)
def C0(xi):
    return 1

def C2(xi):
    return (3/2) * (5 * xi**2 - 1)

# Distribution amplitude in terms of <ξ²>(μ)
def phi_pi_2(xi, xi2_avg):
    # Use known relation: ⟨ξ²⟩ = (1/5) + (12/35) * a_2
    a2 = (35 * xi2_avg - 7) / 12
    return (3/4) * (1 - xi**2) * (C0(xi) + a2 * C2(xi))

# ξ values from -1 to 1
xi_vals = np.linspace(-1, 1, 500)

# Different ⟨ξ²⟩ values
xi2_list = [0.2, 0.25, 0.3]
styles = ['-', '--', '-.']
labels = [r'$\langle \xi^2 \rangle = 0.2$', r'$\langle \xi^2 \rangle = 0.25$', r'$\langle \xi^2 \rangle = 0.3$']

# Plot
plt.figure(figsize=(7, 5))
for xi2, style, label in zip(xi2_list, styles, labels):
    plt.plot(xi_vals, phi_pi_2(xi_vals, xi2), style, label=label, linewidth=2.2)

plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\phi_{\pi}^{(2)}(\xi, \mu^2)$', fontsize=14)
plt.legend(fontsize=12, frameon=True)
plt.grid(False)
plt.tight_layout()
plt.show()

# Gegenbauer polynomials
def C0(xi): return 1
def C1(xi): return 3 * xi
def C2(xi): return (3/2) * (5 * xi**2 - 1)

# Full DA for kaon
def phi_K_2(xi, xi_avg, xi2_avg):
    a1 = (5/3) * xi_avg
    a2 = (35/12) * (xi2_avg - 1/5)
    return (3/4) * (1 - xi**2) * (C0(xi) + a1 * C1(xi) + a2 * C2(xi))

# Grid
xi_vals = np.linspace(-1, 1, 500)

# Different moments to compare
moment_sets = [
    (0.0, 0.2),  # pion-like
    (0.02, 0.2),  # small xi avg
    (0.04, 0.2),  # larger breaking
]
styles = ['-', '--', '-.']
labels = [fr'$\langle \xi \rangle = {xi}, \ \langle \xi^2 \rangle = {xi2}$' for xi, xi2 in moment_sets]

# Plot
plt.figure(figsize=(7, 5))
for (xi_avg, xi2_avg), style, label in zip(moment_sets, styles, labels):
    plt.plot(xi_vals, phi_K_2(xi_vals, xi_avg, xi2_avg), style, label=label, linewidth=2.2)

plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'$\phi_K^{(2)}(\xi, \mu^2)$', fontsize=14)
plt.legend(fontsize=11, frameon=True)
plt.tight_layout()
plt.show()