#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import berkelplot
from fileutils import load

cbes = np.zeros((2, 3))
cbes[0, 0] = load("data/c_ccpcvtz_k2_C_1s_cbe.json")["cbe"]
cbes[0, 1] = load("data/c_ccpcvtz_k3_C_1s_cbe.json")["cbe"]
cbes[1, 0] = load("data/c_ccpcvtz_k2_C_1s_nvir_act25_cbe.json")["cbe"]
cbes[1, 1] = load("data/c_ccpcvtz_k3_C_1s_nvir_act25_cbe.json")["cbe"]
cbes[1, 2] = load("data/c_ccpcvtz_k4_C_1s_nvir_act25_cbe.json")["cbe"]

cbes[1, :] += cbes[0, 1] - cbes[1, 1]

Nk = np.array([2, 3, 4])
Nk = 1 / Nk
slope = (cbes[1, 2] - cbes[1, 1]) / (Nk[2] - Nk[1])
intercept = cbes[1, 2] - slope * Nk[2]
print(intercept)

size = berkelplot.fig_size(n_row=2, n_col=1)
fig, ax = plt.subplots(1, 1, figsize=size)
ax.plot(Nk, cbes[1, :], marker="o", color="blue")
ax.plot((0, Nk[2]), (intercept, cbes[1, 2]), linestyle="--", color="blue")
ax.plot((0, 0.55), (283.9, 283.9), linestyle="-", color="black")
ax.set_xlabel(r"$N_k^{-1/3}$")
ax.set_ylabel("CBE (eV)")
ax.set_xlim(0, 0.55)
plt.tight_layout()
plt.savefig("figures/c_ccpcvtz_C_1s_cbe.pdf")