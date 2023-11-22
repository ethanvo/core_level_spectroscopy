#!/usr/bin/env python
import h5py
with h5py.File("data/c_ccpcvtz_k4_C_1s_nvir_act66_imds.h5", "r") as f:
    print(f.keys())
    print(f["imds"].keys())
