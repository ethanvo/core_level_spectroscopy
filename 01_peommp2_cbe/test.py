#!/usr/bin/env python
import h5py
with h5py.File("data/c_ccpcvtz_k2_C_1s_nvir_act74_imds.h5", "r") as f:
    print(f.keys())
    print(f["imds"].keys())
