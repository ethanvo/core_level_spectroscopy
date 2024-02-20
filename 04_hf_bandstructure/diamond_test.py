#!/usr/bin/env python3
from bandstructure import make_cell, get_all_electron_bandstructure

cell = make_cell('c', 'ccecpccpvdz', pseudo='ccecp')
vbmax_kpt, cbmin_kpt, g_vbmax, g_cbmin = get_all_electron_bandstructure('c', cell, 'GX', 2, kmesh=[2, 2, 2], e_kn_file="testfile.h5", output_file="testfile.json")
print("vbmax_kpt: {}".format(vbmax_kpt))
print("cbmin_kpt: {}".format(cbmin_kpt))
print("g_vbmax: {}".format(g_vbmax))
print("g_cbmin: {}".format(g_cbmin))