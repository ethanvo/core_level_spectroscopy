#!/usr/bin/env python3
from bandstructure import make_cell, get_all_electron_bandstructure
import sys
from fileutils import load

input_file = sys.argv[1]
data = load("data/{}".format(input_file))
cell = make_cell(data["formula"], data["basis"], pseudo=data["pseudo"])
vbmax_kpt, cbmin_kpt, g_vbmax, g_cbmin = get_all_electron_bandstructure(data["formula"], cell, data["path"], data["npoints"], data["kmesh"], e_kn_file=data["e_kn_file"], output_file=data["output_file"])
print("vbmax_kpt: {}".format(vbmax_kpt))
print("cbmin_kpt: {}".format(cbmin_kpt))
print("g_vbmax: {}".format(g_vbmax))
print("g_cbmin: {}".format(g_cbmin))
