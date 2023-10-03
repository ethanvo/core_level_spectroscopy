#!/usr/bin/env python
from fileutils import load, dump
import numpy as np
import sys

au2ev = 27.211386245988

material = load("data/{}".format(sys.argv[1]))

ip = load("data/{}".format(material["eomip_valence"]))
core = load("data/{}".format(material["eomip_core"]))

eip = np.array(ip['eip'])
ecore = np.array(core['eip'])

cbe = (np.amin(ecore) - np.amax(eip)) * au2ev

core_data = dict([
    ('cbe', float(cbe))
    ])

dump(core_data, "data/{}".format(material["cbe"]))

