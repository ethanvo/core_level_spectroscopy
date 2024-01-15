#!/usr/bin/env python
import opt_einsum as oe
import numpy as np
from scipy.optimize import root_scalar

def peommp2_ip_storage(nkpts, nocc, nvir):
    total = 0
    nmo = nocc + nvir
    # SCF
    total += nkpts * nmo * nmo # mo_coeff
    total += nkpts * nmo # mo_energy
    total += nkpts * nkpts * nkpts * nmo * nmo * nmo # CDERI
    # CCSD
    total += nkpts * nocc * nvir # t1
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # t2
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # eris.oooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # eris.ooov
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # eris.oovv
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # eris.ovov
    total += nkpts * nkpts * nkpts * nvir * nocc * nocc * nvir # eris.voov
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # eris.vovv
    # EOM
    total += nkpts * nocc * nocc # Loo
    total += nkpts * nvir * nvir # Lvv
    total += nkpts * nocc * nvir # Fov
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nocc # Wovoo
    total += nkpts * nmo * nmo # fock
    total += nkpts * nkpts # Lpv
    # return total in bytes in gigabytes
    return total * 16 / 1000**3

def peommp2_ea_storage(nkpts, nocc, nvir):
    total = 0
    nmo = nocc + nvir
    # SCF
    total += nkpts * nmo * nmo # mo_coeff
    total += nkpts * nmo # mo_energy
    total += nkpts * nkpts * nkpts * nmo * nmo * nmo # CDERI
    # CCSD
    total += nkpts * nocc * nvir # t1
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # t2
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # eris.oooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # eris.ooov
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # eris.oovv
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # eris.ovov
    total += nkpts * nkpts * nkpts * nvir * nocc * nocc * nvir # eris.voov
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # eris.vovv
    # EOM
    total += nkpts * nocc * nocc # Loo
    total += nkpts * nvir * nvir # Lvv
    total += nkpts * nocc * nvir # Fov
    total += nkpts * nkpts * nkpts * nvir * nvir * nvir * nocc # Wvvvo
    total += nkpts * nmo * nmo # fock
    total += nkpts * nkpts # Lpv
    return total * 16 / 1000**3

def eomccsd_ip_storage(nkpts, nocc, nvir):
    total = 0
    nmo = nocc + nvir
    # SCF
    total += nkpts * nmo * nmo # mo_coeff
    total += nkpts * nmo # mo_energy
    total += nkpts * nkpts * nkpts * nmo * nmo * nmo # CDERI
    # CCSD
    total += nkpts * nocc * nvir # t1
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # t2
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # eris.oooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # eris.ooov
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # eris.oovv
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # eris.ovov
    total += nkpts * nkpts * nkpts * nvir * nocc * nocc * nvir # eris.voov
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # eris.vovv
    # EOM
    total += nkpts * nocc * nocc # Loo
    total += nkpts * nvir * nvir # Lvv
    total += nkpts * nocc * nvir # Fov
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # Wovov
    total += nkpts * nkpts * nkpts * nocc * nvir * nvir * nocc # Wovvo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # Woooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # Wooov
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nocc # Wovoo
    total += nkpts * nmo * nmo # fock
    total += nkpts * nkpts # Lpv
    return total * 16 / 1000**3

def eomccsd_ea_storage(nkpts, nocc, nvir):
    total = 0
    nmo = nocc + nvir
    # SCF
    total += nkpts * nmo * nmo # mo_coeff
    total += nkpts * nmo # mo_energy
    total += nkpts * nkpts * nkpts * nmo * nmo * nmo # CDERI
    # CCSD
    total += nkpts * nocc * nvir # t1
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # t2
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nocc # eris.oooo
    total += nkpts * nkpts * nkpts * nocc * nocc * nocc * nvir # eris.ooov
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # eris.oovv
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # eris.ovov
    total += nkpts * nkpts * nkpts * nvir * nocc * nocc * nvir # eris.voov
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # eris.vovv
    # EOM
    total += nkpts * nocc * nocc # Loo
    total += nkpts * nvir * nvir # Lvv
    total += nkpts * nocc * nvir # Fov
    total += nkpts * nkpts * nkpts * nocc * nvir * nocc * nvir # Wovov
    total += nkpts * nkpts * nkpts * nocc * nvir * nvir * nocc # Wovvo
    total += nkpts * nkpts * nkpts * nocc * nocc * nvir * nvir # Woovv
    total += nkpts * nkpts * nkpts * nvir * nocc * nvir * nvir # Wvovv
    total += nkpts * nkpts * nkpts * nvir * nvir * nvir * nocc # Wvvvo
    total += nkpts * nmo * nmo # fock
    total += nkpts * nkpts # Lpv
    return total * 16 / 1000**3

def computer_flops():
    """
    sockets_per_node = 2
    cores_per_socket = 16
    cycles_per_second = 2.9e9
    flops_per_cycle = 32
    efficiency = 0.9
    flops = sockets_per_node * cores_per_socket * cycles_per_second * flops_per_cycle * efficiency
    print("FLOPS per node: %e" % flops)
    """
    # Benchmark Data
    flops = int(125642e6 * 0.9)
    return flops

def peommp2_ea_complexity(nkpts, nocc, nvir):
    nmo = nocc + nvir
    # contract vovv with oovv to get Wvvvo
    flops = 4 * nkpts * nkpts * nkpts * nkpts * nvir * nvir * nvir * nocc * nocc * nvir * 2
    # Print flops in scientific notation
    print('flops: {:e}'.format(flops))
    print('oe predicted flops {:e}'.format(2.721e+8 * nkpts**4 * 4))
    time = flops / computer_flops()
    # return time in 0-00:00:00 format
    days = int(time / (24 * 60 * 60))
    hours = int((time - days * 24 * 60 * 60) / (60 * 60))
    minutes = int((time - days * 24 * 60 * 60 - hours * 60 * 60) / 60)
    seconds = int(time - days * 24 * 60 * 60 - hours * 60 * 60 - minutes * 60)
    return "%d-%02d:%02d:%02d" % (days, hours, minutes, seconds)                        

nkpts = 3**3
nocc = 6
nvir = 80
print("Total storage for nkpts = %d, nocc = %d, nvir = %d" % (nkpts, nocc, nvir))
print("P-EOM-IP-MP2: %f GB" % peommp2_ip_storage(nkpts, nocc, nvir))
print("P-EOM-EA-MP2: %f GB" % peommp2_ea_storage(nkpts, nocc, nvir))
print("EOM-IP-CCSD: %f GB" % eomccsd_ip_storage(nkpts, nocc, nvir))
print("EOM-EA-CCSD: %f GB" % eomccsd_ea_storage(nkpts, nocc, nvir))
print("P-EOM-EA-MP2 Time: %s" % peommp2_ea_complexity(nkpts, nocc, nvir))

def f(x):
    return peommp2_ip_storage(4**3, 6, x) - 16000
result = root_scalar(f, x0=86, x1=10)
print(result.root)
print(peommp2_ip_storage(4**3, 6, 73))
