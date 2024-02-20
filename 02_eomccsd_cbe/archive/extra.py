'''

# Coupled Cluster
mycc = cc.KRCCSD(mymf, frozen=material["frozen"], mo_coeff=no_coeff)
t1 = t2 = eris = None
if os.path.isfile(f"data/{material['diis']}"):
    mycc.restore_from_diis(f"data/{material['diis']}")
    t1, t2 = mycc.t1, mycc.t2
mycc.keep_exxdiv = True
# check if eris is already saved
if h5py.get("eris", None) is not None:
    eris = _ERIS(mycc)
    eris_dict = load_h5(h5file, "eris")
    eris.__dict__.update(eris_dict)
else:
    eris = mycc.ao2mo(mo_coeff=no_coeff)
    grp_eris = h5file.create_group("eris")
    for key in ["mo_coeff", "fock"]:
        grp_eris.create_dataset(key, data=getattr(eris, key))
    eris_keys = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
    if getattr(eris, "feri1", None):
        for key in eris_keys:
            eris.feri1.copy(key, grp_eris)
    else:
        for key in eris_keys:
            grp_eris.create_dataset(key, data=getattr(eris, key))
ekrccsd, t1, t2 = mycc.kernel(t1=t1, t2=t2, eris=eris)

###############################################################################
# IMDS 1e
###############################################################################
imds = _IMDS(mycc, eris)
imds._make_shared_1e()
grp_imds = h5file.create_group("imds")
for k, v in imds.__dict__.items():
    if (k[0] == "F" or k[0] == "L") and v is not None:
        print("IMDS key to save:", k, ", type:", type(v))
        grp_imds.create_dataset(k, data=v)
###############################################################################
# IMDS 2e
###############################################################################
nkpts, nocc, nvir = t1.shape
ovoo_dest = grp_imds.create_dataset(
    "Wovoo", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc), dtype=t1.dtype.char
)
ovoo = Wovoo(t1, t2, eris, mycc.khelper.kconserv, ovoo_dest)

oooo_dest = grp_imds.create_dataset(
    "Woooo", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=t1.dtype.char
)
oooo = Woooo(t1, t2, eris, mycc.khelper.kconserv, oooo_dest)

ooov_dest = grp_imds.create_dataset(
    "Wooov", (nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir), dtype=t1.dtype.char
)
ooov = Wooov(t1, t2, eris, mycc.khelper.kconserv, ooov_dest)

imds.Wovoo = ovoo
material["made_wovoo"] = True
imds.Woooo = oooo
material["made_woooo"] = True
imds.Wooov = ooov
material["made_wooov"] = True
###############################################################################
# EOM-CCSD
###############################################################################
myeom = EOMIP(mycc)
eip, vip = myeom.ipccsd(nroots=material["vb_nroots"], imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([
    ("eip", eip.tolist()),
    ("convergence", convergence.tolist())
])
###############################################################################
# CVS-EOM-CCSD
###############################################################################
myeom = CVSEOMIP(mycc)
myeom.mandatory = material["core"]
eip, vip = myeom.ipccsd(nroots=len(material["core"]), imds=imds, kptlist=[0])
convergence = np.real(myeom.converged) != 0
myeom_data = dict([
    ("eip", eip.tolist()),
    ("convergence", convergence.tolist())
])
'''