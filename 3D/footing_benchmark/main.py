#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-dimensional footing problem (Section 6.4, Tables 7 and 8 of the paper).

Geometry, boundary conditions and material data follow cases I-III of
Both, Kumar, Nordbotten, Radu, "The gradient flow structures of thermo-poro-visco-elastic
processes in porous media", arXiv:1907.03134 (ref. [17] of the paper):

    u = 0                        on [0,1]^2 x {0}                          (clamped base)
    sigma n = 1e9 t e_3          on [0.25,0.75]^2 x {1}                    (ramped load)
    sigma n = 0                  on the rest of the boundary
    w . n = 0                    on [0,1]^2 x {0}  U  [0.25,0.75]^2 x {1}  (sealed)
    p = 0                        on the rest of the boundary               (drained)

    nu = 0.2, alpha = 1, c0 = 1/M_b = 1e-11, kappa = 1e-13,
    E in {1e12, 1e11, 1e10, 1e9}  ->  tau = M / c0 in {0.18, 1.8, 18, 180},
    5 implicit-Euler steps of dt = 0.1, structured tetrahedral N^3 mesh (N = 8, resolves the patch),
    k = 1, 2, theta = beta / beta_max in {0, 1/4, 1/2, 3/4, 1},  beta_max = M/2 (Lemma 3).

Dual-mixed boundary conditions: the traction sigma n is essential on the stress space
(homogeneous on the sides and the unloaded top; on the patch a time-dependent stress lifting
moved to the mechanics right-hand side); no-flow is essential on the flux space; clamping and
drainage are natural and homogeneous. Essential constraints are imposed by restricting the
stress and flux blocks to their free degrees of freedom.

Units: the raw SI data give assembled blocks spanning ~22 orders of magnitude, so the problem is
solved in non-dimensional form (NONDIM = True): stresses and pressure are divided by
S = E_REF, i.e. mu, lam -> mu/S, lam/S, c0 -> c0 S, kappa -> kappa S, load -> load/S. This is an
exact change of units of the continuous and discrete problem (u and w are unchanged), so all
dimensionless quantities (tau, nu, theta) are preserved. Every direct solve is checked a posteriori
by its relative residual, globally and block-wise.

Linear solves: direct, PETSc/MUMPS with nested-dissection ordering (METIS); SOLVER = "splu" falls
back to SciPy SuperLU.

Outputs (results_3d/):
  footing_beta_grid_3d.csv        one row per (E, k, theta), incl. per-step counts and residuals
  table_footing_beta_grid_3d.tex  Tables 7 (k = 1) and 8 (k = 2), booktabs, per-row minimum in bold
Set TABLES_ONLY = True to rebuild the tables from the csv without recomputing.

Author: Tugay Dagli
"""

# ======================================================================
# Imports
# ======================================================================
import ngsolve
from ngsolve import *
import ngsolve.meshes as ngm
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, hstack, vstack
import csv
import os


# ======================================================================
# Paths
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_3d")
os.makedirs(RESULTS_DIR, exist_ok=True)

BETA_GRID_CSV = os.path.join(RESULTS_DIR, "footing_beta_grid_3d.csv")
BETA_GRID_TEX = os.path.join(RESULTS_DIR, "table_footing_beta_grid_3d.tex")


# ======================================================================
# Settings
# ======================================================================
CHEAP_TEST = False         # True: laptop smoke test (N = 4, one E value)
TABLES_ONLY = False        # True: rebuild the tables from BETA_GRID_CSV
VERBOSE_ITER = False

NONDIM = True              # True: non-dimensional form (paper setting); False: raw SI (ill-conditioned)

# ---- material data of [17], cases I-III (SI units) ----------------------
NU = 0.2                   # Poisson ratio
ALPHA = 1.0                # Biot-Willis coefficient
M_BIOT = 1e11              # Biot modulus [Pa]  ->  c0 = 1/M_BIOT = 1e-11
KAPPA = 1e-13              # permeability
LOAD_RATE = 1e9            # sigma n = LOAD_RATE * t * e_3 on the patch
E_REF = 1e10               # stress scale of the non-dimensionalization [Pa]

# ---- time stepping ------------------------------------------------------
N_STEPS = 5
DT = 0.1

# ---- sweep ---------------------------------------------------------------
if CHEAP_TEST:
    K_LIST = [1, 2]
    SCAN_N = 4
    E_VALUES = [1e10]
else:
    K_LIST = [1, 2]
    SCAN_N = 8                                  # multiple of 4: mesh resolves the patch edges
    E_VALUES = [1e12, 1e11, 1e10, 1e9]         # table rows (decreasing E = increasing coupling)
THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

assert SCAN_N % 4 == 0, "SCAN_N must be a multiple of 4 to resolve the [0.25,0.75]^2 patch."

# reference to the material data in the table captions (replace by your \cite key if desired)
REF_TEXT = "[17]"

# ---- direct-solver backend -----------------------------------------------
SOLVER = "mumps"              # "mumps" (paper setting) or "splu"
MUMPS_SOLVER_TYPE = "mumps"
MUMPS_ICNTL_7 = 5             # sequential ordering: 5 = METIS (nested dissection), 3 = SCOTCH
MUMPS_ICNTL_14 = 40           # workspace headroom in %
MUMPS_OUT_OF_CORE = False

# ---- splitting -------------------------------------------------------------
TOL = 1e-6                    # stopping criterion (47): relative update of [w,p] and [sigma,omega,u]
MAX_ITERATIONS = 10000

# ---- diagnostics thresholds --------------------------------------------------
RESID_WARN = 1e-4             # block-wise reconstruction residual above which a warning is printed


# ======================================================================
# Versions
# ======================================================================
MUMPS_ORDERING_NAMES = {0: "AMD", 1: "user", 2: "AMF", 3: "SCOTCH", 4: "PORD", 5: "METIS", 6: "QAMD"}


def petsc_version():
    try:
        from petsc4py import PETSc
        return ".".join(str(v) for v in PETSc.Sys.getVersion())
    except Exception:
        return "n/a"


# ======================================================================
# Direct solver: factor once, solve many, with a swappable backend
# ======================================================================
class DirectSolver:
    """
    Factor-once / solve-many exact direct solver.

        backend = "splu"  : SciPy SuperLU (COLAMD ordering)
        backend = "mumps" : PETSc + MUMPS (KSP "preonly" + PC "lu"), nested-dissection ordering

    The factorization is computed once in __init__ and reused by every solve().
    `ordering` records the ordering MUMPS actually used (INFOG(7)), if petsc4py exposes it.
    Non-finite solutions are NOT raised here: they are detected by the residual diagnostics of
    biot_footing_3d (solve_finite flag), which report them per run.
    """

    def __init__(self, A, backend="mumps", name=""):
        self.backend = backend
        self.ordering = "COLAMD (SuperLU)" if backend == "splu" else "unknown"

        if backend == "splu":
            self._lu = sp.linalg.splu(A.tocsc())
            print(f"[solver] {name}: factorized with SciPy SuperLU (splu)")

        elif backend == "mumps":
            from petsc4py import PETSc

            opts = PETSc.Options()
            opts["mat_mumps_icntl_7"] = MUMPS_ICNTL_7
            opts["mat_mumps_icntl_14"] = MUMPS_ICNTL_14
            if MUMPS_OUT_OF_CORE:
                opts["mat_mumps_icntl_22"] = 1

            Acsr = A.tocsr()
            Pm = PETSc.Mat().createAIJ(
                size=Acsr.shape,
                csr=(Acsr.indptr.astype(PETSc.IntType),
                     Acsr.indices.astype(PETSc.IntType),
                     Acsr.data),
            )
            Pm.assemble()

            ksp = PETSc.KSP().create()
            ksp.setOperators(Pm)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType(MUMPS_SOLVER_TYPE)
            ksp.setFromOptions()
            ksp.setUp()               # symbolic + numeric factorization

            try:
                code = int(pc.getFactorMatrix().getMumpsInfog(7))
                self.ordering = f"{MUMPS_ORDERING_NAMES.get(code, code)} (MUMPS INFOG(7)={code})"
            except Exception:
                self.ordering = f"requested ICNTL(7)={MUMPS_ICNTL_7} (INFOG(7) not available)"

            self._P = Pm
            self._ksp = ksp
            self._b = Pm.createVecLeft()
            self._x = Pm.createVecRight()
            print(f"[solver] {name}: factorized with PETSc/MUMPS, ordering {self.ordering}, "
                  f"icntl_14={MUMPS_ICNTL_14}{', out-of-core' if MUMPS_OUT_OF_CORE else ''}")

        else:
            raise ValueError("SOLVER must be 'splu' or 'mumps'.")

    def solve(self, rhs):
        rhs = np.asarray(rhs)
        flat = rhs.reshape(-1)
        if self.backend == "splu":
            out = self._lu.solve(flat)
        else:
            self._b.setArray(flat)
            self._ksp.solve(self._b, self._x)
            out = self._x.getArray().copy()   # copy: _x is overwritten by the next solve
        return out.reshape(rhs.shape)


# ======================================================================
# Material data of [17] -> assembled coefficients (optionally non-dimensional)
# ======================================================================
def material_from_E(E):
    """
    Lame parameters and poroelastic coefficients from (E, nu, M_b, kappa) of [17].

    NONDIM = True rescales stresses and pressure by S = E_REF (lengths: unit cube, time: unscaled):
        mu, lam -> mu/S, lam/S,   c0 -> c0*S,   kappa -> kappa*S,   load -> load/S.
    This is an exact change of units (u and w unchanged, sigma and p divided by S); the
    dimensionless quantities tau = M/c0, nu and theta are preserved. Note that the stopping
    criterion (47) measures the relative update of [w,p] jointly, so it is not strictly invariant
    under the rescaling of p; counts are reported for the non-dimensional form only.
    """
    d = 3
    mu = E / (2.0 * (1.0 + NU))
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    c0 = 1.0 / M_BIOT
    kappa = KAPPA
    load = LOAD_RATE
    if NONDIM:
        S = E_REF
        mu, lam = mu / S, lam / S
        c0 = c0 * S
        kappa = kappa * S
        load = load / S
    M = d * ALPHA**2 / (2.0 * mu + d * lam)          # M = d alpha^2 / (2 mu + d lam)
    return dict(mu=mu, lam=lam, c0=c0, kappa=kappa, load=load, M=M, tau=M / c0, poisson=NU)


# ======================================================================
# Boundary-facet classification and DOF masks (coordinate based)
# ======================================================================
def facet_dof_mask(fes, mesh, pred):
    """Boolean mask over fes.ndof: True for the DOFs of boundary facets whose centroid satisfies pred."""
    mask = np.zeros(fes.ndof, dtype=bool)
    for el in mesh.Elements(BND):
        c = np.array([mesh.vertices[v.nr].point for v in el.vertices]).mean(axis=0)
        if pred(c):
            for dof in fes.GetDofNrs(el):
                if dof >= 0:
                    mask[dof] = True
    return mask


_TOL_GEO = 1e-6
def is_bottom(c): return abs(c[2]) < _TOL_GEO
def is_top(c):    return abs(c[2] - 1.0) < _TOL_GEO
def is_patch(c):  return is_top(c) and (0.25 < c[0] < 0.75) and (0.25 < c[1] < 0.75)


# ======================================================================
# Footing solver -> fixed-stress iteration counts per time step
# ======================================================================
def biot_footing_3d(E_young, k, theta, N=SCAN_N, n_steps=N_STEPS, dt_val=DT, verbose_iter=VERBOSE_ITER):
    """
    Fixed-stress splitting for the footing problem with beta = theta * beta_max, beta_max = M/2.
    Returns a dict with the iteration counts and diagnostics of one (E, k, theta) run.
    """
    d = 3
    mesh = ngm.MakeStructured3DMesh(hexes=False, nx=N, ny=N, nz=N)

    # ---- spaces: RT_k stress rows and flux, dP_k displacement and pressure, cont. P_k rotation ----
    FESHdiv = HDiv(mesh, order=k, RT=True)
    FESHdivW = HDiv(mesh, order=k, RT=True)
    U = VectorL2(mesh, order=k)
    P = L2(mesh, order=k)
    FESH1v = VectorH1(mesh, order=k)
    FESHdivS = FESpace([FESHdiv, FESHdiv, FESHdiv])

    sigma1, sigma2, sigma3 = FESHdivS.TrialFunction()
    tau1, tau2, tau3 = FESHdivS.TestFunction()
    u, zt = U.TnT()
    w, v = FESHdivW.TnT()
    p, q = P.TnT()
    omega, eta = FESH1v.TnT()

    # ---- coefficients ----
    matp = material_from_E(E_young)
    mu, lam = matp["mu"], matp["lam"]
    alpha = ALPHA
    M = matp["M"]
    c0 = matp["c0"]
    KinvC = 1.0 / matp["kappa"]
    load_rate = matp["load"]
    C1 = M / (d * alpha)                             # alpha_tilde
    C2 = c0 + M                                      # c0_tilde
    beta_max = 0.5 * M                               # Lemma 3
    beta_paper = theta * beta_max                    # paper convention: beta >= 0
    beta_asm = 0.0 - beta_paper                      # flow block (c0_tilde - beta) M_p ; "0.0 -" avoids -0.0
    tau = M / c0

    def tr3(s1, s2, s3):
        return s1[0] + s2[1] + s3[2]

    e1 = CoefficientFunction((1, 0, 0))
    e2 = CoefficientFunction((0, 1, 0))
    e3 = CoefficientFunction((0, 0, 1))
    Asigma1 = (1 / (2 * mu)) * (sigma1 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e1)
    Asigma2 = (1 / (2 * mu)) * (sigma2 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e2)
    Asigma3 = (1 / (2 * mu)) * (sigma3 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e3)

    # ---- time-dependent traction lifting (third stress row, normal trace on the patch) ----
    t = Parameter(0.0)
    patch_ind = IfPos((x - 0.25) * (0.75 - x), IfPos((y - 0.25) * (0.75 - y), 1.0, 0.0), 0.0)
    ztop = IfPos(z - 0.5, 1.0, 0.0)
    trac_cf = CoefficientFunction((0, 0, load_rate * t * patch_ind * ztop))
    gf_sig_lift = GridFunction(FESHdivS)

    def set_lifting():
        gf_sig_lift.vec[:] = 0.0
        gf_sig_lift.components[2].Set(trac_cf, definedon=mesh.Boundaries(".*"))

    gfusigma = GridFunction(FESHdivS)
    gfusigma_old = GridFunction(FESHdivS)
    gfup = GridFunction(P)
    gfup_old = GridFunction(P)

    nW, nP = FESHdivW.ndof, P.ndof
    nSig, nU, nRot = FESHdivS.ndof, U.ndof, FESH1v.ndof

    # ---- essential DOFs: stress on top + sides (all but bottom), flux on bottom + patch ----
    sig_ess = facet_dof_mask(FESHdivS, mesh, lambda c: not is_bottom(c))
    w_ess = facet_dof_mask(FESHdivW, mesh, lambda c: is_bottom(c) or is_patch(c))
    sf = np.where(~sig_ess)[0]
    sc = np.where(sig_ess)[0]
    wf = np.where(~w_ess)[0]

    # ---- assembled blocks (NGSolve -> SciPy CSR) ----
    a = BilinearForm(FESHdivW); a += (w * v) * dx; a.Assemble()
    rows, cols, vals = a.mat.COO()
    A = KinvC * csr_matrix((vals, (rows, cols)), shape=(nW, nW))

    b = BilinearForm(trialspace=P, testspace=FESHdivW); b += -(p * div(v)) * dx; b.Assemble()
    rows, cols, vals = b.mat.COO()
    B = csr_matrix((vals, (rows, cols)), shape=(nW, nP))

    c = BilinearForm(P); c += (p * q) * dx; c.Assemble()
    rows, cols, vals = c.mat.COO()
    Cm = csr_matrix((vals, (rows, cols)), shape=(nP, nP))
    Cmat = C2 * Cm
    TP = beta_asm * Cm

    e = BilinearForm(trialspace=FESHdivS, testspace=P)
    e += (sigma1[0] * q + sigma2[1] * q + sigma3[2] * q) * dx; e.Assemble()
    rows, cols, vals = e.mat.COO()
    E = C1 * csr_matrix((vals, (rows, cols)), shape=(nP, nSig))

    Pbf = BilinearForm(FESHdivS)
    Pbf += (Asigma1 * tau1 + Asigma2 * tau2 + Asigma3 * tau3) * dx; Pbf.Assemble()
    rows, cols, vals = Pbf.mat.COO()
    Pmat = csr_matrix((vals, (rows, cols)), shape=(nSig, nSig))

    r = BilinearForm(trialspace=U, testspace=FESHdivS)
    r += (div(tau1) * u[0] + div(tau2) * u[1] + div(tau3) * u[2]) * dx; r.Assemble()
    rows, cols, vals = r.mat.COO()
    R = csr_matrix((vals, (rows, cols)), shape=(nSig, nU))

    s = BilinearForm(trialspace=FESHdivS, testspace=FESH1v)
    s += ((sigma2[2] - sigma3[1]) * eta[0]
          + (sigma3[0] - sigma1[2]) * eta[1]
          + (sigma1[1] - sigma2[0]) * eta[2]) * dx
    s.Assemble()
    rows, cols, vals = s.mat.COO()
    S = csr_matrix((vals, (rows, cols)), shape=(nRot, nSig))

    # ---- flow block on free flux DOFs ----
    A_ff = A[wf][:, wf]
    B_f = B[wf][:, :]
    LHS1 = vstack([hstack([A_ff, B_f]),
                   hstack([-dt_val * B_f.T, (Cmat + TP)])]).tocsc()
    B1 = DirectSolver(LHS1, backend=SOLVER, name="LHS1 (flow)")

    # ---- mechanics block on free stress DOFs ----
    Pmat_ff = Pmat[sf][:, sf]
    StT = (S[:, sf]).T.tocsr()
    S_sf = S[:, sf]
    R_f = R[sf][:, :]
    ZrotRot = csr_matrix((nRot, nRot))
    ZrotU = csr_matrix((nRot, nU))
    ZUU = csr_matrix((nU, nU))
    PTILDE = vstack([hstack([Pmat_ff, StT]), hstack([-S_sf, ZrotRot])])
    RTILDE = vstack([R_f, ZrotU])
    LHS2 = vstack([hstack([PTILDE, RTILDE]),
                   hstack([-RTILDE.T, ZUU])]).tocsc()
    B2 = DirectSolver(LHS2, backend=SOLVER, name="LHS2 (mechanics)")

    # ---- couplings with the lifting (constant in time) ----
    ET_sf = (E[:, sf]).T.tocsr()
    Pmat_sf_sc = Pmat[sf][:, sc]
    S_sc = S[:, sc]
    RT_sc = (R[sc][:, :]).T.tocsr()

    def scatter_stress(sig_free_vals, sig_lift_full):
        full = sig_lift_full.copy()
        full[sf] = sig_free_vals.reshape(-1)
        return full

    # ---- a posteriori residual diagnostics (global and block-wise) ----
    LHS1_csr = LHS1.tocsr()
    LHS2_csr = LHS2.tocsr()
    n_wf = len(wf)          # flow: rows [0:n_wf] flux, [n_wf:] pressure
    n_sf = len(sf)          # mechanics: rows [0:n_sf] stress, the rest rotation + displacement
    diag = dict(flow_global=0.0, flow_flux=0.0, flow_pressure=0.0,
                mech_global=0.0, mech_stress=0.0, mech_rest=0.0)
    solve_finite = True

    def _rel(res_blk, rhs_blk, rhs_full):
        num = np.linalg.norm(res_blk)
        nb = np.linalg.norm(rhs_blk)
        if nb == 0.0:                      # block with zero right-hand side: scale by the full rhs
            nb = np.linalg.norm(rhs_full)
        if not (np.isfinite(num) and np.isfinite(nb)):
            return float("inf")
        return float(num / nb) if nb > 0 else 0.0

    def check_solve(x, rhs, lhs, nsplit, keys):
        nonlocal solve_finite
        if not np.isfinite(x).all():
            solve_finite = False
        rhs = rhs.reshape(-1)
        res = lhs @ x.reshape(-1) - rhs
        g, b1, b2 = keys
        diag[g] = max(diag[g], _rel(res, rhs, rhs))
        diag[b1] = max(diag[b1], _rel(res[:nsplit], rhs[:nsplit], rhs))
        diag[b2] = max(diag[b2], _rel(res[nsplit:], rhs[nsplit:], rhs))

    # ==================================================================
    # Time loop with fixed-stress splitting
    # ==================================================================
    iters_per_step = []
    hit_cap = False

    for n in range(n_steps):
        t.Set((n + 1) * dt_val)
        set_lifting()
        sig_lift_full = gf_sig_lift.vec.FV().NumPy().copy()
        sig_c = sig_lift_full[sc]

        p_old = gfup_old.vec.FV().NumPy().copy()
        sig_old_full = gfusigma_old.vec.FV().NumPy().copy()

        rot_blk = S_sc @ sig_c
        disp_blk = RT_sc @ sig_c
        lift_stress = Pmat_sf_sc @ sig_c

        # initial flow solve (previous-time-step stress)
        add1 = Cmat @ p_old + TP @ p_old
        RHS1 = np.concatenate([np.zeros(n_wf), add1]).reshape(-1, 1)
        RES1 = B1.solve(RHS1)
        check_solve(RES1, RHS1, LHS1_csr, n_wf, ("flow_global", "flow_flux", "flow_pressure"))
        p_cur = RES1[n_wf:].reshape(-1)
        gfup.vec.FV().NumPy()[:] = p_cur

        # initial mechanics solve
        stress_blk = -(ET_sf @ p_cur) - lift_stress
        RHS2 = np.concatenate([stress_blk, rot_blk, disp_blk]).reshape(-1, 1)
        RES2 = B2.solve(RHS2)
        check_solve(RES2, RHS2, LHS2_csr, n_sf, ("mech_global", "mech_stress", "mech_rest"))
        sig_full = scatter_stress(RES2[:n_sf], sig_lift_full)
        gfusigma.vec.FV().NumPy()[:] = sig_full

        prev1, prev2 = RES1, RES2
        err1 = err2 = np.inf
        it = 0
        while (err1 > TOL or err2 > TOL) and it < MAX_ITERATIONS:
            # flow
            add1 = E @ sig_old_full - E @ sig_full + Cmat @ p_old + TP @ p_cur
            RHS1 = np.concatenate([np.zeros(n_wf), add1]).reshape(-1, 1)
            RES1 = B1.solve(RHS1)
            check_solve(RES1, RHS1, LHS1_csr, n_wf, ("flow_global", "flow_flux", "flow_pressure"))
            p_cur = RES1[n_wf:].reshape(-1)
            gfup.vec.FV().NumPy()[:] = p_cur

            # mechanics
            stress_blk = -(ET_sf @ p_cur) - lift_stress
            RHS2 = np.concatenate([stress_blk, rot_blk, disp_blk]).reshape(-1, 1)
            RES2 = B2.solve(RHS2)
            check_solve(RES2, RHS2, LHS2_csr, n_sf, ("mech_global", "mech_stress", "mech_rest"))
            sig_full = scatter_stress(RES2[:n_sf], sig_lift_full)
            gfusigma.vec.FV().NumPy()[:] = sig_full

            # stopping criterion (47)
            err1 = np.linalg.norm(RES1 - prev1) / (np.linalg.norm(prev1) + 1e-14)
            err2 = np.linalg.norm(RES2 - prev2) / (np.linalg.norm(prev2) + 1e-14)
            prev1, prev2 = RES1, RES2
            it += 1
            if verbose_iter:
                print(f"    iter {it + 1} | flow {err1:.2e}, mech {err2:.2e}")

        iters_per_step.append(it + 1)
        if err1 > TOL or err2 > TOL:
            hit_cap = True
        gfusigma_old.vec.data = gfusigma.vec.data
        gfup_old.vec.data = gfup.vec.data

    return {
        "E_young": E_young, "nondim": NONDIM, "k": k, "theta": theta, "N": N,
        "n_steps": n_steps, "dt": dt_val, "tol": TOL,
        "beta": beta_paper, "beta_max": beta_max,
        "coupling_strength": tau, "poisson_ratio": float(matp["poisson"]),
        "lam": lam, "mu": mu, "alpha": alpha, "c0": c0, "kappa": matp["kappa"], "load_rate": load_rate,
        "ndof_flow": int(LHS1.shape[0]), "ndof_mech": int(LHS2.shape[0]),
        "n_free_stress": int(len(sf)), "n_constr_stress": int(len(sc)), "n_free_flux": int(len(wf)),
        "lifting_norm": float(np.linalg.norm(sig_lift_full)),
        "hit_cap": bool(hit_cap), "solve_finite": bool(solve_finite),
        "converged": (not hit_cap) and solve_finite,
        "resid_flow_global": diag["flow_global"], "resid_flow_flux": diag["flow_flux"],
        "resid_flow_pressure": diag["flow_pressure"],
        "resid_mech_global": diag["mech_global"], "resid_mech_stress": diag["mech_stress"],
        "resid_mech_rest": diag["mech_rest"],
        "iters_each": ",".join(map(str, iters_per_step)),
        "iterations_mean": float(np.mean(iters_per_step)),
        "iterations_max": int(np.max(iters_per_step)),
        "solver": SOLVER, "ordering_flow": B1.ordering, "ordering_mech": B2.ordering,
        "ngsolve": ngsolve.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
        "petsc": petsc_version() if SOLVER == "mumps" else "n/a",
    }


# ======================================================================
# Helpers
# ======================================================================
def write_rows_to_csv(rows, csv_file):
    if not rows:
        raise ValueError("No rows to write.")
    if os.path.isfile(csv_file):
        os.remove(csv_file)
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_rows(csv_file):
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found. Run with TABLES_ONLY = False first.")
    with open(csv_file, mode="r", newline="") as f:
        return list(csv.DictReader(f))


def sci_notation_latex(x, digits=3):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return rf"{mantissa:.{digits}f} \times 10^{{{exponent}}}"


def approx_eq(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def report_run(res):
    print(f"    theta={float(res['theta']):5.3f}  beta={float(res['beta']):11.4e}  "
          f"-> iterations per step {res['iters_each']}, mean {float(res['iterations_mean']):.2f}")
    print(f"       residual flow[global/flux/pressure] = {res['resid_flow_global']:.1e}/"
          f"{res['resid_flow_flux']:.1e}/{res['resid_flow_pressure']:.1e}   "
          f"mech[global/stress/rest] = {res['resid_mech_global']:.1e}/"
          f"{res['resid_mech_stress']:.1e}/{res['resid_mech_rest']:.1e}")
    if res["lifting_norm"] == 0.0:
        print("    !! WARNING: lifting norm is zero -- the load was not applied.")
    if not res["solve_finite"]:
        print("    !! ERROR: a direct solve returned NaN/Inf; the counts of this run are meaningless.")
    if max(res["resid_flow_global"], res["resid_flow_pressure"],
           res["resid_mech_global"], res["resid_mech_stress"]) > RESID_WARN:
        print(f"    !! WARNING: reconstruction residual > {RESID_WARN:g}; conditioning may affect the counts.")
    if res["hit_cap"]:
        print("    !! WARNING: a time step hit MAX_ITERATIONS; the mean is not a converged count.")


def check_rows(rows):
    bad = [r for r in rows if str(r["hit_cap"]) == "True" or str(r["solve_finite"]) == "False"]
    if bad:
        print(f"!! WARNING: {len(bad)} run(s) hit the iteration cap or had non-finite solves:")
        for r in bad:
            print(f"   E={r['E_young']}, k={r['k']}, theta={r['theta']}")


# ======================================================================
# Driver
# ======================================================================
def run_beta_grid():
    rows = []
    for E in E_VALUES:
        for k in K_LIST:
            print(f"[footing] E={E:.0e}, k={k}, N={SCAN_N}, NONDIM={NONDIM}, "
                  f"tau = {material_from_E(E)['tau']:.4g}")
            for theta in THETAS:
                res = biot_footing_3d(E_young=E, k=k, theta=theta)
                report_run(res)
                rows.append(res)
    write_rows_to_csv(rows, BETA_GRID_CSV)
    print(f"Footing summary saved to {BETA_GRID_CSV}")
    check_rows(rows)
    return rows


# ======================================================================
# Tables 7 and 8 (booktabs; requires \usepackage{booktabs})
# ======================================================================
def format_theta(th):
    return f"{th:g}"


def make_beta_grid_tables(rows, out_tex):
    def find_row(E, k, th):
        for r in rows:
            if approx_eq(float(r["E_young"]), E) and int(r["k"]) == k and approx_eq(float(r["theta"]), th):
                return r
        raise ValueError(f"Missing row E={E}, k={k}, theta={th}")

    nth = len(THETAS)
    lines = []
    for k in K_LIST:
        lines += [r"\begin{table}[ht!]", r"\centering", r"\renewcommand{\arraystretch}{1.15}",
                  r"\begin{tabular}{" + "c" * (2 + nth) + "}", r"\toprule",
                  r"$E$ & coupling strength $\tau$ & "
                  + rf"\multicolumn{{{nth}}}{{c}}{{$\theta = \beta/\beta_{{\max}}$}} \\",
                  rf"\cmidrule(lr){{3-{2 + nth}}}",
                  " & ".join(["", ""] + [format_theta(th) for th in THETAS]) + r" \\",
                  r"\midrule"]
        for E in E_VALUES:
            vals = [float(find_row(E, k, th)["iterations_mean"]) for th in THETAS]
            vmin = min(vals)
            cells = [f"${sci_notation_latex(E, 0)}$",
                     f"${sci_notation_latex(float(find_row(E, k, THETAS[0])['coupling_strength']))}$"]
            for val in vals:
                sv = f"{val:.2f}"
                cells.append(rf"\textbf{{{sv}}}" if approx_eq(val, vmin) else sv)
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{Three-dimensional footing benchmark with the dimensional parameters of "
                  + REF_TEXT
                  + r": average number of fixed-stress iterations per time step versus the destabilization "
                  + rf"fraction $\theta = \beta/\beta_{{\max}}$ for $k={k}$. Here $\theta = 0$ is the plain split "
                  + r"and $\theta = 1$ the theoretical optimum $\beta_{\max}$; the practical minimum is in bold.}",
                  rf"\label{{tab:footing_dim_beta_grid_3d_k{k}}}", r"\end{table}", ""]

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX tables written to {out_tex}")


def print_summary(rows):
    print("\nAverage iterations per time step (rows E, columns theta):")
    for k in K_LIST:
        print(f"  k = {k}")
        print("    {:>8s} {:>9s} ".format("E", "tau") + " ".join(f"{th:>8g}" for th in THETAS))
        for E in E_VALUES:
            sel = [r for r in rows if approx_eq(float(r["E_young"]), E) and int(r["k"]) == k]
            vals = [float(next(r for r in sel if approx_eq(float(r["theta"]), th))["iterations_mean"])
                    for th in THETAS]
            print(f"    {E:8.0e} {float(sel[0]['coupling_strength']):9.4g} " + " ".join(f"{v:8.2f}" for v in vals))


# ======================================================================
# Main
# ======================================================================
def main():
    print("3D footing problem (Section 6.4, Tables 7 and 8).")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}, PETSc {petsc_version()}")
    print(f"CHEAP_TEST={CHEAP_TEST}, K_LIST={K_LIST}, SCAN_N={SCAN_N}, NONDIM={NONDIM}, "
          f"N_STEPS={N_STEPS}, DT={DT}, TOL={TOL}")
    print(f"E_VALUES={E_VALUES}, THETAS={THETAS}")
    print(f"SOLVER = {SOLVER}"
          + (f" (mat_solver_type={MUMPS_SOLVER_TYPE}, icntl_7={MUMPS_ICNTL_7}, icntl_14={MUMPS_ICNTL_14}, "
             f"out_of_core={MUMPS_OUT_OF_CORE})" if SOLVER == "mumps" else "") + "\n")

    if not TABLES_ONLY:
        rows = run_beta_grid()
    else:
        print("TABLES_ONLY = True: skipping computation, using the csv in results_3d/.")
        rows = load_csv_rows(BETA_GRID_CSV)
        check_rows(rows)

    print_summary(rows)
    make_beta_grid_tables(rows, BETA_GRID_TEX)
    print("\nFooting runs and tables completed.")


if __name__ == "__main__":
    main()