#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-dimensional manufactured-solution parameter study for the iteration tables
(Section 6.3, Tables 5 and 6 of the paper).

  Table 5: varying gamma_1 with fixed gamma_2 = 1
  Table 6: varying gamma_2 with fixed gamma_1 = 1

for k = 1, 2 and the two splitting variants of the paper:
  beta = beta_FS  = 0         (plain fixed-stress split, theta = 0)
  beta = beta_max = M/2       (maximal destabilization of Lemma 3, theta = 1)
with theta := beta / beta_max, M := d alpha^2 / (2 mu + d lambda), d = 3, tau = M / c0.

Same manufactured solution (51), time step dt = 1/4 on [0, 1] and stopping criterion (47)
as the convergence study (main3d.py), on a fixed SCAN_N^3 structured tetrahedral mesh (SCAN_N = 8).
At strong coupling the counts depend moderately on the mesh; MESH_CHECK documents this.

Linear solves: direct, PETSc/MUMPS with nested-dissection ordering (as stated in the paper);
SOLVER = "splu" falls back to SciPy SuperLU.

Flags:
  TABLES_ONLY = True : rebuild both LaTeX tables from the csv files in results_3d/
  MESH_CHECK  = True : additionally rerun the strongest-coupling cases on several meshes
                       (MESH_CHECK_N) and print/store the counts (mesh dependence of the counts)

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
# Paths / settings
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_3d")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_GAMMA_SCAN_CSV = os.path.join(RESULTS_DIR, "manufactured_gamma_scan_summary_3d.csv")
SUMMARY_SCALING2_SCAN_CSV = os.path.join(RESULTS_DIR, "manufactured_scaling2_scan_summary_3d.csv")
MESH_CHECK_CSV = os.path.join(RESULTS_DIR, "manufactured_mesh_check_3d.csv")

TABLE_GAMMA_SCAN_TEX = os.path.join(RESULTS_DIR, "table_gamma_scan_3d.tex")
TABLE_SCALING2_SCAN_TEX = os.path.join(RESULTS_DIR, "table_scaling2_scan_3d.tex")

# CHEAP_TEST = True : tiny lists / mesh, laptop check of the code path
# CHEAP_TEST = False: the full scans of Tables 5 and 6
CHEAP_TEST = False

if CHEAP_TEST:
    KS = [1, 2]
    SCAN_N = 2
    GAMMA1_VALUES = [1.0, 0.1, 0.01]
    GAMMA2_VALUES = [1.0, 10.0, 100.0]
else:
    KS = [1, 2]
    SCAN_N = 8                                                # mesh of Tables 5-6
    GAMMA1_VALUES = [10, 1, 0.1, 0.01, 0.001]                 # Table 5 (gamma_2 = 1)
    GAMMA2_VALUES = [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2]       # Table 6 (gamma_1 = 1)

# splitting variants: label -> theta = beta / beta_max
VARIANTS = {"FS": 0.0, "max": 1.0}
BETA_HEADERS = {"FS": r"$\beta_{\mathrm{FS}}$", "max": r"$\beta_{\max}$"}

TABLES_ONLY = False      # True: skip computation, rebuild tables from csv files in results_3d/
COMPUTE_ERRORS = False   # tables only need iteration counts
VERBOSE_ITER = False     # print every fixed-point iteration
INT_ORDER = 12           # quadrature order for the optional error integrals (exact for k <= 2)

MESH_CHECK = True                        # True: rerun strongest coupling on several meshes
MESH_CHECK_N = [2, 4, 8, 16]
MESH_CHECK_GAMMA1 = [0.01, 0.001]        # gamma_2 = 1

# ---- direct-solver backend ---------------------------------------------------
SOLVER = "mumps"              # "mumps" (paper setting) or "splu"
MUMPS_SOLVER_TYPE = "mumps"
MUMPS_ICNTL_7 = 5             # sequential ordering: 5 = METIS (nested dissection), 3 = SCOTCH
MUMPS_ICNTL_14 = 40           # workspace headroom in %
MUMPS_OUT_OF_CORE = False

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
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("direct solve returned non-finite values")
        return out.reshape(rhs.shape)


# ======================================================================
# Core solver (3D)
# ======================================================================
def biot_splitting_3d(
    theta=0.0,
    gamma=1.0,
    k=1,
    scaling2=1.0,
    N=4,
    beta_label=None,
    compute_errors=COMPUTE_ERRORS,
    verbose_iter=VERBOSE_ITER,
):
    # ------------------------------------------------------------------
    # Mesh, FE spaces
    # ------------------------------------------------------------------
    mesh = ngm.MakeStructured3DMesh(hexes=False, nx=N, ny=N, nz=N)
    d = 3

    FESHdiv = HDiv(mesh, order=k, RT=True)          # one stress row: RT_k
    FESHdivW = HDiv(mesh, order=k, RT=True)          # Darcy flux: RT_k
    U = VectorL2(mesh, order=k)                      # displacement: discontinuous P_k^3
    P = L2(mesh, order=k)                            # pressure: discontinuous P_k
    FESH1v = VectorH1(mesh, order=k)                 # rotation: continuous P_k^3
    FESHdivS = FESpace([FESHdiv, FESHdiv, FESHdiv])  # total stress (3 rows)

    sigma1, sigma2, sigma3 = FESHdivS.TrialFunction()
    tau1, tau2, tau3 = FESHdivS.TestFunction()
    u, zt = U.TnT()
    w, v = FESHdivW.TnT()
    p, q = P.TnT()
    omega, eta = FESH1v.TnT()

    # ------------------------------------------------------------------
    # Parameters (gamma = gamma_1, scaling2 = gamma_2)
    # ------------------------------------------------------------------
    lam = scaling2 * 0.6
    mu = 0.6
    alpha = 1.0
    cr = (d * alpha**2) / (2 * mu + d * lam)    # M = d alpha^2 / (2 mu + d lam)
    c0 = gamma
    Kc = gamma / scaling2                         # permeability kappa
    KinvC = scaling2 / gamma                      # kappa^{-1}
    C1 = cr / (d * alpha)                         # alpha~
    C2 = c0 + cr                                  # c0~

    # destabilization in the paper's sign convention (beta >= 0, Lemma 3);
    # the assembled flow block is (c0~ - beta_paper) * M_p
    beta_max = 0.5 * cr
    beta_paper = theta * beta_max
    beta = -beta_paper

    def tr3(s1, s2, s3):
        return s1[0] + s2[1] + s3[2]

    e1 = CoefficientFunction((1, 0, 0))
    e2 = CoefficientFunction((0, 1, 0))
    e3 = CoefficientFunction((0, 0, 1))

    Asigma1 = (1 / (2 * mu)) * (sigma1 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e1)
    Asigma2 = (1 / (2 * mu)) * (sigma2 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e2)
    Asigma3 = (1 / (2 * mu)) * (sigma3 - (lam / (2 * mu + d * lam)) * tr3(sigma1, sigma2, sigma3) * e3)

    # ------------------------------------------------------------------
    # Time stepping + manufactured sources (symbolic; same as main3d.py)
    # ------------------------------------------------------------------
    t0 = 0.0
    dtdiv = 4
    dt = 1.0 / dtdiv
    t = Parameter(t0)
    T = Parameter(1.0)

    bs = x * (1 - x) * y * (1 - y) * z * (1 - z)
    U0s, U1s, U2s = bs, bs, bs
    Ps = bs

    E00 = U0s.Diff(x); E11 = U1s.Diff(y); E22 = U2s.Diff(z)
    E01 = 0.5 * (U0s.Diff(y) + U1s.Diff(x))
    E02 = 0.5 * (U0s.Diff(z) + U2s.Diff(x))
    E12 = 0.5 * (U1s.Diff(z) + U2s.Diff(y))
    divUs = E00 + E11 + E22

    TH00 = 2 * mu * E00 + lam * divUs
    TH11 = 2 * mu * E11 + lam * divUs
    TH22 = 2 * mu * E22 + lam * divUs
    TH01 = 2 * mu * E01; TH02 = 2 * mu * E02; TH12 = 2 * mu * E12

    g0s = -(TH00.Diff(x) + TH01.Diff(y) + TH02.Diff(z)) + alpha * Ps.Diff(x)
    g1s = -(TH01.Diff(x) + TH11.Diff(y) + TH12.Diff(z)) + alpha * Ps.Diff(y)
    g2s = -(TH02.Diff(x) + TH12.Diff(y) + TH22.Diff(z)) + alpha * Ps.Diff(z)
    ft = CoefficientFunction((t * g0s, t * g1s, t * g2s))

    LapPs = Ps.Diff(x).Diff(x) + Ps.Diff(y).Diff(y) + Ps.Diff(z).Diff(z)
    gt = c0 * Ps + alpha * divUs - Kc * t * LapPs

    # ------------------------------------------------------------------
    # Grid functions
    # ------------------------------------------------------------------
    gfu1 = GridFunction(FESpace([FESHdivW, P]))
    gfusigma = GridFunction(FESHdivS)
    gfusigma_old = GridFunction(FESHdivS)
    gfup = GridFunction(P)
    gfup_old = GridFunction(P)
    gfu = GridFunction(FESpace([FESHdiv, FESHdiv, FESHdiv, FESH1v, U]))

    V = FESpace([FESHdiv, FESHdiv, FESHdiv, U, FESHdivW, P, FESH1v])
    gfucomplete = GridFunction(V)

    # ------------------------------------------------------------------
    # Bilinear forms -> scipy blocks (explicit shapes)
    # ------------------------------------------------------------------
    nW = FESHdivW.ndof
    nP = P.ndof
    nSig = FESHdivS.ndof
    nU = U.ndof
    nRot = FESH1v.ndof

    a = BilinearForm(FESHdivW); a += (w * v) * dx; a.Assemble()
    rows, cols, vals = a.mat.COO()
    A = KinvC * csr_matrix((vals, (rows, cols)), shape=(nW, nW))

    b = BilinearForm(trialspace=P, testspace=FESHdivW); b += -(p * div(v)) * dx; b.Assemble()
    rows, cols, vals = b.mat.COO()
    B = csr_matrix((vals, (rows, cols)), shape=(nW, nP))

    c = BilinearForm(P); c += (p * q) * dx; c.Assemble()
    rows, cols, vals = c.mat.COO()
    C = csr_matrix((vals, (rows, cols)), shape=(nP, nP))
    Cmat = C2 * C
    TP = beta * C

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

    source1 = LinearForm(P); source1 += (gt * q) * dx
    source2 = LinearForm(U); source2 += (ft * zt) * dx

    # ------------------------------------------------------------------
    # Block operators + factorization (once)
    # ------------------------------------------------------------------
    LHS1 = vstack([hstack([A, B]), hstack([-dt * B.T, (Cmat + TP)])]).tocsc()
    B1 = DirectSolver(LHS1, backend=SOLVER, name="LHS1 (flow)")

    ZERO1 = csr_matrix((nW, 1))
    ZERO2 = csr_matrix((nRot, nRot))
    ZERO3 = csr_matrix((nRot, nU))
    ZEROU = csr_matrix((nU, nU))
    ZERO4 = csr_matrix((nRot, 1))

    PTILDE = vstack([hstack([Pmat, S.T]), hstack([-S, ZERO2])])
    RTILDE = vstack([R, ZERO3])
    LHS2 = vstack([hstack([PTILDE, RTILDE]), hstack([-RTILDE.T, ZEROU])]).tocsc()
    B2 = DirectSolver(LHS2, backend=SOLVER, name="LHS2 (mechanics)")

    # ------------------------------------------------------------------
    # Time loop with fixed-stress splitting
    # ------------------------------------------------------------------
    iterations_in_each_time = np.array([])
    tol = 1e-6
    max_iterations = 10000
    hit_cap = False

    for n in range(dtdiv):
        t.Set((n + 1) * dt)

        # initialization of the current time step (i = 1)
        source1.Assemble()
        add1 = Cmat * gfup_old.vec + TP * gfup_old.vec
        add1 = csr_matrix(add1).transpose()
        RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()
        RESULT1 = B1.solve(RHS1)
        gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
        gfup.vec.data = gfu1.components[1].vec.data

        add = -E.T * gfup.vec
        add = csr_matrix(add).transpose()
        source2.Assemble()
        RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()
        RESULT2 = B2.solve(RHS2)
        gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)
        gfusigma.components[0].vec.data = gfu.components[0].vec.data
        gfusigma.components[1].vec.data = gfu.components[1].vec.data
        gfusigma.components[2].vec.data = gfu.components[2].vec.data

        prev_iter_RESULT1 = RESULT1
        prev_iter_RESULT2 = RESULT2
        error1 = np.inf
        error2 = np.inf
        iteration = 0

        while (error1 > tol or error2 > tol) and iteration < max_iterations:
            source1.Assemble()
            add1 = E * gfusigma_old.vec - E * gfusigma.vec + Cmat * gfup_old.vec + TP * gfup.vec
            add1 = csr_matrix(add1).transpose()
            RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()
            RESULT1 = B1.solve(RHS1)
            gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
            gfup.vec.data = gfu1.components[1].vec.data

            add = -E.T * gfup.vec
            add = csr_matrix(add).transpose()
            source2.Assemble()
            RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()
            RESULT2 = B2.solve(RHS2)
            gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)
            gfusigma.components[0].vec.data = gfu.components[0].vec.data
            gfusigma.components[1].vec.data = gfu.components[1].vec.data
            gfusigma.components[2].vec.data = gfu.components[2].vec.data

            # stopping criterion (47): relative update of successive iterates
            error1 = np.linalg.norm(RESULT1 - prev_iter_RESULT1) / (np.linalg.norm(prev_iter_RESULT1) + 1e-14)
            error2 = np.linalg.norm(RESULT2 - prev_iter_RESULT2) / (np.linalg.norm(prev_iter_RESULT2) + 1e-14)

            prev_iter_RESULT1 = RESULT1
            prev_iter_RESULT2 = RESULT2
            iteration += 1
            if verbose_iter:
                print(f"    iter {iteration+1} | flow {error1:.2e}, mech {error2:.2e}")

        if error1 > tol or error2 > tol:
            hit_cap = True
            print(f"!! WARNING: iteration cap {max_iterations} reached at time step {n + 1}")

        iterations_in_each_time = np.append(iterations_in_each_time, iteration + 1)

        gfusigma_old.vec.data = gfusigma.vec.data
        gfup_old.vec.data = gfup.vec.data

        gfucomplete.components[0].vec.data = gfusigma.components[0].vec.data
        gfucomplete.components[1].vec.data = gfusigma.components[1].vec.data
        gfucomplete.components[2].vec.data = gfusigma.components[2].vec.data
        gfucomplete.components[3].vec.data = gfu.components[4].vec.data
        gfucomplete.components[4].vec.data = gfu1.components[0].vec.data
        gfucomplete.components[5].vec.data = gfup.vec.data
        gfucomplete.components[6].vec.data = gfu.components[3].vec.data

    # ------------------------------------------------------------------
    # Derived scalars
    # ------------------------------------------------------------------
    tau = cr / c0 if c0 > 1e-17 else float("inf")   # coupling strength tau = M / c0
    poisson_ratio = float(lam / (2 * (lam + mu)))

    # ------------------------------------------------------------------
    # Optional error computation (tables do not need it)
    # ------------------------------------------------------------------
    err_fields = {kk: float("nan") for kk in ["u_L2_abs", "p_L2_abs", "w_hdiv_abs", "sigma_hdiv_abs"]}
    if compute_errors:
        int_order = INT_ORDER
        u0e, u1e, u2e = T * bs, T * bs, T * bs
        p_exact = T * bs
        u_exact = CoefficientFunction((u0e, u1e, u2e))

        e00 = u0e.Diff(x); e11 = u1e.Diff(y); e22 = u2e.Diff(z)
        e01 = 0.5 * (u0e.Diff(y) + u1e.Diff(x))
        e02 = 0.5 * (u0e.Diff(z) + u2e.Diff(x))
        e12 = 0.5 * (u1e.Diff(z) + u2e.Diff(y))
        dive = e00 + e11 + e22

        s00 = 2 * mu * e00 + lam * dive - alpha * p_exact
        s11 = 2 * mu * e11 + lam * dive - alpha * p_exact
        s22 = 2 * mu * e22 + lam * dive - alpha * p_exact
        s01 = 2 * mu * e01; s02 = 2 * mu * e02; s12 = 2 * mu * e12

        sigmaex_1 = CoefficientFunction((s00, s01, s02))
        sigmaex_2 = CoefficientFunction((s01, s11, s12))
        sigmaex_3 = CoefficientFunction((s02, s12, s22))
        dS1 = s00.Diff(x) + s01.Diff(y) + s02.Diff(z)
        dS2 = s01.Diff(x) + s11.Diff(y) + s12.Diff(z)
        dS3 = s02.Diff(x) + s12.Diff(y) + s22.Diff(z)

        L2s = sqrt(Integrate((gfucomplete.components[0] - sigmaex_1)**2, mesh, order=int_order)
                   + Integrate((gfucomplete.components[1] - sigmaex_2)**2, mesh, order=int_order)
                   + Integrate((gfucomplete.components[2] - sigmaex_3)**2, mesh, order=int_order))
        divs = sqrt(Integrate((div(gfucomplete.components[0]) - dS1)**2, mesh, order=int_order)
                    + Integrate((div(gfucomplete.components[1]) - dS2)**2, mesh, order=int_order)
                    + Integrate((div(gfucomplete.components[2]) - dS3)**2, mesh, order=int_order))
        wex = CoefficientFunction((-Kc * p_exact.Diff(x), -Kc * p_exact.Diff(y), -Kc * p_exact.Diff(z)))
        divwex = -Kc * (p_exact.Diff(x).Diff(x) + p_exact.Diff(y).Diff(y) + p_exact.Diff(z).Diff(z))
        L2w = sqrt(Integrate((gfucomplete.components[4] - wex)**2, mesh, order=int_order))
        divw = sqrt(Integrate((div(gfucomplete.components[4]) - divwex)**2, mesh, order=int_order))
        err_fields["u_L2_abs"] = float(sqrt(Integrate((u_exact - gfucomplete.components[3])**2, mesh, order=int_order)))
        err_fields["p_L2_abs"] = float(sqrt(Integrate((p_exact - gfucomplete.components[5])**2, mesh, order=int_order)))
        err_fields["w_hdiv_abs"] = float(np.sqrt(L2w**2 + divw**2))
        err_fields["sigma_hdiv_abs"] = float(np.sqrt(L2s**2 + divs**2))

    results = {
        "beta_label": beta_label,
        "theta": theta,
        "beta": beta_paper,
        "beta_max": beta_max,
        "gamma": gamma,
        "k": k,
        "scaling2": scaling2,
        "N": N,
        "dt": dt,
        "dtdiv": dtdiv,
        "tol": tol,
        "max_iterations": max_iterations,
        "lam": lam,
        "mu": mu,
        "poisson_ratio": poisson_ratio,
        "alpha": alpha,
        "c0": c0,
        "coupling_strength": tau,
        **err_fields,
        "iterations_each_time": ",".join(map(str, iterations_in_each_time.astype(int))),
        "iterations_mean": float(np.mean(iterations_in_each_time)),
        "iterations_max": int(np.max(iterations_in_each_time)),
        "hit_cap": hit_cap,
        "solver": SOLVER,
        "ordering": B2.ordering,
        "ngsolve_version": ngsolve.__version__,
        "scipy_version": scipy.__version__,
        "petsc_version": petsc_version() if SOLVER == "mumps" else "n/a",
    }
    return results


# ======================================================================
# Helpers
# ======================================================================
def remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def write_rows_to_csv(rows, csv_file):
    if len(rows) == 0:
        raise ValueError("No rows to write.")
    remove_file_if_exists(csv_file)
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


def format_value(g):
    if abs(g - round(g)) < 1e-14 and g >= 1:
        return f"{int(round(g))}"
    return f"{g:g}"


def format_poisson_ratio(v):
    s = f"{v:.5f}"
    return s.rstrip("0").rstrip(".")


def approx_eq(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def check_hit_cap(rows, name):
    capped = [r for r in rows if str(r["hit_cap"]) == "True"]
    if capped:
        print(f"!! WARNING ({name}): {len(capped)} run(s) hit the iteration cap; their means are not converged counts:")
        for r in capped:
            print(f"   gamma_1={r['gamma']}, gamma_2={r['scaling2']}, k={r['k']}, beta={r['beta_label']}")


# ======================================================================
# Scan drivers
# ======================================================================
def run_scan(gamma_values, scaling2_values, out_csv, name):
    rows = []
    for gamma in gamma_values:
        for scaling2 in scaling2_values:
            for k in KS:
                for label, theta in VARIANTS.items():
                    print(f"[{name}] gamma_1={gamma}, gamma_2={scaling2}, k={k}, beta=beta_{label} (theta={theta}), N={SCAN_N}")
                    res = biot_splitting_3d(theta=theta, gamma=gamma, k=k, scaling2=scaling2,
                                            N=SCAN_N, beta_label=label)
                    print(f"    -> iterations per step {res['iterations_each_time']}, mean {res['iterations_mean']:.2f} "
                          f"(tau = {res['coupling_strength']:.4g})")
                    rows.append(res)
    write_rows_to_csv(rows, out_csv)
    print(f"{name} summary saved to {out_csv}")
    check_hit_cap(rows, name)


def run_mesh_check():
    """Strongest-coupling cases of Table 5 on several meshes (mesh dependence of the counts)."""
    rows = []
    for gamma in MESH_CHECK_GAMMA1:
        for k in KS:
            for label, theta in VARIANTS.items():
                for N in MESH_CHECK_N:
                    print(f"[mesh check] gamma_1={gamma}, gamma_2=1, k={k}, beta=beta_{label}, N={N}")
                    res = biot_splitting_3d(theta=theta, gamma=gamma, k=k, scaling2=1.0, N=N, beta_label=label)
                    rows.append(res)
    write_rows_to_csv(rows, MESH_CHECK_CSV)
    print("\nMesh check (average iterations per time step, gamma_2 = 1):")
    print("  gamma_1 | k | beta  | " + " | ".join(f"N={N:<3d}" for N in MESH_CHECK_N))
    for gamma in MESH_CHECK_GAMMA1:
        for k in KS:
            for label in VARIANTS:
                vals = [float(r["iterations_mean"]) for r in rows
                        if approx_eq(float(r["gamma"]), gamma) and int(r["k"]) == k and r["beta_label"] == label]
                print(f"  {gamma:7g} | {k} | {label:<5s} | " + " | ".join(f"{v:6.2f}" for v in vals))
    print(f"Mesh check saved to {MESH_CHECK_CSV}")


# ======================================================================
# Table generation (booktabs; requires \usepackage{booktabs})
# ======================================================================
def make_scan_table(summary_csv, out_tex, scan_key, var_order, include_poisson, caption, label):
    rows = load_csv_rows(summary_csv)
    check_hit_cap(rows, scan_key)

    def find_row(val, k, blabel):
        for row in rows:
            if approx_eq(float(row[scan_key]), val) and int(row["k"]) == k and row["beta_label"] == blabel:
                return row
        raise ValueError(f"Missing row for {scan_key}={val}, k={k}, beta_{blabel}")

    nvar = len(VARIANTS)
    n_lead = 3 if include_poisson else 2
    ncols = n_lead + nvar * len(KS)

    if include_poisson:
        lead1 = [r"$\gamma_2$", r"Poisson ratio", r"coupling strength $\tau$"]
    else:
        lead1 = [r"$\gamma_1$", r"coupling strength $\tau$"]
    head1 = lead1 + [rf"\multicolumn{{{nvar}}}{{c}}{{$k={k}$}}" for k in KS]

    cmid = ""
    col = n_lead + 1
    for _ in KS:
        cmid += rf"\cmidrule(lr){{{col}-{col + nvar - 1}}}"
        col += nvar
    head2 = [""] * n_lead + [BETA_HEADERS[lbl] for _ in KS for lbl in VARIANTS]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{" + "c" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(head1) + r" \\")
    lines.append(cmid)
    lines.append(" & ".join(head2) + r" \\")
    lines.append(r"\midrule")

    for val in var_order:
        ref = find_row(val, KS[0], "FS")
        cells = [format_value(val)]
        if include_poisson:
            cells.append(format_poisson_ratio(float(ref["poisson_ratio"])))
        cells.append(f"${sci_notation_latex(float(ref['coupling_strength']))}$")
        for k in KS:
            for lbl in VARIANTS:
                cells.append(f"{float(find_row(val, k, lbl)['iterations_mean']):.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table written to {out_tex}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("Starting 3D manufactured-solution parameter-study runs (Tables 5 and 6).")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}, PETSc {petsc_version()}")
    print(f"CHEAP_TEST={CHEAP_TEST}, KS={KS}, SCAN_N={SCAN_N}")
    print(f"SOLVER = {SOLVER}"
          + (f" (mat_solver_type={MUMPS_SOLVER_TYPE}, icntl_7={MUMPS_ICNTL_7}, icntl_14={MUMPS_ICNTL_14}, "
             f"out_of_core={MUMPS_OUT_OF_CORE})" if SOLVER == "mumps" else ""))
    print(f"GAMMA1_VALUES={GAMMA1_VALUES}")
    print(f"GAMMA2_VALUES={GAMMA2_VALUES}\n")

    if not TABLES_ONLY:
        run_scan(GAMMA1_VALUES, [1.0], SUMMARY_GAMMA_SCAN_CSV, "gamma_1 scan")
        run_scan([1.0], GAMMA2_VALUES, SUMMARY_SCALING2_SCAN_CSV, "gamma_2 scan")
        if MESH_CHECK:
            run_mesh_check()
    else:
        print("TABLES_ONLY = True: skipping computation, using csv files in results_3d/.")

    make_scan_table(
        SUMMARY_GAMMA_SCAN_CSV, TABLE_GAMMA_SCAN_TEX,
        scan_key="gamma", var_order=GAMMA1_VALUES, include_poisson=False,
        caption=r"Three-dimensional manufactured solution: average number of iterations per time step for varying $\gamma_1$ with fixed $\gamma_2 = 1$.",
        label="tab:manufactured_gamma_scan_3d",
    )
    make_scan_table(
        SUMMARY_SCALING2_SCAN_CSV, TABLE_SCALING2_SCAN_TEX,
        scan_key="scaling2", var_order=GAMMA2_VALUES, include_poisson=True,
        caption=r"Three-dimensional manufactured solution: average number of iterations per time step for varying $\gamma_2$ with fixed $\gamma_1 = 1$.",
        label="tab:manufactured_scaling2_scan_3d",
    )

    print("\nAll 3D scan runs and tables completed.")


if __name__ == "__main__":
    main()
