#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-dimensional manufactured-solution convergence study for the Biot splitting scheme
(Section 6.3, Figure 4 of the paper).

  Omega = [0,1]^3, structured tetrahedral N x N x N mesh (h ~ 1/N),
  b(x) = x(1-x) y(1-y) z(1-z),  u = t (1,1,1)^T b,  p = t b,  0 <= t <= 1   (eq. (51)),
  gamma_1 = gamma_2 = 1, beta = 0 (plain fixed-stress split), dt = 1/4, errors at t = 1,
  k = 1 (N = 2,...,32) and k = 2 (N = 2,...,16).

Discretization: three RT_k stress rows, RT_k Darcy flux, discontinuous P_k pressure,
discontinuous P_k^3 displacement, continuous P_k^3 rotation; weak symmetry with
as(sigma) = (s23 - s32, s31 - s13, s12 - s21).
Sources are built by symbolic differentiation (.Diff) of the manufactured u, p.

Linear solves: direct, PETSc/MUMPS with nested-dissection ordering (as stated in the paper);
SOLVER = "splu" falls back to SciPy SuperLU (no petsc4py needed). Both are exact direct
solves, so errors and iteration counts agree up to round-off.

Set PLOT_ONLY = True to regenerate the figures/tables from the csv files in results_3d/.

Author: Tugay Dagli
"""

# ======================================================================
# Imports
# ======================================================================
import ngsolve
from ngsolve import *
import ngsolve.meshes as ngm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import csr_matrix, hstack, vstack
import csv
import os


# ======================================================================
# User settings
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_3d")
FIGURES_DIR = os.path.join(BASE_DIR, "figures_3d")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# CHEAP_TEST = True : tiny meshes (laptop check of the code path)
# CHEAP_TEST = False: meshes of Figure 4 (N = 32 for k = 1 and N = 16 for k = 2 might need the server)
CHEAP_TEST = False

if CHEAP_TEST:
    MESHES_BY_K = {
        1: [2, 4],
        2: [2, 4],
    }
else:
    MESHES_BY_K = {
        1: [2, 4, 8, 16, 32],
        2: [2, 4, 8, 16],
    }

THETA = 0.0             # beta = THETA * beta_max; Figure 4 uses beta = 0
GAMMA = 1.0             # gamma_1
SCALING2 = 1.0          # gamma_2
SAVE_RESULTS = True

USE_TEX = True          # set False on a machine without a LaTeX installation (mathalpha/libus, cm-super)
PLOT_ONLY = False       # True: skip computation, regenerate figures/tables from csv files in results_3d/
INT_ORDER = 12          # quadrature order of the error integrals: exact (exact fields have degree 6, k <= 2)

# ---- direct-solver backend ---------------------------------------------------
#   "mumps" : PETSc + MUMPS, nested-dissection ordering (paper setting)
#   "splu"  : SciPy SuperLU (fallback without petsc4py; memory-hungry in 3D)
SOLVER = "mumps"
MUMPS_SOLVER_TYPE = "mumps"
MUMPS_ICNTL_7 = 5             # sequential ordering: 5 = METIS (nested dissection), 3 = SCOTCH
MUMPS_ICNTL_14 = 40           # workspace headroom in %; raise to 60-100 if MUMPS reports INFO(1) = -9
MUMPS_OUT_OF_CORE = False     # True: factors written to disk (ICNTL 22), last resort for memory

SUMMARY_CSV = os.path.join(RESULTS_DIR, "manufactured_convergence_summary_3d.csv")
SUMMARY_CSV_K1 = os.path.join(RESULTS_DIR, "manufactured_convergence_3d_k1.csv")
SUMMARY_CSV_K2 = os.path.join(RESULTS_DIR, "manufactured_convergence_3d_k2.csv")

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
    N,
    theta=0.0,
    gamma=1.0,
    k=1,
    scaling2=1.0,
    save_results=True,
    csv_file="biot_results_3d.csv",
    int_order=INT_ORDER,
):
    # ------------------------------------------------------------------
    # Mesh, FE spaces
    # ------------------------------------------------------------------
    # Structured tetrahedral mesh on the unit cube [0,1]^3 (hexes=False -> tets).
    mesh = ngm.MakeStructured3DMesh(hexes=False, nx=N, ny=N, nz=N)

    d = 3

    FESHdiv = HDiv(mesh, order=k, RT=True)          # one stress row: RT_k
    FESHdivW = HDiv(mesh, order=k, RT=True)          # Darcy flux: RT_k
    U = VectorL2(mesh, order=k)                      # displacement: discontinuous P_k^3
    P = L2(mesh, order=k)                            # pressure: discontinuous P_k
    FESH1v = VectorH1(mesh, order=k)                 # rotation: continuous P_k^3
    FESHdivS = FESpace([FESHdiv, FESHdiv, FESHdiv])  # total stress (3 rows)

    # Trial/test functions
    sigma1, sigma2, sigma3 = FESHdivS.TrialFunction()
    tau1, tau2, tau3 = FESHdivS.TestFunction()
    u, zt = U.TnT()
    w, v = FESHdivW.TnT()
    p, q = P.TnT()
    omega, eta = FESH1v.TnT()   # eta is the rotation test function (vector)

    # ------------------------------------------------------------------
    # Parameters (gamma = gamma_1, scaling2 = gamma_2)
    # ------------------------------------------------------------------
    lam = scaling2 * 0.6
    mu = 0.6
    alpha = 1.0
    cr = (d * alpha**2) / (2 * mu + d * lam)   # M = d alpha^2 / (2 mu + d lam)
    c0 = gamma
    Kc = gamma / scaling2                        # permeability kappa
    KinvC = scaling2 / gamma                     # kappa^{-1}
    C1 = cr / (d * alpha)                        # alpha~ = alpha / (2 mu + d lam)
    C2 = c0 + cr                                 # c0~ = c0 + M

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
    # Time stepping
    # ------------------------------------------------------------------
    t0 = 0.0
    dtdiv = 4
    dt = 1.0 / dtdiv
    t = Parameter(t0)
    T = Parameter(1.0)

    # ------------------------------------------------------------------
    # Manufactured solution (51) and source terms (symbolic differentiation)
    #
    #   b(x,y,z) = x(1-x) y(1-y) z(1-z)   (vanishes on the whole boundary)
    #   u = t (b, b, b),  p = t b
    #
    # Momentum: -div(theta) + alpha grad(p) = g,  theta = 2 mu eps(u) + lam div(u) I
    #   => g = t * g_s  with g_s the t-independent spatial part.
    # Mass: d/dt(c0 p + alpha div u) + div(w) = f,  w = -kappa grad p
    #   => f = c0 P_s + alpha div(U_s) - kappa t Lap(P_s).
    # ------------------------------------------------------------------
    bs = x * (1 - x) * y * (1 - y) * z * (1 - z)   # spatial part, t-independent
    U0s, U1s, U2s = bs, bs, bs
    Ps = bs

    # spatial strain
    E00 = U0s.Diff(x)
    E11 = U1s.Diff(y)
    E22 = U2s.Diff(z)
    E01 = 0.5 * (U0s.Diff(y) + U1s.Diff(x))
    E02 = 0.5 * (U0s.Diff(z) + U2s.Diff(x))
    E12 = 0.5 * (U1s.Diff(z) + U2s.Diff(y))
    divUs = E00 + E11 + E22

    # spatial effective stress theta
    TH00 = 2 * mu * E00 + lam * divUs
    TH11 = 2 * mu * E11 + lam * divUs
    TH22 = 2 * mu * E22 + lam * divUs
    TH01 = 2 * mu * E01
    TH02 = 2 * mu * E02
    TH12 = 2 * mu * E12

    # momentum source (spatial), g = -div(theta) + alpha grad(p)
    g0s = -(TH00.Diff(x) + TH01.Diff(y) + TH02.Diff(z)) + alpha * Ps.Diff(x)
    g1s = -(TH01.Diff(x) + TH11.Diff(y) + TH12.Diff(z)) + alpha * Ps.Diff(y)
    g2s = -(TH02.Diff(x) + TH12.Diff(y) + TH22.Diff(z)) + alpha * Ps.Diff(z)
    ft = CoefficientFunction((t * g0s, t * g1s, t * g2s))

    # mass source
    LapPs = Ps.Diff(x).Diff(x) + Ps.Diff(y).Diff(y) + Ps.Diff(z).Diff(z)
    gt = c0 * Ps + alpha * divUs - Kc * t * LapPs

    # ------------------------------------------------------------------
    # Grid functions
    # ------------------------------------------------------------------
    gfu1 = GridFunction(FESpace([FESHdivW, P]))                 # [w, p]
    gfusigma = GridFunction(FESHdivS)                           # [s1, s2, s3]
    gfusigma_old = GridFunction(FESHdivS)
    gfup = GridFunction(P)
    gfup_old = GridFunction(P)
    gfu = GridFunction(FESpace([FESHdiv, FESHdiv, FESHdiv, FESH1v, U]))  # [s1,s2,s3,omega,u]

    V = FESpace([FESHdiv, FESHdiv, FESHdiv, U, FESHdivW, P, FESH1v])     # [s1,s2,s3,u,w,p,omega]
    gfucomplete = GridFunction(V)

    # ------------------------------------------------------------------
    # Bilinear forms -> scipy blocks (explicit shapes)
    # ------------------------------------------------------------------
    nW = FESHdivW.ndof
    nP = P.ndof
    nSig = FESHdivS.ndof
    nU = U.ndof
    nRot = FESH1v.ndof

    # A = kappa^{-1} (w, v)
    a = BilinearForm(FESHdivW)
    a += (w * v) * dx
    a.Assemble()
    rows, cols, vals = a.mat.COO()
    A = csr_matrix((vals, (rows, cols)), shape=(nW, nW))
    A = KinvC * A

    # B = -(p, div v)      (test = flux, trial = pressure)
    b = BilinearForm(trialspace=P, testspace=FESHdivW)
    b += -(p * div(v)) * dx
    b.Assemble()
    rows, cols, vals = b.mat.COO()
    B = csr_matrix((vals, (rows, cols)), shape=(nW, nP))

    # C = (p, q)
    c = BilinearForm(P)
    c += (p * q) * dx
    c.Assemble()
    rows, cols, vals = c.mat.COO()
    C = csr_matrix((vals, (rows, cols)), shape=(nP, nP))
    Cmat = C2 * C
    TP = beta * C

    # E = alpha~ (tr sigma, q)   (test = pressure, trial = stress)
    e = BilinearForm(trialspace=FESHdivS, testspace=P)
    e += (sigma1[0] * q + sigma2[1] * q + sigma3[2] * q) * dx
    e.Assemble()
    rows, cols, vals = e.mat.COO()
    E = csr_matrix((vals, (rows, cols)), shape=(nP, nSig))
    E = C1 * E

    # Pmat = (A sigma, tau)   (compliance)
    Pbf = BilinearForm(FESHdivS)
    Pbf += (Asigma1 * tau1 + Asigma2 * tau2 + Asigma3 * tau3) * dx
    Pbf.Assemble()
    rows, cols, vals = Pbf.mat.COO()
    Pmat = csr_matrix((vals, (rows, cols)), shape=(nSig, nSig))

    # R = (u, div tau)   (test = stress, trial = displacement)
    r = BilinearForm(trialspace=U, testspace=FESHdivS)
    r += (div(tau1) * u[0] + div(tau2) * u[1] + div(tau3) * u[2]) * dx
    r.Assemble()
    rows, cols, vals = r.mat.COO()
    R = csr_matrix((vals, (rows, cols)), shape=(nSig, nU))

    # S = weak symmetry (as(sigma), eta),  as(sigma) = ( s23 - s32 , s31 - s13 , s12 - s21 )
    s = BilinearForm(trialspace=FESHdivS, testspace=FESH1v)
    s += ((sigma2[2] - sigma3[1]) * eta[0]
          + (sigma3[0] - sigma1[2]) * eta[1]
          + (sigma1[1] - sigma2[0]) * eta[2]) * dx
    s.Assemble()
    rows, cols, vals = s.mat.COO()
    S = csr_matrix((vals, (rows, cols)), shape=(nRot, nSig))

    # Right-hand-side linear forms
    source1 = LinearForm(P)
    source1 += (gt * q) * dx
    source2 = LinearForm(U)
    source2 += (ft * zt) * dx

    # ------------------------------------------------------------------
    # Assemble the two block operators and factorize once
    # ------------------------------------------------------------------
    LHS1 = vstack([
        hstack([A, B]),
        hstack([-dt * B.T, (Cmat + TP)])
    ]).tocsc()
    B1 = DirectSolver(LHS1, backend=SOLVER, name="LHS1 (flow)")

    ZERO1 = csr_matrix((nW, 1))
    ZERO2 = csr_matrix((nRot, nRot))
    ZERO3 = csr_matrix((nRot, nU))
    ZEROU = csr_matrix((nU, nU))
    ZERO4 = csr_matrix((nRot, 1))

    PTILDE = vstack([
        hstack([Pmat, S.T]),
        hstack([-S, ZERO2])
    ])
    RTILDE = vstack([R, ZERO3])

    LHS2 = vstack([
        hstack([PTILDE, RTILDE]),
        hstack([-RTILDE.T, ZEROU])
    ]).tocsc()
    B2 = DirectSolver(LHS2, backend=SOLVER, name="LHS2 (mechanics)")

    # ------------------------------------------------------------------
    # Time loop with fixed-stress splitting (identical logic to 2D)
    # ------------------------------------------------------------------
    iterations_in_each_time = np.array([])
    tol = 1e-6
    max_iterations = 1000
    time_step = 0
    hit_cap = False

    for n in range(dtdiv):
        t.Set((n + 1) * dt)
        time_step += 1
        print(f"+=====================================+\nTime step {time_step}:")

        # --- initialization of the current time step (i = 1) ---
        # step 1 (flow)
        source1.Assemble()
        add1 = Cmat * gfup_old.vec + TP * gfup_old.vec
        add1 = csr_matrix(add1).transpose()
        RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()
        RESULT1 = B1.solve(RHS1)
        gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
        gfup.vec.data = gfu1.components[1].vec.data

        # step 2 (mechanics)
        add = -E.T * gfup.vec
        add = csr_matrix(add).transpose()
        source2.Assemble()
        RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()
        RESULT2 = B2.solve(RHS2)
        gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)
        gfusigma.components[0].vec.data = gfu.components[0].vec.data
        gfusigma.components[1].vec.data = gfu.components[1].vec.data
        gfusigma.components[2].vec.data = gfu.components[2].vec.data

        # --- fixed-point iteration ---
        prev_iter_RESULT1 = RESULT1
        prev_iter_RESULT2 = RESULT2
        error1 = np.inf
        error2 = np.inf
        iteration = 0

        while (error1 > tol or error2 > tol) and iteration < max_iterations:
            # step 1 (flow)
            source1.Assemble()
            add1 = E * gfusigma_old.vec - E * gfusigma.vec + Cmat * gfup_old.vec + TP * gfup.vec
            add1 = csr_matrix(add1).transpose()
            RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()
            RESULT1 = B1.solve(RHS1)
            gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
            gfup.vec.data = gfu1.components[1].vec.data

            # step 2 (mechanics)
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
            print(f"Iter. {iteration+1} | Residual(flow) {error1}, Residual(mechanics) {error2}")

        if error1 > tol or error2 > tol:
            hit_cap = True
            print(f"!! WARNING: iteration cap {max_iterations} reached at time step {time_step}")

        iterationnumber = iteration + 1
        iterations_in_each_time = np.append(iterations_in_each_time, iterationnumber)

        gfusigma_old.vec.data = gfusigma.vec.data
        gfup_old.vec.data = gfup.vec.data

        # store [s1,s2,s3,u,w,p,omega]
        gfucomplete.components[0].vec.data = gfusigma.components[0].vec.data
        gfucomplete.components[1].vec.data = gfusigma.components[1].vec.data
        gfucomplete.components[2].vec.data = gfusigma.components[2].vec.data
        gfucomplete.components[3].vec.data = gfu.components[4].vec.data     # u
        gfucomplete.components[4].vec.data = gfu1.components[0].vec.data    # w
        gfucomplete.components[5].vec.data = gfup.vec.data                  # p
        gfucomplete.components[6].vec.data = gfu.components[3].vec.data     # omega

        print(f"+=====================================+\n")

    # ------------------------------------------------------------------
    # Error computations (exact fields built symbolically, at t = T = 1)
    # ------------------------------------------------------------------
    u0e, u1e, u2e = T * bs, T * bs, T * bs
    p_exact = T * bs
    u_exact = CoefficientFunction((u0e, u1e, u2e))

    # exact strain / total stress sigma = 2 mu eps(u) + lam div(u) I - alpha p I
    e00 = u0e.Diff(x)
    e11 = u1e.Diff(y)
    e22 = u2e.Diff(z)
    e01 = 0.5 * (u0e.Diff(y) + u1e.Diff(x))
    e02 = 0.5 * (u0e.Diff(z) + u2e.Diff(x))
    e12 = 0.5 * (u1e.Diff(z) + u2e.Diff(y))
    dive = e00 + e11 + e22

    s00 = 2 * mu * e00 + lam * dive - alpha * p_exact
    s11 = 2 * mu * e11 + lam * dive - alpha * p_exact
    s22 = 2 * mu * e22 + lam * dive - alpha * p_exact
    s01 = 2 * mu * e01
    s02 = 2 * mu * e02
    s12 = 2 * mu * e12

    sigmaex_1 = CoefficientFunction((s00, s01, s02))
    sigmaex_2 = CoefficientFunction((s01, s11, s12))
    sigmaex_3 = CoefficientFunction((s02, s12, s22))

    div_sigma1 = s00.Diff(x) + s01.Diff(y) + s02.Diff(z)
    div_sigma2 = s01.Diff(x) + s11.Diff(y) + s12.Diff(z)
    div_sigma3 = s02.Diff(x) + s12.Diff(y) + s22.Diff(z)

    L2errorstress = sqrt(
        Integrate((gfucomplete.components[0] - sigmaex_1)**2, mesh, order=int_order)
        + Integrate((gfucomplete.components[1] - sigmaex_2)**2, mesh, order=int_order)
        + Integrate((gfucomplete.components[2] - sigmaex_3)**2, mesh, order=int_order)
    )
    diverrorstress = sqrt(
        Integrate((div(gfucomplete.components[0]) - div_sigma1)**2, mesh, order=int_order)
        + Integrate((div(gfucomplete.components[1]) - div_sigma2)**2, mesh, order=int_order)
        + Integrate((div(gfucomplete.components[2]) - div_sigma3)**2, mesh, order=int_order)
    )
    Hdiverrorstress = np.sqrt(L2errorstress**2 + diverrorstress**2)

    # exact Darcy flux w = -kappa grad p
    wex = CoefficientFunction((-Kc * p_exact.Diff(x), -Kc * p_exact.Diff(y), -Kc * p_exact.Diff(z)))
    divwex = -Kc * (p_exact.Diff(x).Diff(x) + p_exact.Diff(y).Diff(y) + p_exact.Diff(z).Diff(z))

    L2errorw = sqrt(Integrate((gfucomplete.components[4] - wex)**2, mesh, order=int_order))
    diverrorw = sqrt(Integrate((div(gfucomplete.components[4]) - divwex)**2, mesh, order=int_order))
    Hdiverrorw = np.sqrt(L2errorw**2 + diverrorw**2)

    u_error = sqrt(Integrate((u_exact - gfucomplete.components[3])**2, mesh, order=int_order))
    p_error = sqrt(Integrate((p_exact - gfucomplete.components[5])**2, mesh, order=int_order))

    # coupling strength tau = M / c0
    tau = cr / c0 if c0 > 1e-17 else float("inf")

    print("Error w      L2   :", L2errorw)
    print("Error div(w) L2   :", diverrorw)
    print("Error w      Hdiv :", Hdiverrorw)
    print("Error sigma  L2   :", L2errorstress)
    print("Error div(s) L2   :", diverrorstress)
    print("Error sigma  Hdiv :", Hdiverrorstress)
    print("Error u      L2   :", u_error)
    print("Error p      L2   :", p_error)
    print("Iterations/step   :", iterations_in_each_time)
    print(f"gamma_1 = {gamma}, gamma_2 = {scaling2}, theta = {theta}, beta (paper sign) = {beta_paper}")
    print(f"Coupling strength tau = M/c0 = {tau}")
    print(f"Discrete order k = {k}, N = {N}, solver = {SOLVER}, ordering = {B2.ordering}")
    if hit_cap:
        print("!! WARNING: at least one time step hit the iteration cap; iteration counts are not converged.")

    results = {
        "theta": theta,
        "beta": beta_paper,
        "gamma": gamma,
        "k": k,
        "scaling2": scaling2,
        "N": N,
        "h": 1.0 / N,
        "dt": dt,
        "dtdiv": dtdiv,
        "tol": tol,
        "max_iterations": max_iterations,
        "lam": lam,
        "mu": mu,
        "alpha": alpha,
        "c0": c0,
        "coupling_strength": tau,
        "ndof_flow": int(LHS1.shape[0]),
        "ndof_mech": int(LHS2.shape[0]),
        "w_L2": float(L2errorw),
        "w_div_L2": float(diverrorw),
        "w_hdiv": float(Hdiverrorw),
        "sigma_L2": float(L2errorstress),
        "sigma_div_L2": float(diverrorstress),
        "sigma_hdiv": float(Hdiverrorstress),
        "u_L2": float(u_error),
        "p_L2": float(p_error),
        "iterations_each_time": ",".join(map(str, iterations_in_each_time.astype(int))),
        "iterations_mean": float(np.mean(iterations_in_each_time)),
        "iterations_max": int(np.max(iterations_in_each_time)),
        "hit_cap": hit_cap,
        "quad_order": int_order,
        "solver": SOLVER,
        "ordering": B2.ordering,
        "ngsolve_version": ngsolve.__version__,
        "scipy_version": scipy.__version__,
        "petsc_version": petsc_version() if SOLVER == "mumps" else "n/a",
    }

    if save_results and csv_file is not None:
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        print(f"Results saved to {csv_file}")

    return results


# ======================================================================
# Postprocessing utilities (dimension-agnostic; same as 2D)
# ======================================================================
def estimated_rates(h, err):
    rates = [np.nan]
    for i in range(1, len(h)):
        r = np.log(err[i] / err[i - 1]) / np.log(h[i] / h[i - 1])
        rates.append(r)
    return np.array(rates)


def read_csv_rows(csv_file):
    with open(csv_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError(f"No data rows found in {csv_file}")
    return rows


def sort_rows_by_h_desc(rows):
    return sorted(rows, key=lambda d: float(d["h"]), reverse=True)


def arrays_from_rows(rows):
    rows = sort_rows_by_h_desc(rows)
    h = np.array([float(r["h"]) for r in rows])
    Nvals = np.array([int(r["N"]) for r in rows])
    u_L2 = np.array([float(r["u_L2"]) for r in rows])
    p_L2 = np.array([float(r["p_L2"]) for r in rows])
    w_hdiv = np.array([float(r["w_hdiv"]) for r in rows])
    sigma_hdiv = np.array([float(r["sigma_hdiv"]) for r in rows])
    return Nvals, h, u_L2, p_L2, w_hdiv, sigma_hdiv


def print_convergence_table(csv_file, title=None):
    rows = read_csv_rows(csv_file)
    Nvals, h, u_L2, p_L2, w_hdiv, sigma_hdiv = arrays_from_rows(rows)

    rate_u = estimated_rates(h, u_L2)
    rate_p = estimated_rates(h, p_L2)
    rate_w = estimated_rates(h, w_hdiv)
    rate_sigma = estimated_rates(h, sigma_hdiv)

    if title is not None:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    print("\nConvergence table:")
    print("-" * 90)
    print(f"{'N':>6} {'h':>12} {'u_L2':>14} {'rate_u':>10} {'p_L2':>14} {'rate_p':>10}")
    print("-" * 90)
    for i in range(len(h)):
        ru = "-" if np.isnan(rate_u[i]) else f"{rate_u[i]:.4f}"
        rp = "-" if np.isnan(rate_p[i]) else f"{rate_p[i]:.4f}"
        print(f"{Nvals[i]:6d} {h[i]:12.5e} {u_L2[i]:14.6e} {ru:>10} {p_L2[i]:14.6e} {rp:>10}")

    print("\n")
    print("-" * 90)
    print(f"{'N':>6} {'h':>12} {'w_Hdiv':>14} {'rate_w':>10} {'sigma_Hdiv':>14} {'rate_sigma':>12}")
    print("-" * 90)
    for i in range(len(h)):
        rw = "-" if np.isnan(rate_w[i]) else f"{rate_w[i]:.4f}"
        rs = "-" if np.isnan(rate_sigma[i]) else f"{rate_sigma[i]:.4f}"
        print(f"{Nvals[i]:6d} {h[i]:12.5e} {w_hdiv[i]:14.6e} {rw:>10} {sigma_hdiv[i]:14.6e} {rs:>12}")

    return {
        "Nvals": Nvals, "h": h,
        "u_L2": u_L2, "p_L2": p_L2, "w_hdiv": w_hdiv, "sigma_hdiv": sigma_hdiv,
        "rate_u": rate_u, "rate_p": rate_p, "rate_w": rate_w, "rate_sigma": rate_sigma,
    }


def setup_matplotlib():
    mpl.rcParams["text.usetex"] = USE_TEX
    mpl.rcParams["font.family"] = "serif"
    if USE_TEX:
        mpl.rcParams["text.latex.preamble"] = r"""
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[bb=libus]{mathalpha}
"""
    mpl.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def error_labels():
    """Legend labels; the LaTeX versions match the paper, the fallback works with matplotlib mathtext."""
    if USE_TEX:
        return {
            "u": r"$\|\mathbf{u}-\mathbf{u}_h\|_{0}$",
            "p": r"$\|p-p_h\|_{0}$",
            "w": r"$\|\mathbf{w}-\mathbf{w}_h\|_{{H}(\operatorname{div})}$",
            "sigma": r"$\|\underline{\boldsymbol{\sigma}}-\underline{\boldsymbol{\sigma}}_h\|_{{H}(\operatorname{div})}$",
        }
    return {
        "u": r"$\|\mathbf{u}-\mathbf{u}_h\|_{0}$",
        "p": r"$\|p-p_h\|_{0}$",
        "w": r"$\|\mathbf{w}-\mathbf{w}_h\|_{H(\mathrm{div})}$",
        "sigma": r"$\|\sigma-\sigma_h\|_{H(\mathrm{div})}$",
    }


def _plot_axis(ax, data, k):
    lab = error_labels()
    h = data["h"]
    ax.loglog(h, data["u_L2"], "o-", label=lab["u"])
    ax.loglog(h, data["p_L2"], "s-", label=lab["p"])
    ax.loglog(h, data["w_hdiv"], "^-", label=lab["w"])
    ax.loglog(h, data["sigma_hdiv"], "d-", label=lab["sigma"])
    ref_order = k + 1
    Cref = 3.5
    ax.loglog(h, Cref * (data["u_L2"][-1] / (h[-1]**ref_order)) * h**ref_order, "--",
              label=rf"$O(h^{ref_order})$")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("error")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    ax.invert_xaxis()


def plot_single_convergence(csv_file, k, outfile_pdf, outfile_png=None):
    data = print_convergence_table(csv_file, title=f"3D convergence results for k={k}")
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    _plot_axis(ax, data, k)
    fig.tight_layout()
    fig.savefig(outfile_pdf, format="pdf", bbox_inches="tight")
    if outfile_png is not None:
        fig.savefig(outfile_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_combined_convergence(csv_files, ks, outfile_pdf, outfile_png=None):
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes = np.atleast_1d(axes).ravel()
    subtitles = [r"(a) $k=1$", r"(b) $k=2$"]
    for ax, csv_file, k, subtitle in zip(axes, csv_files, ks, subtitles):
        data = print_convergence_table(csv_file, title=None)
        _plot_axis(ax, data, k)
        ax.set_title(subtitle, y=-0.30)
    fig.tight_layout()
    fig.savefig(outfile_pdf, format="pdf", bbox_inches="tight")
    if outfile_png is not None:
        fig.savefig(outfile_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def fmt_err(x):
    return f"{x:.3e}"


def fmt_rate(x):
    return "-" if np.isnan(x) else f"{x:.2f}"


def latex_table_single(csv_file, filename, caption, label):
    data = print_convergence_table(csv_file, title=None)
    Nvals = data["Nvals"]; h = data["h"]
    u_L2 = data["u_L2"]; p_L2 = data["p_L2"]
    w_hdiv = data["w_hdiv"]; sigma_hdiv = data["sigma_hdiv"]
    rate_u = data["rate_u"]; rate_p = data["rate_p"]
    rate_w = data["rate_w"]; rate_sigma = data["rate_sigma"]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{cccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $h$ & $\|\mathbf{u}-\mathbf{u}_h\|_{0}$ & rate & $\|p-p_h\|_{0}$ & rate & $\|\mathbf{w}-\mathbf{w}_h\|_{H(\operatorname{div})}$ & rate & $\|\underline{\boldsymbol{\sigma}}-\underline{\boldsymbol{\sigma}}_h\|_{H(\operatorname{div})}$ & rate \\")
    lines.append(r"\midrule")
    for i in range(len(Nvals)):
        lines.append(
            f"{Nvals[i]} & {h[i]:.5e} & "
            f"{fmt_err(u_L2[i])} & {fmt_rate(rate_u[i])} & "
            f"{fmt_err(p_L2[i])} & {fmt_rate(rate_p[i])} & "
            f"{fmt_err(w_hdiv[i])} & {fmt_rate(rate_w[i])} & "
            f"{fmt_err(sigma_hdiv[i])} & {fmt_rate(rate_sigma[i])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table written to: {filename}")


# ======================================================================
# Experiment drivers
# ======================================================================
def remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def run_case_list(mesh_list, k, csv_file, theta=0.0, gamma=1.0, scaling2=1.0, save_results=True):
    remove_file_if_exists(csv_file)
    for N in mesh_list:
        print("=" * 100)
        print(f"[3D] manufactured solution: N={N}, k={k}, theta={theta}, gamma_1={gamma}, gamma_2={scaling2}")
        print("=" * 100)
        biot_splitting_3d(
            N=N, theta=theta, gamma=gamma, k=k, scaling2=scaling2,
            save_results=save_results, csv_file=csv_file,
        )


def merge_csvs(csv_files, merged_csv):
    all_rows = []
    for csv_file in csv_files:
        all_rows.extend(read_csv_rows(csv_file))
    if len(all_rows) == 0:
        raise ValueError("No rows to merge.")
    fieldnames = list(all_rows[0].keys())
    remove_file_if_exists(merged_csv)
    with open(merged_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Merged summary csv written to: {merged_csv}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("Starting 3D manufactured-solution convergence study (Figure 4).")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}, PETSc {petsc_version()}")
    print(f"CHEAP_TEST = {CHEAP_TEST}, meshes = {MESHES_BY_K}, quadrature order = {INT_ORDER}")
    print(f"SOLVER = {SOLVER}"
          + (f" (mat_solver_type={MUMPS_SOLVER_TYPE}, icntl_7={MUMPS_ICNTL_7}, icntl_14={MUMPS_ICNTL_14}, "
             f"out_of_core={MUMPS_OUT_OF_CORE})" if SOLVER == "mumps" else ""))

    if not PLOT_ONLY:
        run_case_list(MESHES_BY_K[1], k=1, csv_file=SUMMARY_CSV_K1,
                      theta=THETA, gamma=GAMMA, scaling2=SCALING2, save_results=SAVE_RESULTS)
        run_case_list(MESHES_BY_K[2], k=2, csv_file=SUMMARY_CSV_K2,
                      theta=THETA, gamma=GAMMA, scaling2=SCALING2, save_results=SAVE_RESULTS)
        merge_csvs([SUMMARY_CSV_K1, SUMMARY_CSV_K2], SUMMARY_CSV)
    else:
        print("PLOT_ONLY = True: skipping computation, using csv files in results_3d/.")

    plot_single_convergence(SUMMARY_CSV_K1, 1,
                            os.path.join(FIGURES_DIR, "biot_convergence_3d_k1.pdf"),
                            os.path.join(FIGURES_DIR, "biot_convergence_3d_k1.png"))
    plot_single_convergence(SUMMARY_CSV_K2, 2,
                            os.path.join(FIGURES_DIR, "biot_convergence_3d_k2.pdf"),
                            os.path.join(FIGURES_DIR, "biot_convergence_3d_k2.png"))
    # Figure 4 of the paper
    plot_combined_convergence([SUMMARY_CSV_K1, SUMMARY_CSV_K2], [1, 2],
                              os.path.join(FIGURES_DIR, "biot_convergence_3d_combined.pdf"),
                              os.path.join(FIGURES_DIR, "biot_convergence_3d_combined.png"))

    latex_table_single(SUMMARY_CSV_K1,
                       os.path.join(RESULTS_DIR, "biot_convergence_3d_table_k1.tex"),
                       caption=r"3D convergence rates for $k=1$.",
                       label="tab:biot_convergence_3d_k1")
    latex_table_single(SUMMARY_CSV_K2,
                       os.path.join(RESULTS_DIR, "biot_convergence_3d_table_k2.tex"),
                       caption=r"3D convergence rates for $k=2$.",
                       label="tab:biot_convergence_3d_k2")

    print("3D manufactured-solution convergence study completed.")


if __name__ == "__main__":
    main()