#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manufactured-solution convergence study for the Biot splitting scheme
(Section 6.1, Figure 1 of the paper).

  1. run the manufactured-solution experiments for multiple mesh sizes N,
  2. collect all outputs in csv files,
  3. compute experimental convergence rates,
  4. generate single-panel plots for k=1,2,3,
  5. generate one combined multi-panel figure (Figure 1).

Set PLOT_ONLY = True to skip the computation and regenerate the figures and
tables from the csv files already present in results/.

Author: Tugay Dagli
"""

# ======================================================================
# Imports
# ======================================================================
import ngsolve
from ngsolve import *
import ngsolve.meshes as ngm
import numpy as np
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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Mesh lists used for Figure 1 (N x N structured triangulation, h = 1/N).
# These are only run lists, not mathematical changes to the method.
MESHES_BY_K = {
    1: [8, 16, 32, 64],
    2: [4, 8, 16, 32],
    3: [2, 4, 8, 16],  # check against Figure 1(c) of the paper
}

# Extra refined run for the fourth panel, Figure 1(d)
REFINED_MESHES_K3 = [2, 4, 8, 12, 16, 24, 32, 48]

TUNING = "off"      # "off": beta = beta_FS = 0 (Figure 1); "on": beta = beta_max = M/2
GAMMA = 1.0         # gamma_1 in the paper
SCALING2 = 1.0      # gamma_2 in the paper
SAVE_RESULTS = True

USE_TEX = True      # set False if no LaTeX installation (with mathalpha/libus) is available
PLOT_ONLY = False   # True: skip computation, regenerate figures/tables from csv files in results/

QUAD = 10           # quadrature order for all error integrals; exact for squared integrands up to k=3

SUMMARY_CSV = os.path.join(RESULTS_DIR, "manufactured_convergence_summary.csv")
SUMMARY_CSV_K1 = os.path.join(RESULTS_DIR, "manufactured_convergence_k1.csv")
SUMMARY_CSV_K2 = os.path.join(RESULTS_DIR, "manufactured_convergence_k2.csv")
SUMMARY_CSV_K3 = os.path.join(RESULTS_DIR, "manufactured_convergence_k3.csv")
SUMMARY_CSV_K3_REFINED = os.path.join(RESULTS_DIR, "manufactured_convergence_k3_refined.csv")


# ======================================================================
# Core solver
# ======================================================================
def biot_splitting(
    N,
    tuning="off",
    gamma=1,
    k=1,
    scaling2=1,
    save_results=True,
    csv_file="biot_results.csv"
):
    # Mesh, FE Spaces, Parameters
    mesh = ngm.MakeStructured2DMesh(quads=False, nx=N, ny=N, flip_triangles=True)

    FESHdiv = HDiv(mesh, order=k, RT=True)    # stress rows: RT_k
    FESHdivW = HDiv(mesh, order=k, RT=True)   # Darcy flux: RT_k
    U = VectorL2(mesh, order=k)               # displacement: discontinuous P_k^2
    P = L2(mesh, order=k)                     # pressure: discontinuous P_k
    FESH1 = H1(mesh, order=k)                 # rotation: continuous P_k
    FESHdiv2 = FESpace([FESHdiv, FESHdiv])

    # Trial and Test Functions
    sigma1, sigma2 = FESHdiv2.TrialFunction()
    tau1, tau2 = FESHdiv2.TestFunction()
    u, z = U.TnT()
    w, v = FESHdivW.TnT()
    p, q = P.TnT()
    omegascalar, gammascalar = FESH1.TnT()

    # Parameters with scaling gamma (gamma_1) and scaling2 (gamma_2)
    lam = scaling2 * 0.6
    mu = 0.6
    d = 2
    alpha = 1
    cr = (d * alpha**2) / (2 * mu + d * lam)   # M = d alpha^2 / (2 mu + d lam)
    c0 = gamma
    Kc = (gamma / scaling2)                    # permeability kappa
    KinvC = scaling2 / gamma                   # kappa^{-1}
    C1 = cr / (d * alpha)                      # alpha~ = alpha / (2 mu + d lam)
    C2 = c0 + cr                               # c0~ = c0 + M

    # Tuning: beta_paper >= 0 as in Lemma 3 / eq. (35);
    # the assembled flow block is (c0~ - beta_paper) * M_p
    if tuning == "off":
        beta_paper = 0.0
    elif tuning == "on":
        beta_paper = 0.5 * cr   # beta_max = M/2
    else:
        raise ValueError("Invalid value for tuning parameter. Should be 'on' or 'off'.")
    beta = -beta_paper

    def tr(sig1, sig2):
        return sig1[0] + sig2[1]

    Asigma1 = ((1 / (2 * mu)) * (sigma1 - (lam / (2 * mu + d * lam)) * tr(sigma1, sigma2) * CoefficientFunction((1, 0))))
    Asigma2 = ((1 / (2 * mu)) * (sigma2 - (lam / (2 * mu + d * lam)) * tr(sigma1, sigma2) * CoefficientFunction((0, 1))))

    # initial time, dt, t as parameter and T final time as parameter
    t0 = 0.0
    dtdiv = 4
    dt = 1 / dtdiv
    t = Parameter(t0)
    T = Parameter(1.0)

    # source terms: g = -div sigma (momentum), gt = mass source
    f1 = alpha * y * (1 - y) * (1 - 2 * x) + (4 * mu + 2 * lam) * y * (1 - y) - lam * (1 - 2 * y) * (1 - 2 * x) - mu * (-2 * x * (1 - x) + (1 - 2 * x) * (1 - 2 * y))
    f2 = alpha * x * (1 - x) * (1 - 2 * y) + (4 * mu + 2 * lam) * x * (1 - x) - lam * (1 - 2 * y) * (1 - 2 * x) - mu * (-2 * y * (1 - y) + (1 - 2 * x) * (1 - 2 * y))
    ft = CoefficientFunction((t * f1, t * f2))
    gt = c0 * x * (1 - x) * y * (1 - y) + alpha * (y * (1 - y) * (1 - 2 * x) + x * (1 - x) * (1 - 2 * y)) + Kc * 2 * t * (y * (1 - y) + x * (1 - x))

    # grid functions (solution variables)
    gfu1 = GridFunction(FESpace([FESHdivW, P]))

    gfusigma = GridFunction(FESHdiv2)
    gfusigma_old = GridFunction(FESHdiv2)

    gfup = GridFunction(P)
    gfup_old = GridFunction(P)

    gfu = GridFunction(FESpace([FESHdiv, FESHdiv, FESH1, U]))
    V = FESpace([FESHdiv, FESHdiv, U, FESHdivW, P, FESH1])
    gfucomplete = GridFunction(V)
    gfucompleteT = GridFunction(gfucomplete.space, multidim=0)

    # bilinear forms and related matrices
    a = BilinearForm(FESHdivW)
    a += (w * v) * dx
    a.Assemble()

    rows, cols, vals = a.mat.COO()
    A = sp.csr_matrix((vals, (rows, cols)))
    A = KinvC * A

    b = BilinearForm(trialspace=P, testspace=FESHdivW)
    b += -(p * div(v)) * dx
    b.Assemble()

    rows, cols, vals = b.mat.COO()
    B = sp.csr_matrix((vals, (rows, cols)))

    c = BilinearForm(P)
    c += ((p * q)) * dx
    c.Assemble()

    rows, cols, vals = c.mat.COO()
    C = sp.csr_matrix((vals, (rows, cols)))
    Cmat = C2 * C
    TP = beta * C

    e = BilinearForm(trialspace=FESHdiv2, testspace=P)
    e += ((sigma1[0] * q + sigma2[1] * q)) * dx
    e.Assemble()

    rows, cols, vals = e.mat.COO()
    E = sp.csr_matrix((vals, (rows, cols)))
    E = C1 * E

    source1 = LinearForm(P)
    source1 += ((gt * q) * dx)
    source2 = LinearForm(U)
    source2 += ((ft * z) * dx)

    Pbf = BilinearForm(FESHdiv2)
    Pbf += (Asigma1 * tau1 + Asigma2 * tau2) * dx
    Pbf.Assemble()

    rows, cols, vals = Pbf.mat.COO()
    Pmat = sp.csr_matrix((vals, (rows, cols)))

    r = BilinearForm(trialspace=U, testspace=FESHdiv2)
    r += (div(tau1) * u[0] + div(tau2) * u[1]) * dx
    r.Assemble()

    rows, cols, vals = r.mat.COO()
    R = sp.csr_matrix((vals, (rows, cols)))

    s = BilinearForm(trialspace=FESHdiv2, testspace=FESH1)
    s += ((sigma2[0] - sigma1[1]) * gammascalar) * dx
    s.Assemble()

    rows, cols, vals = s.mat.COO()
    S = sp.csr_matrix((vals, (rows, cols)))

    # LHS matrices for each step
    LHS1 = vstack([
        hstack([A, B]),
        hstack([-dt * B.T, (Cmat + TP)])
    ]).tocsc()

    B1 = sp.linalg.splu(LHS1)
    Vaux = FESpace([FESHdiv, FESHdiv, FESH1])

    ZERO1 = sp.csr_matrix((sum(FESHdivW.FreeDofs()), 1))
    ZERO2 = sp.csr_matrix(((sum(Vaux.FreeDofs()) - sum(FESHdiv2.FreeDofs())), sum(Vaux.FreeDofs()) - sum(FESHdiv2.FreeDofs())))
    ZERO3 = sp.csr_matrix(((sum(U.FreeDofs())), sum(Vaux.FreeDofs()) - sum(FESHdiv2.FreeDofs())))
    ZERO3 = ZERO3.T
    ZEROU = sp.csr_matrix(((sum(U.FreeDofs())), sum(U.FreeDofs())))
    ZERO4 = sp.csr_matrix((sum(FESH1.FreeDofs()), 1))

    PTILDE = vstack([
        hstack([Pmat, S.T]),
        hstack([-S, ZERO2])
    ])

    RTILDE = vstack([R, ZERO3])

    LHS2 = vstack([
        hstack([PTILDE, RTILDE]),
        hstack([-RTILDE.T, ZEROU])
    ]).tocsc()
    B2 = sp.linalg.splu(LHS2)

    iterations_in_each_time = np.array([])
    tol = 1e-6
    max_iterations = 1000
    time_step = 0
    hit_cap = False

    for n in range(dtdiv):
        t.Set((n + 1) * dt)
        time_step += 1
        print(f"+=====================================+\nTime step {time_step}:")

        # INITIALIZATION OF CURRENT ITERATION (i = 1, initialized with previous time step data)

        # step 1
        source1.Assemble()
        add1 = Cmat * gfup_old.vec + TP * gfup_old.vec
        add1 = csr_matrix(add1).transpose()
        RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()

        # solve for w and p
        RESULT1 = B1.solve(RHS1)

        # store in [w, p]
        gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)

        # update p
        gfup.vec.data = gfu1.components[1].vec.data

        # step 2
        add = -E.T * gfup.vec
        add = csr_matrix(add).transpose()
        source2.Assemble()
        RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()

        # solve for sigma, omega, u
        RESULT2 = B2.solve(RHS2)

        # store in [sigma, omega, u]
        gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)

        # update sigma
        gfusigma.components[0].vec.data = gfu.components[0].vec.data
        gfusigma.components[1].vec.data = gfu.components[1].vec.data

        # stopping criterion setup
        prev_iter_RESULT1 = RESULT1
        prev_iter_RESULT2 = RESULT2
        error1 = np.inf
        error2 = np.inf
        iteration = 0

        while (error1 > tol or error2 > tol) and iteration < max_iterations:
            # step 1
            source1.Assemble()
            add1 = E * gfusigma_old.vec - E * gfusigma.vec + Cmat * gfup_old.vec + TP * gfup.vec
            add1 = csr_matrix(add1).transpose()
            RHS1 = vstack([ZERO1, dt * csr_matrix(source1.vec.FV()).transpose() + add1]).toarray()

            # solve for w and p
            RESULT1 = B1.solve(RHS1)

            # store in [w, p]
            gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)

            # update p
            gfup.vec.data = gfu1.components[1].vec.data

            # step 2
            add = -E.T * gfup.vec
            add = csr_matrix(add).transpose()
            source2.Assemble()
            RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()

            # solve for sigma, omega, u
            RESULT2 = B2.solve(RHS2)

            # store in [sigma, omega, u]
            gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)

            # update sigma
            gfusigma.components[0].vec.data = gfu.components[0].vec.data
            gfusigma.components[1].vec.data = gfu.components[1].vec.data

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

        # accepted solution
        iterationnumber = iteration + 1
        iterations_in_each_time = np.append(iterations_in_each_time, iterationnumber)

        gfusigma_old.vec.data = gfusigma.vec.data
        gfup_old.vec.data = gfup.vec.data

        # store everything in one [sigma, u, w, omega, p]
        gfucomplete.components[0].vec.data = gfusigma.components[0].vec.data
        gfucomplete.components[1].vec.data = gfusigma.components[1].vec.data
        gfucomplete.components[2].vec.data = gfu.components[3].vec.data
        gfucomplete.components[3].vec.data = gfu1.components[0].vec.data
        gfucomplete.components[4].vec.data = gfup.vec.data
        gfucomplete.components[5].vec.data = gfu.components[2].vec.data

        gfucompleteT.AddMultiDimComponent(gfucomplete.vec)
        print(f"+=====================================+\n")

    # ERROR COMPUTATIONS (at final time T = 1)
    u_exact = CoefficientFunction((T * x * (1 - x) * y * (1 - y), T * x * (1 - x) * y * (1 - y)))
    p_exact = CoefficientFunction(T * x * (1 - x) * y * (1 - y))
    trace_eps = T * y * (1 - y) * (1 - 2 * x) + T * x * (1 - x) * (1 - 2 * y)

    sigma11 = 2 * mu * T * y * (1 - y) * (1 - 2 * x) + lam * trace_eps - alpha * p_exact
    sigma22 = 2 * mu * T * x * (1 - x) * (1 - 2 * y) + lam * trace_eps - alpha * p_exact
    sigma12 = mu * T * (x * (1 - x) * (1 - 2 * y) + y * (1 - y) * (1 - 2 * x))

    sigmaex_1 = CoefficientFunction((sigma11, sigma12))
    sigmaex_2 = CoefficientFunction((sigma12, sigma22))

    div_sigma1 = T * (-4 * mu * y * (1 - y) - 2 * lam * y * (1 - y) + lam * (1 - 2 * x) * (1 - 2 * y) - alpha * (1 - 2 * x) * y * (1 - y)) + T * mu * (-2 * x * (1 - x) + (1 - 2 * y) * (1 - 2 * x))
    div_sigma2 = T * mu * (-2 * y * (1 - y) + (1 - 2 * y) * (1 - 2 * x)) + T * (-4 * mu * x * (1 - x) + lam * (1 - 2 * x) * (1 - 2 * y) - 2 * lam * x * (1 - x) - alpha * (1 - 2 * y) * x * (1 - x))

    L2errorstress = sqrt(Integrate((gfucomplete.components[0] - sigmaex_1)**2, mesh, order=QUAD)
                         + Integrate((gfucomplete.components[1] - sigmaex_2)**2, mesh, order=QUAD))
    diverrorstress = sqrt(Integrate((div(gfucomplete.components[0]) - div_sigma1)**2, mesh, order=QUAD)
                          + Integrate((div(gfucomplete.components[1]) - div_sigma2)**2, mesh, order=QUAD))

    Hdiverrorstress = np.sqrt(L2errorstress**2 + diverrorstress**2)

    wex = CoefficientFunction((-Kc * T * (1 - 2 * x) * y * (1 - y), -Kc * T * (1 - 2 * y) * x * (1 - x)))
    divwex = 2 * Kc * T * (x * (1 - x) + y * (1 - y))

    L2errorw = sqrt(Integrate((gfucomplete.components[3] - wex)**2, mesh, order=QUAD))
    diverrorw = sqrt(Integrate((div(gfucomplete.components[3]) - divwex)**2, mesh, order=QUAD))

    Hdiverrorw = np.sqrt(L2errorw**2 + diverrorw**2)

    u_error = sqrt(Integrate((u_exact - gfucomplete.components[2])**2, mesh, order=QUAD))
    p_error = sqrt(Integrate((p_exact - gfucomplete.components[4])**2, mesh, order=QUAD))

    print("Error between approximated and exact w in L2 norm:", L2errorw)
    print("Error between approximated and exact div(w) in L2 norm", diverrorw)
    print("Error between approximated and exact w in hdiv norm:", Hdiverrorw)

    print("Error between approximated and exact sigma in L2 norm:", L2errorstress)
    print("Error between approximated and exact div(sigma) in L2 norm:", diverrorstress)
    print("Error between approximated and exact sigma in hdiv norm:", Hdiverrorstress)

    print("Error between approximated and exact u in L2 norm:", u_error)
    print("Error between approximated and exact p in L2 norm:", p_error)
    print("Iterations in each time step:", iterations_in_each_time)
    print("Tolerance:", tol)
    print("gamma (gamma_1):", gamma)
    print("scaling2 (gamma_2):", scaling2)
    print("beta (paper convention, >= 0):", beta_paper)

    def coupling_strength():
        # tau = d alpha^2 / (c0 (2 mu + d lam)) = M / c0
        return cr / c0 if c0 > 1e-17 else float("inf")

    tau = coupling_strength()
    print(f"The coupling strength is {tau}.")
    print(f"The discrete space order is {k}.")
    if hit_cap:
        print("!! WARNING: at least one time step hit the iteration cap; iteration counts are not converged.")

    results = {
        "tuning": tuning,
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
        "beta": beta_paper,
        "coupling_strength": tau,
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
        "quad_order": QUAD,
        "ngsolve_version": ngsolve.__version__,
        "scipy_version": scipy.__version__,
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
# Postprocessing utilities
# ======================================================================
def estimated_rates(h, err):
    rates = [np.nan]
    for i in range(1, len(h)):
        r = np.log(err[i] / err[i - 1]) / np.log(h[i] / h[i - 1])
        rates.append(r)
    return np.array(rates)


def read_csv_rows(csv_file):
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found. Run with PLOT_ONLY = False first.")
    with open(csv_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError(f"No data rows found in {csv_file}")
    return rows


def sort_rows_by_h_desc(rows):
    rows_sorted = sorted(rows, key=lambda d: float(d["h"]), reverse=True)
    return rows_sorted


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
        "Nvals": Nvals,
        "h": h,
        "u_L2": u_L2,
        "p_L2": p_L2,
        "w_hdiv": w_hdiv,
        "sigma_hdiv": sigma_hdiv,
        "rate_u": rate_u,
        "rate_p": rate_p,
        "rate_w": rate_w,
        "rate_sigma": rate_sigma,
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


def plot_errors_on_axis(ax, data, k):
    lab = error_labels()
    h = data["h"]
    u_L2 = data["u_L2"]

    ax.loglog(h, u_L2, "o-", label=lab["u"])
    ax.loglog(h, data["p_L2"], "s-", label=lab["p"])
    ax.loglog(h, data["w_hdiv"], "^-", label=lab["w"])
    ax.loglog(h, data["sigma_hdiv"], "d-", label=lab["sigma"])

    ref_order = k + 1
    Cref = 3.5
    ax.loglog(h, Cref * (u_L2[-1] / (h[-1]**ref_order)) * h**ref_order, "--", label=rf"$O(h^{ref_order})$")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel("error")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    ax.invert_xaxis()


def plot_single_convergence(csv_file, k, outfile_pdf, outfile_png=None):
    data = print_convergence_table(csv_file, title=f"Convergence results for k={k}")

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    plot_errors_on_axis(ax, data, k)

    fig.tight_layout()
    fig.savefig(outfile_pdf, format="pdf", bbox_inches="tight")
    if outfile_png is not None:
        fig.savefig(outfile_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_combined_convergence(csv_files, ks, outfile_pdf, outfile_png=None):
    setup_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.ravel()

    subtitles = [
        r"(a) $k=1$",
        r"(b) $k=2$",
        r"(c) $k=3$",
        r"(d) $k=3$, further mesh refinement",
    ]

    for ax, csv_file, k, subtitle in zip(axes, csv_files, ks, subtitles):
        data = print_convergence_table(csv_file, title=None)
        plot_errors_on_axis(ax, data, k)
        ax.set_title(subtitle, y=-0.28)

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

    Nvals = data["Nvals"]
    h = data["h"]
    u_L2 = data["u_L2"]
    p_L2 = data["p_L2"]
    w_hdiv = data["w_hdiv"]
    sigma_hdiv = data["sigma_hdiv"]
    rate_u = data["rate_u"]
    rate_p = data["rate_p"]
    rate_w = data["rate_w"]
    rate_sigma = data["rate_sigma"]

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{c c c c c c c c c c}")
    lines.append(r"\hline")
    lines.append(r"$N$ & $h$ & $\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$ & rate & $\|p-p_h\|_{L^2}$ & rate & $\|\mathbf{w}-\mathbf{w}_h\|_{H(\operatorname{div})}$ & rate & $\|\underline{\boldsymbol{\sigma}}-\underline{\boldsymbol{\sigma}}_h\|_{H(\operatorname{div})}$ & rate \\")
    lines.append(r"\hline")

    for i in range(len(Nvals)):
        lines.append(
            f"{Nvals[i]} & "
            f"{h[i]:.5e} & "
            f"{fmt_err(u_L2[i])} & {fmt_rate(rate_u[i])} & "
            f"{fmt_err(p_L2[i])} & {fmt_rate(rate_p[i])} & "
            f"{fmt_err(w_hdiv[i])} & {fmt_rate(rate_w[i])} & "
            f"{fmt_err(sigma_hdiv[i])} & {fmt_rate(rate_sigma[i])} \\\\"
        )

    lines.append(r"\hline")
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


def run_case_list(mesh_list, k, csv_file, tuning="off", gamma=1.0, scaling2=1.0, save_results=True):
    remove_file_if_exists(csv_file)
    for N in mesh_list:
        print("=" * 100)
        print(f"Running manufactured solution with N={N}, k={k}, tuning={tuning}, gamma={gamma}, scaling2={scaling2}")
        print("=" * 100)
        biot_splitting(
            N=N,
            tuning=tuning,
            gamma=gamma,
            k=k,
            scaling2=scaling2,
            save_results=save_results,
            csv_file=csv_file
        )


def merge_csvs(csv_files, merged_csv):
    all_rows = []
    for csv_file in csv_files:
        rows = read_csv_rows(csv_file)
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise ValueError("No rows found to merge.")

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
    print("Starting manufactured-solution convergence study.")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}")

    if not PLOT_ONLY:
        for k, csv_file in [(1, SUMMARY_CSV_K1), (2, SUMMARY_CSV_K2), (3, SUMMARY_CSV_K3)]:
            run_case_list(
                mesh_list=MESHES_BY_K[k],
                k=k,
                csv_file=csv_file,
                tuning=TUNING,
                gamma=GAMMA,
                scaling2=SCALING2,
                save_results=SAVE_RESULTS
            )

        # k = 3 refined
        run_case_list(
            mesh_list=REFINED_MESHES_K3,
            k=3,
            csv_file=SUMMARY_CSV_K3_REFINED,
            tuning=TUNING,
            gamma=GAMMA,
            scaling2=SCALING2,
            save_results=SAVE_RESULTS
        )

        # merged csv
        merge_csvs(
            [SUMMARY_CSV_K1, SUMMARY_CSV_K2, SUMMARY_CSV_K3, SUMMARY_CSV_K3_REFINED],
            SUMMARY_CSV
        )
    else:
        print("PLOT_ONLY = True: skipping computation, using csv files in results/.")

    # single-panel figures
    for k, csv_file in [(1, SUMMARY_CSV_K1), (2, SUMMARY_CSV_K2), (3, SUMMARY_CSV_K3)]:
        plot_single_convergence(
            csv_file=csv_file,
            k=k,
            outfile_pdf=os.path.join(FIGURES_DIR, f"biot_convergence_k{k}.pdf"),
            outfile_png=os.path.join(FIGURES_DIR, f"biot_convergence_k{k}.png")
        )

    # combined 2x2 figure (Figure 1 of the paper)
    plot_combined_convergence(
        csv_files=[SUMMARY_CSV_K1, SUMMARY_CSV_K2, SUMMARY_CSV_K3, SUMMARY_CSV_K3_REFINED],
        ks=[1, 2, 3, 3],
        outfile_pdf=os.path.join(FIGURES_DIR, "biot_convergence_combined.pdf"),
        outfile_png=os.path.join(FIGURES_DIR, "biot_convergence_combined.png")
    )

    # optional latex tables
    for k, csv_file in [(1, SUMMARY_CSV_K1), (2, SUMMARY_CSV_K2), (3, SUMMARY_CSV_K3)]:
        latex_table_single(
            csv_file=csv_file,
            filename=os.path.join(RESULTS_DIR, f"biot_convergence_table_single_k{k}.tex"),
            caption=rf"Convergence rates for $k={k}$.",
            label=f"tab:biot_convergence_single_k{k}"
        )

    print("Manufactured-solution convergence study completed.")


if __name__ == "__main__":
    main()