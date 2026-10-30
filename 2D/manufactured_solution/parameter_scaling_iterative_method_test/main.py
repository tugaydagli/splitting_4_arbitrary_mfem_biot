#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manufactured-solution parameter study for the iteration tables
(Section 6.1, Tables 1 and 2 of the paper).

  Table 1: varying gamma_1 with fixed gamma_2 = 1
  Table 2: varying gamma_2 with fixed gamma_1 = 1

for k = 1, 2, 3 and the two splitting variants of the paper:
  beta = beta_FS  = 0         (plain fixed-stress split, theta = 0)
  beta = beta_max = M/2       (maximal destabilization of Lemma 3, theta = 1)
with theta := beta / beta_max and M := d alpha^2 / (2 mu + d lambda).

Set TABLES_ONLY = True to skip the computation and regenerate the LaTeX tables
from the csv files already present in results/.

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
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_GAMMA_SCAN_CSV = os.path.join(RESULTS_DIR, "manufactured_gamma_scan_summary.csv")
SUMMARY_SCALING2_SCAN_CSV = os.path.join(RESULTS_DIR, "manufactured_scaling2_scan_summary.csv")

TABLE_GAMMA_SCAN_TEX = os.path.join(RESULTS_DIR, "table_gamma_scan.tex")
TABLE_SCALING2_SCAN_TEX = os.path.join(RESULTS_DIR, "table_scaling2_scan.tex")

GAMMA1_VALUES = [10, 1, 0.1, 0.01, 0.001]                 # Table 1 (gamma_2 = 1)
GAMMA2_VALUES = [1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2]       # Table 2 (gamma_1 = 1)
KS = [1, 2, 3]
N_SCAN = 16                                               # fixed N x N mesh for the iteration counts

# splitting variants: label -> theta = beta / beta_max
VARIANTS = {"FS": 0.0, "max": 1.0}

# table headers for the two variants (adjust if the paper uses macros)
BETA_HEADERS = {"FS": r"$\beta_{\mathrm{FS}}$", "max": r"$\beta_{\max}$"}

COMPUTE_ERRORS = True   # errors are not in Tables 1-2 but support the large-gamma_2 discussion in Sec. 6.1
TABLES_ONLY = False     # True: skip computation, regenerate tables from csv files in results/
QUAD = 10               # quadrature order for all error integrals; exact for squared integrands up to k=3


# ======================================================================
# Core solver
# ======================================================================
def biot_splitting(
    theta=0.0,
    gamma=1,
    k=1,
    scaling2=1,
    N=16,
    beta_label=None,
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

    # Tuning: beta_paper = theta * beta_max >= 0 as in Lemma 3 / eq. (35);
    # the assembled flow block is (c0~ - beta_paper) * M_p
    beta_max = 0.5 * cr
    beta_paper = theta * beta_max
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
    max_iterations = 10000
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

        RESULT1 = B1.solve(RHS1)
        gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
        gfup.vec.data = gfu1.components[1].vec.data

        # step 2
        add = -E.T * gfup.vec
        add = csr_matrix(add).transpose()
        source2.Assemble()
        RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()

        RESULT2 = B2.solve(RHS2)
        gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)
        gfusigma.components[0].vec.data = gfu.components[0].vec.data
        gfusigma.components[1].vec.data = gfu.components[1].vec.data

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

            RESULT1 = B1.solve(RHS1)
            gfu1.vec.FV().NumPy()[:] = RESULT1.reshape(-1)
            gfup.vec.data = gfu1.components[1].vec.data

            # step 2
            add = -E.T * gfup.vec
            add = csr_matrix(add).transpose()
            source2.Assemble()
            RHS2 = vstack([add, ZERO4, csr_matrix(source2.vec.FV()).transpose()]).toarray()

            RESULT2 = B2.solve(RHS2)
            gfu.vec.FV().NumPy()[:] = RESULT2.reshape(-1)
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

        print(f"+=====================================+\n")

    # coupling strength tau = d alpha^2 / (c0 (2 mu + d lam)) = M / c0
    tau = cr / c0 if c0 > 1e-17 else float("inf")

    print("Iterations in each time step:", iterations_in_each_time)
    print("Tolerance:", tol)
    print("gamma_1:", gamma, "| gamma_2:", scaling2)
    print(f"theta = beta/beta_max: {theta} | beta (paper convention, >= 0): {beta_paper}")
    print(f"The coupling strength is {tau}.")
    print(f"The discrete space order is {k}.")
    if hit_cap:
        print("!! WARNING: at least one time step hit the iteration cap; iteration counts are not converged.")

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
        "poisson_ratio": float(lam / (2 * (lam + mu))),
        "alpha": alpha,
        "c0": c0,
        "coupling_strength": tau,
        "iterations_each_time": ",".join(map(str, iterations_in_each_time.astype(int))),
        "iterations_mean": float(np.mean(iterations_in_each_time)),
        "iterations_max": int(np.max(iterations_in_each_time)),
        "hit_cap": hit_cap,
    }

    # ERROR COMPUTATIONS (at final time T = 1)
    error_keys = ["w_L2", "w_div_L2", "w_hdiv", "sigma_L2", "sigma_div_L2", "sigma_hdiv", "u_L2", "p_L2"]

    if COMPUTE_ERRORS:
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

        wex = CoefficientFunction((-Kc * T * (1 - 2 * x) * y * (1 - y), -Kc * T * (1 - 2 * y) * x * (1 - x)))
        divwex = 2 * Kc * T * (x * (1 - x) + y * (1 - y))

        # norms of the exact solution (for relative errors)
        L2norm_uexact = sqrt(Integrate(u_exact**2, mesh, order=QUAD))
        L2norm_pexact = sqrt(Integrate(p_exact**2, mesh, order=QUAD))

        L2norm_sigma_exact = sqrt(Integrate(sigmaex_1**2, mesh, order=QUAD) + Integrate(sigmaex_2**2, mesh, order=QUAD))
        divL2norm_sigma_exact = sqrt(Integrate(div_sigma1**2, mesh, order=QUAD) + Integrate(div_sigma2**2, mesh, order=QUAD))
        Hdivnorm_sigma_exact = np.sqrt(L2norm_sigma_exact**2 + divL2norm_sigma_exact**2)

        L2norm_w_exact = sqrt(Integrate(wex**2, mesh, order=QUAD))
        divL2norm_w_exact = sqrt(Integrate(divwex**2, mesh, order=QUAD))
        Hdivnorm_w_exact = np.sqrt(L2norm_w_exact**2 + divL2norm_w_exact**2)

        # absolute errors
        L2errorstress = sqrt(Integrate((gfucomplete.components[0] - sigmaex_1)**2, mesh, order=QUAD)
                             + Integrate((gfucomplete.components[1] - sigmaex_2)**2, mesh, order=QUAD))
        diverrorstress = sqrt(Integrate((div(gfucomplete.components[0]) - div_sigma1)**2, mesh, order=QUAD)
                              + Integrate((div(gfucomplete.components[1]) - div_sigma2)**2, mesh, order=QUAD))
        Hdiverrorstress = np.sqrt(L2errorstress**2 + diverrorstress**2)

        L2errorw = sqrt(Integrate((gfucomplete.components[3] - wex)**2, mesh, order=QUAD))
        diverrorw = sqrt(Integrate((div(gfucomplete.components[3]) - divwex)**2, mesh, order=QUAD))
        Hdiverrorw = np.sqrt(L2errorw**2 + diverrorw**2)

        u_error = sqrt(Integrate((u_exact - gfucomplete.components[2])**2, mesh, order=QUAD))
        p_error = sqrt(Integrate((p_exact - gfucomplete.components[4])**2, mesh, order=QUAD))

        abs_err = {
            "w_L2": L2errorw, "w_div_L2": diverrorw, "w_hdiv": Hdiverrorw,
            "sigma_L2": L2errorstress, "sigma_div_L2": diverrorstress, "sigma_hdiv": Hdiverrorstress,
            "u_L2": u_error, "p_L2": p_error,
        }
        ref_norm = {
            "w_L2": L2norm_w_exact, "w_div_L2": divL2norm_w_exact, "w_hdiv": Hdivnorm_w_exact,
            "sigma_L2": L2norm_sigma_exact, "sigma_div_L2": divL2norm_sigma_exact, "sigma_hdiv": Hdivnorm_sigma_exact,
            "u_L2": L2norm_uexact, "p_L2": L2norm_pexact,
        }

        for key in error_keys:
            results[f"{key}_abs"] = float(abs_err[key])
        for key in error_keys:
            results[f"{key}_rel"] = float(abs_err[key] / ref_norm[key])

        print("Relative errors:", {key: results[f"{key}_rel"] for key in error_keys})
    else:
        for key in error_keys:
            results[f"{key}_abs"] = float("nan")
        for key in error_keys:
            results[f"{key}_rel"] = float("nan")

    results["quad_order"] = QUAD
    results["ngsolve_version"] = ngsolve.__version__
    results["scipy_version"] = scipy.__version__

    return results


# ======================================================================
# Utilities for automation / tables
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


def sci_notation_latex(x, digits=3):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return rf"{mantissa:.{digits}f} \times 10^{{{exponent}}}"


def format_gamma_value(g):
    if abs(g - round(g)) < 1e-14 and g >= 1:
        return f"{int(round(g))}"
    return f"{g:g}"


def format_poisson_ratio(v):
    s = f"{v:.5f}"
    s = s.rstrip("0").rstrip(".")
    return s


def load_csv_rows(csv_file):
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found. Run with TABLES_ONLY = False first.")
    with open(csv_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_hit_cap(rows, name):
    capped = [r for r in rows if str(r["hit_cap"]) == "True"]
    if capped:
        print(f"!! WARNING ({name}): {len(capped)} run(s) hit the iteration cap; their means are not converged counts:")
        for r in capped:
            print(f"   gamma_1={r['gamma']}, gamma_2={r['scaling2']}, k={r['k']}, beta={r['beta_label']}")


# ======================================================================
# Parameter-study drivers
# ======================================================================
def run_scan(gamma_values, scaling2_values, out_csv, name):
    rows = []
    for gamma in gamma_values:
        for scaling2 in scaling2_values:
            for k in KS:
                for label, theta in VARIANTS.items():
                    print("=" * 100)
                    print(f"Running {name}: gamma_1={gamma}, gamma_2={scaling2}, k={k}, beta=beta_{label} (theta={theta})")
                    print("=" * 100)
                    result = biot_splitting(
                        theta=theta,
                        gamma=gamma,
                        k=k,
                        scaling2=scaling2,
                        N=N_SCAN,
                        beta_label=label,
                    )
                    rows.append(result)

    write_rows_to_csv(rows, out_csv)
    print(f"{name} summary saved to {out_csv}")
    check_hit_cap(rows, name)


def run_gamma_scan():
    run_scan(GAMMA1_VALUES, [1], SUMMARY_GAMMA_SCAN_CSV, "gamma_1 scan")


def run_scaling2_scan():
    run_scan([1], GAMMA2_VALUES, SUMMARY_SCALING2_SCAN_CSV, "gamma_2 scan")


# ======================================================================
# Table generation (booktabs; requires \usepackage{booktabs})
# ======================================================================
def find_row(rows, key, value, k, label):
    for row in rows:
        if abs(float(row[key]) - value) < 1e-14 and int(row["k"]) == k and row["beta_label"] == label:
            return row
    raise ValueError(f"Missing row for {key}={value}, k={k}, beta_{label}")


def iteration_cells(rows, key, value):
    cells = []
    for k in KS:
        for label in VARIANTS:
            row = find_row(rows, key, value, k, label)
            cells.append(f"{float(row['iterations_mean']):.2f}")
    return " & ".join(cells)


def k_header_lines(first_cols):
    """Two header rows: k groups with cmidrules, then beta_FS / beta_max per group."""
    nvar = len(VARIANTS)
    groups = " & ".join(rf"\multicolumn{{{nvar}}}{{c}}{{$k={k}$}}" for k in KS)
    cmid = ""
    col = first_cols + 1
    for _ in KS:
        cmid += rf"\cmidrule(lr){{{col}-{col + nvar - 1}}}"
        col += nvar
    sub = " & ".join(BETA_HEADERS[label] for _ in KS for label in VARIANTS)
    return groups, cmid, sub


def make_gamma_scan_table(summary_csv, out_tex):
    rows = load_csv_rows(summary_csv)
    check_hit_cap(rows, "gamma_1 scan")

    first_cols = 2
    groups, cmid, sub = k_header_lines(first_cols)
    ncols = first_cols + len(KS) * len(VARIANTS)

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{" + "c" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma_1$ & coupling strength $\tau$ & " + groups + r" \\")
    lines.append(cmid)
    lines.append(r"& & " + sub + r" \\")
    lines.append(r"\midrule")

    for gamma in GAMMA1_VALUES:
        ref = find_row(rows, "gamma", gamma, KS[0], "FS")
        coupling = float(ref["coupling_strength"])
        lines.append(
            f"{format_gamma_value(gamma)} & "
            f"${sci_notation_latex(coupling)}$ & "
            f"{iteration_cells(rows, 'gamma', gamma)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Average number of iterations per time step for varying $\gamma_1$ with fixed $\gamma_2 = 1$.}")
    lines.append(r"\label{tab:manufactured_gamma_scan}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table written to {out_tex}")


def make_scaling2_scan_table(summary_csv, out_tex):
    rows = load_csv_rows(summary_csv)
    check_hit_cap(rows, "gamma_2 scan")

    first_cols = 3
    groups, cmid, sub = k_header_lines(first_cols)
    ncols = first_cols + len(KS) * len(VARIANTS)

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{" + "c" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma_2$ & Poisson ratio & coupling strength $\tau$ & " + groups + r" \\")
    lines.append(cmid)
    lines.append(r"& & & " + sub + r" \\")
    lines.append(r"\midrule")

    for scaling2 in GAMMA2_VALUES:
        ref = find_row(rows, "scaling2", scaling2, KS[0], "FS")
        poisson_ratio = float(ref["poisson_ratio"])
        coupling = float(ref["coupling_strength"])
        lines.append(
            f"{format_gamma_value(scaling2)} & "
            f"{format_poisson_ratio(poisson_ratio)} & "
            f"${sci_notation_latex(coupling)}$ & "
            f"{iteration_cells(rows, 'scaling2', scaling2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Average number of iterations per time step for varying $\gamma_2$ with fixed $\gamma_1 = 1$.}")
    lines.append(r"\label{tab:manufactured_scaling2_scan}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table written to {out_tex}")


# ======================================================================
# Main
# ======================================================================
def main():
    print("Starting manufactured-solution parameter-study runs.")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}")

    if not TABLES_ONLY:
        run_gamma_scan()
        run_scaling2_scan()
    else:
        print("TABLES_ONLY = True: skipping computation, using csv files in results/.")

    make_gamma_scan_table(SUMMARY_GAMMA_SCAN_CSV, TABLE_GAMMA_SCAN_TEX)
    make_scaling2_scan_table(SUMMARY_SCALING2_SCAN_CSV, TABLE_SCALING2_SCAN_TEX)

    print("All runs and tables completed.")


if __name__ == "__main__":
    main()