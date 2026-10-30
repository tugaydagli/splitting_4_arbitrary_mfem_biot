#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mandel (quarter domain) with row-wise H(div) stress, VectorL2 displacement,
paper BCs (37)-(40), u2 = U2(b,t) on y=b, and structured anisotropic mesh.

Analytical solutions from:
    Abousleiman, Y., Cheng, A.H.-D., Cui, L., Detournay, E., Roegiers, J.-C.
    "Mandel's problem revisited." Géotechnique 46(2), 187-195 (1996).

Numerical setup (parameters, BCs, splitting) following:
    Mikelic, A., Wang, B., Wheeler, M.F.
    "Numerical convergence study of iterative coupling for coupled flow
    and geomechanics." Comput. Geosci. 18, 325-341 (2014).
    
Author: Tugay Dagli
"""

from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import os
import pickle
import csv

# =============================================================================
# Paths and table generate
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ITER_TABLE_TEX = os.path.join(RESULTS_DIR, "mandel_iterations_summary.tex")
ITER_TABLE_CSV = os.path.join(RESULTS_DIR, "mandel_iterations_summary.csv")


def tuning_to_latex_name(tuning):
    if tuning == "off":
        return r"$\theta = 0$"
    elif tuning == "on":
        return r"$\theta = 1$"
    elif tuning == "on1d":
        return r"$\theta = \tfrac{3}{4}$"
    else:
        return tuning


def write_iteration_summary(out_dict, csv_path=ITER_TABLE_CSV, tex_path=ITER_TABLE_TEX):
    """
    Write a small summary of iteration counts for the three tuning choices.
    out_dict should be like:
        {
            "off": out_off,
            "on": out_on,
            "on1d": out_on1d,
        }
    """
    rows = []
    for tuning_key, out in out_dict.items():
        its = np.asarray(out["iterations"], dtype=float)
        beta = 0.0 - float(out["beta"])  # paper sign (beta >= 0); avoids -0.0 for theta = 0
        rows.append({
            "tuning": tuning_key,
            "beta": beta,
            "iterations_mean": float(np.mean(its)),
            "iterations_max": int(np.max(its)),
            "num_steps": int(len(its)),
        })

    # csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tuning", "beta", "iterations_mean", "iterations_max", "num_steps"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # tex
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"$\theta = \beta/\beta_{\max}$ & $\beta$ & avg. iterations & max iterations \\")
    lines.append(r"\hline")

    for row in rows:
        lines.append(
            f"{tuning_to_latex_name(row['tuning'])} & "
            f"${row['beta']:.6e}$ & "
            f"{row['iterations_mean']:.4f} & "
            f"{row['iterations_max']} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Average and maximum number of fixed-stress iterations per time step "
        r"for Mandel's problem with $\theta \in \{0, \tfrac{3}{4}, 1\}$.}"
    )
    lines.append(r"\label{tab:mandel_iteration_summary}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Iteration summary written to: {csv_path}")
    print(f"Iteration summary written to: {tex_path}")
# =============================================================================
# Matplotlib / paper style
# =============================================================================

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
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

LINE_W = 1.0
MS_O   = 4
MS_X   = 4
MEW    = 0.9


# =============================================================================
# I/O helpers
# =============================================================================

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def savefig_pub(name, outdir=FIGURES_DIR, dpi=600):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.gcf()
    fig.set_size_inches(5.2, 3.8)

    fig.savefig(os.path.join(outdir, f"{name}.pdf"),
                bbox_inches="tight")

    fig.savefig(os.path.join(outdir, f"{name}.png"),
                dpi=dpi, bbox_inches="tight")


# =============================================================================
# SciPy / NGSolve utilities
# =============================================================================

def to_csr(mat_ngsolve):
    rows, cols, vals = mat_ngsolve.COO()
    return sp.csr_matrix((vals, (rows, cols)))


def to_np(vec_ngsolve):
    return vec_ngsolve.FV().NumPy().copy()


def bitarray_to_bool(ba):
    n = len(ba)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        out[i] = bool(ba[i])
    return out


def apply_row_dirichlet(A_csc, free_dofs_bool):
    """Enforce x_i = 0 for fixed dofs by overwriting rows with e_i^T."""
    A = A_csc.tolil()
    fixed = np.where(~free_dofs_bool)[0]
    for i in fixed:
        A.rows[i] = [i]
        A.data[i] = [1.0]
    return A.tocsc()


def apply_rhs_dirichlet(b, free_dofs_bool):
    bb = b.copy()
    bb[~free_dofs_bool] = 0.0
    return bb


def hdiv_normal_trace(hdiv_proxy_vec, nrm_vec):
    """Normal trace of an H(div) proxy on boundary (version-robust)."""
    if hasattr(hdiv_proxy_vec, "Trace"):
        return (hdiv_proxy_vec.Trace() * nrm_vec)
    return Trace(hdiv_proxy_vec * nrm_vec)


def eval_gf_2d(gf, mesh, x, y):
    """Point-evaluation helper."""
    try:
        return gf(mesh(x, y))
    except Exception:
        return gf(x, y)


def eval_sigma_components(gfsigma, mesh, x, y):
    """
    Evaluate total stress components at a point.
    gfsigma on Sigma = [row1,row2], row1=(s11,s12), row2=(s21,s22).
    """
    pt = mesh(x, y)
    r1 = gfsigma.components[0](pt)
    r2 = gfsigma.components[1](pt)
    return float(r1[0]), float(r1[1]), float(r2[0]), float(r2[1])


# =============================================================================
# Structured anisotropic mesh builder
# =============================================================================

def make_graded_x_nodes(a, hx_coarse, n_fine, r):
    """
    Build a non-uniform x-node array on [0, a] that is uniform with spacing
    hx_coarse in the interior and geometrically refined near x = a.
    """
    L_fine_approx = hx_coarse * n_fine * 0.5
    n_coarse = max(1, int(round((a - L_fine_approx) / hx_coarse)))
    x_split  = n_coarse * hx_coarse
    L_fine   = a - x_split

    H       = L_fine * (1.0 - r) / (1.0 - r**n_fine)
    widths  = np.array([H * r**i for i in range(n_fine)])
    hx_fine = widths[-1]

    x_coarse = np.linspace(0.0, x_split, n_coarse + 1)
    x_fine   = x_split + np.concatenate([[0.0], np.cumsum(widths)])
    x_nodes  = np.concatenate([x_coarse, x_fine[1:]])
    x_nodes[-1] = a
    return x_nodes, n_coarse, hx_fine


def make_structured_rect_mesh(a, b, nx, ny, x_nodes=None):
    """
    Structured rectangular mesh on [0,a]x[0,b] with triangular elements.
    BCs labelled: bottom, right, top, left.
    """
    from netgen.meshing import (Mesh as NetMesh, MeshPoint,
                                Element2D, Element1D, FaceDescriptor)
    from netgen.csg import Pnt

    if x_nodes is None:
        xs = np.linspace(0.0, a, nx + 1)
    else:
        xs = np.asarray(x_nodes, dtype=float)
        nx = len(xs) - 1

    ys = np.linspace(0.0, b, ny + 1)

    m = NetMesh(dim=2)
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))

    pids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            pids[(i, j)] = m.Add(MeshPoint(Pnt(xs[i], ys[j], 0)))

    for j in range(ny):
        for i in range(nx):
            p00 = pids[(i,   j  )]
            p10 = pids[(i+1, j  )]
            p11 = pids[(i+1, j+1)]
            p01 = pids[(i,   j+1)]
            m.Add(Element2D(1, [p00, p10, p11]))
            m.Add(Element2D(1, [p00, p11, p01]))

    m.SetBCName(0, "bottom")
    for i in range(nx):
        m.Add(Element1D([pids[(i, 0)], pids[(i+1, 0)]], index=1))

    m.SetBCName(1, "right")
    for j in range(ny):
        m.Add(Element1D([pids[(nx, j)], pids[(nx, j+1)]], index=2))

    m.SetBCName(2, "top")
    for i in range(nx - 1, -1, -1):
        m.Add(Element1D([pids[(i+1, ny)], pids[(i, ny)]], index=3))

    m.SetBCName(3, "left")
    for j in range(ny - 1, -1, -1):
        m.Add(Element1D([pids[(0, j+1)], pids[(0, j)]], index=4))

    m.Compress()
    return Mesh(m)


# =============================================================================
# Mandel analytic helpers
# =============================================================================

def mandel_roots(n_terms, nu, nu_u):
    m = (1.0 - nu) / (nu_u - nu)
    eps = 1e-12
    alphas = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        aL = (n - 1) * np.pi + eps
        aR = (n - 0.5) * np.pi - eps
        f = lambda a: np.tan(a) - m * a
        alphas[n - 1] = brentq(f, aL, aR, maxiter=500)
    return alphas


def mandel_coeffs(alphas):
    sa = np.sin(alphas)
    ca = np.cos(alphas)
    denom = alphas - sa * ca
    Cp = sa / denom
    Cu = (sa * ca) / denom
    return Cp, Cu, ca


def mandel_A1_A2(B, nu, nu_u, alpha=1.0):
    A1 = 3.0 / (B * (1.0 + nu_u))
    A2 = alpha * (1.0 - 2.0 * nu) / (1.0 - nu)
    return A1, A2


def p_analytic(x, t, a, F, B, nu_u, c, alphas, Cp, ca):
    x = np.asarray(x, dtype=float)
    p0 = (B * (1.0 + nu_u) * F) / (3.0 * a)
    cos_ax = np.cos(np.outer(alphas, x / a))
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    s = (Cp[:, None] * (cos_ax - ca[:, None]) * expo[:, None]).sum(axis=0)
    return 2.0 * p0 * s


def p_initial_exact(a, F, B, nu_u, x):
    x = np.asarray(x, dtype=float)
    p0 = (B * (1.0 + nu_u) * F) / (3.0 * a)
    return p0 * np.ones_like(x)


def U2_top_series(t, a, b, F, nu, nu_u, G, c, alphas, Cu):
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    bracket = (-(F * (1.0 - nu)) / (2.0 * G * a)
               + (F * (1.0 - nu_u)) / (G * a) * np.sum(Cu * expo))
    return bracket * b


def u1_analytic_series(x, t, a, F, G, nu, nu_u, c, alphas):
    x = np.asarray(x, dtype=float)
    sa = np.sin(alphas)
    ca = np.cos(alphas)
    denom = alphas - sa * ca
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    sum1 = np.sum((sa * ca / denom) * expo)
    sum2 = np.sum((ca / denom)[:, None]
                  * np.sin(np.outer(alphas, x / a))
                  * expo[:, None], axis=0)
    return (F * nu / (2 * G * a) - F * nu_u / (G * a) * sum1) * x + (F / G) * sum2


def u2_analytic_linear_in_y(y, U2_top, b):
    y = np.asarray(y, dtype=float)
    return U2_top * (y / b)


def u1_initial_exact(x, a, F, G, nu_u):
    x = np.asarray(x, dtype=float)
    return (F * nu_u / (2.0 * G * a)) * x


def u2_initial_exact(y, a, F, G, nu_u):
    y = np.asarray(y, dtype=float)
    return -(F * (1.0 - nu_u) / (2.0 * G * a)) * y


def sigma22_analytic(x, t, a, b, F, B, nu, nu_u, c,
                     alphas, Cp, Cu, ca, lam, mu, alpha=1.0):
    x = np.asarray(x, dtype=float)
    A1, A2 = mandel_A1_A2(B, nu, nu_u, alpha)

    expo   = np.exp(-(alphas**2) * c * t / (a**2))
    cos_ax = np.cos(np.outer(alphas, x / a))

    sum_cos = np.sum(Cp[:, None] * cos_ax * expo[:, None], axis=0)
    sum_cu  = np.sum(Cu * expo)

    return -F / a \
           - (2.0 * F * A2) / (a * A1) * sum_cos \
           + (2.0 * F / a) * sum_cu


def sigma11_analytic(x, t, a, b, F, B, nu, nu_u, c,
                     alphas, Cp, Cu, ca, lam, mu, alpha=1.0):
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def sigma22_initial_exact(x, a, F, B, nu_u, lam, mu, alpha=1.0):
    x = np.asarray(x, dtype=float)
    return np.full_like(x, -F / a)


def sigma11_initial_exact(x, a, F, B, nu_u, lam, mu, alpha=1.0):
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def qx_analytic(x, t, a, F, B, nu_u, c, k_perm, eta, alphas, Cp):
    x  = np.asarray(x, dtype=float)
    A1 = 3.0 / (B * (1.0 + nu_u))
    expo   = np.exp(-(alphas**2) * c * t / (a**2))
    sin_ax = np.sin(np.outer(alphas, x / a))
    return ((2.0 * F * k_perm) / (a**2 * eta * A1)
            * np.sum(alphas[:, None] * Cp[:, None] * sin_ax * expo[:, None],
                     axis=0))


def qx_initial_exact(x, a, F, B, nu_u, k_perm, eta, alphas, Cp):
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def qx_cell_avg_analytic(x_edges, t, a, F, B, nu_u, c, k_perm, eta, alphas, Cp):
    x_edges = np.asarray(x_edges, dtype=float)
    A1   = 3.0 / (B * (1.0 + nu_u))
    expo = np.exp(-(alphas**2) * c * t / (a**2))

    xL = x_edges[:-1]
    xR = x_edges[1:]
    dx = xR - xL

    cos_xL = np.cos(np.outer(alphas, xL / a))
    cos_xR = np.cos(np.outer(alphas, xR / a))
    integral_qx = ((2.0 * F * k_perm) / (a * eta * A1)
                   * np.sum(Cp[:, None] * (cos_xL - cos_xR) * expo[:, None],
                            axis=0))

    x_mid  = 0.5 * (xL + xR)
    qx_avg = integral_qx / dx
    return x_mid, qx_avg


def qx_cell_avg_initial(x_edges):
    x_edges = np.asarray(x_edges, dtype=float)
    n_cells = len(x_edges) - 1
    x_mid   = 0.5 * (x_edges[:-1] + x_edges[1:])
    return x_mid, np.zeros(n_cells)


def qx_cell_avg_numerical(gfw_x, mesh, x_edges, y):
    gl_pts = np.array([-0.8611363115940526, -0.3399810435848563,
                        0.3399810435848563,  0.8611363115940526])
    gl_wts = np.array([ 0.3478548451374538,  0.6521451548625461,
                        0.6521451548625461,  0.3478548451374538])

    xL    = x_edges[:-1]
    xR    = x_edges[1:]
    x_mid = 0.5 * (xL + xR)
    qx_avg = np.zeros(len(x_mid))

    for i, (xl, xr) in enumerate(zip(xL, xR)):
        xq = 0.5*(xl + xr) + 0.5*(xr - xl)*gl_pts
        fq = np.array([float(eval_gf_2d(gfw_x, mesh, xxi, y)) for xxi in xq])
        qx_avg[i] = 0.5 * np.dot(gl_wts, fq)

    return x_mid, qx_avg


# =============================================================================
# Mandel solver
# =============================================================================

def mandel_splitting_U2BC(
    tuning="off",
    k=1,
    a=100.0,
    b=10.0,
    hx=2.5,
    hy=0.25,
    nx=None,
    ny=None,
    dt=10.0,
    Tfinal=50000.0,
    tol=1e-6,
    max_iterations=300,
    F=1.0,
    B=0.83333,
    nu_u=0.44,
    c_diff=0.465,
    n_terms=60,
    record_times=(10, 50, 1000, 5000, 10000, 20000, 50000),
    nx_plot=121,
    y_plot_factor=0.5,
    verbose=True,
    graded=False,
    graded_n_fine=12,
    graded_r=0.75,
):
    if graded:
        x_nodes, n_coarse, hx_fine = make_graded_x_nodes(
            a, hx_coarse=hx, n_fine=graded_n_fine, r=graded_r)
        nx = len(x_nodes) - 1
        if verbose:
            print(f"Graded mesh: {n_coarse} coarse cells (hx={hx}) + "
                  f"{graded_n_fine} fine cells (hx_fine={hx_fine:.4f} m), "
                  f"nx={nx} total")
    else:
        if nx is None:
            nx = int(round(a / hx))
        x_nodes = None
        if verbose:
            print(f"Uniform mesh: nx={nx}  (hx={a/nx:.4g})")

    if ny is None:
        ny = int(round(b / hy))

    if verbose:
        print(f"Mesh: nx={nx}, ny={ny}  (hy={b/ny:.4g})")

    mesh = make_structured_rect_mesh(a, b, nx, ny, x_nodes=x_nodes)

    if x_nodes is not None:
        x_edges = np.asarray(x_nodes, dtype=float)
    else:
        x_edges = np.linspace(0.0, a, nx + 1)

    if verbose:
        print("Boundaries:", mesh.GetBoundaries())

    Eyoung = 5.94e9
    nu     = 0.2
    alpha  = 1.0
    M      = 1.65e10
    c0     = 1.0 / M

    k_perm    = 0.1 * 9.869233e-13
    eta       = 1.0e-3
    K_over_eta = k_perm / eta
    eta_over_k = 1.0 / K_over_eta

    lam = Eyoung * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = Eyoung / (2 * (1 + nu))

    d  = 2
    cr = (d * alpha**2) / (2 * mu + d * lam)
    C1 = cr / (d * alpha)
    C2 = c0 + cr

    if tuning == "off":
        beta = 0.0
    elif tuning == "on":
        beta = -0.5 * cr
    elif tuning == "on1d":
        beta = -(alpha**2 * mu * (2 * mu + lam)**(-1) * (mu + lam)**(-1))
    else:
        raise ValueError("tuning must be 'off', 'on', or 'on1d'")

    if verbose:
        print(f"beta = {beta:.6e}")
        print(f"tau = M/c0 = {cr / c0:g}")

    alphas    = mandel_roots(n_terms, nu=nu, nu_u=nu_u)
    Cp, Cu, ca = mandel_coeffs(alphas)

    W   = HDiv(mesh, order=k, RT=True, dirichlet="left|bottom|top")
    Qp  = L2(mesh, order=k)
    WQ  = FESpace([W, Qp])
    gfwq = GridFunction(WQ)
    gfw, gfp = gfwq.components
    gfp_old  = GridFunction(Qp)

    Sigma_row1 = HDiv(mesh, order=k, RT=True, dirichlet="bottom|top|right")
    Sigma_row2 = HDiv(mesh, order=k, RT=True, dirichlet="left|right")
    Sigma      = FESpace([Sigma_row1, Sigma_row2])
    Omega      = H1(mesh, order=k)
    U          = VectorL2(mesh, order=k)

    Mech    = FESpace([Sigma, Omega, U])
    gfmech  = GridFunction(Mech)
    gfsigma, gfomega, gfu = gfmech.components
    gfsigma_old = GridFunction(Sigma)

    w, v       = W.TnT()
    p, q       = Qp.TnT()
    (sigma1, sigma2) = Sigma.TrialFunction()
    (tau1,   tau2  ) = Sigma.TestFunction()
    ome, gam   = Omega.TnT()
    u, z       = U.TnT()
    nrm        = specialcf.normal(mesh.dim)

    def tr_rows(s1, s2):
        return s1[0] + s2[1]

    A_sigma1 = (1/(2*mu)) * (
        sigma1
        - (lam/(2*mu + d*lam)) * tr_rows(sigma1, sigma2)
          * CoefficientFunction((1, 0))
    )
    A_sigma2 = (1/(2*mu)) * (
        sigma2
        - (lam/(2*mu + d*lam)) * tr_rows(sigma1, sigma2)
          * CoefficientFunction((0, 1))
    )

    aW = BilinearForm(W)
    aW += (w * v) * dx
    aW.Assemble()
    A = eta_over_k * to_csr(aW.mat)

    bWp = BilinearForm(trialspace=Qp, testspace=W)
    bWp += -(p * div(v)) * dx
    bWp.Assemble()
    Bmat = to_csr(bWp.mat)

    mP = BilinearForm(Qp)
    mP += (p * q) * dx
    mP.Assemble()
    Mp = to_csr(mP.mat)

    Cmat = C2 * Mp
    TP   = beta * Mp

    eSigP = BilinearForm(trialspace=Sigma, testspace=Qp)
    eSigP += ((sigma1[0] + sigma2[1]) * q) * dx
    eSigP.Assemble()
    Etr = C1 * to_csr(eSigP.mat)

    ndW = W.ndof
    ndQ = Qp.ndof

    LHS1 = vstack([
        hstack([A,             Bmat  ]),
        hstack([-dt * Bmat.T,  Cmat + TP])
    ]).tocsc()

    freeW  = bitarray_to_bool(W.FreeDofs())
    free1  = np.ones(ndW + ndQ, dtype=bool)
    free1[:ndW] = freeW

    LHS1_bc = apply_row_dirichlet(LHS1, free1)
    B1 = sp.linalg.splu(LHS1_bc)

    Pbf = BilinearForm(Sigma)
    Pbf += (A_sigma1 * tau1 + A_sigma2 * tau2) * dx
    Pbf.Assemble()
    Pmat = to_csr(Pbf.mat)

    r = BilinearForm(trialspace=U, testspace=Sigma)
    r += (div(tau1) * u[0] + div(tau2) * u[1]) * dx
    r.Assemble()
    R = to_csr(r.mat)

    s = BilinearForm(trialspace=Sigma, testspace=Omega)
    s += ((sigma2[0] - sigma1[1]) * gam) * dx
    s.Assemble()
    S = to_csr(s.mat)

    ndS  = Sigma.ndof
    ndO  = Omega.ndof
    ndU  = U.ndof
    ndSO = ndS + ndO

    ZERO_OO = sp.csr_matrix((ndO, ndO))
    ZERO_UU = sp.csr_matrix((ndU, ndU))

    PTILDE = vstack([
        hstack([Pmat,  S.T    ]),
        hstack([-S,    ZERO_OO])
    ]).tocsr()

    RTILDE = vstack([
        R,
        sp.csr_matrix((ndO, ndU))
    ]).tocsr()

    LHS2 = vstack([
        hstack([PTILDE,    RTILDE   ]),
        hstack([-RTILDE.T, ZERO_UU  ])
    ]).tocsc()

    freeSigma = bitarray_to_bool(Sigma.FreeDofs())
    free2     = np.ones(ndSO + ndU, dtype=bool)
    free2[:ndS] = freeSigma

    LHS2_bc = apply_row_dirichlet(LHS2, free2)
    B2 = sp.linalg.splu(LHS2_bc)

    U2par    = Parameter(0.0)
    lf_u2top = LinearForm(Sigma)
    lf_u2top += U2par * hdiv_normal_trace(tau2, nrm) * ds("top")

    p0_val = (B * (1.0 + nu_u) * F) / (3.0 * a)
    G      = mu

    U2_0 = U2_top_series(
        0.0, a=a, b=b, F=F, nu=nu, nu_u=nu_u, G=mu,
        c=c_diff, alphas=alphas, Cu=Cu
    )

    u1_0_cf = (F * nu_u / (2.0 * G * a)) * x
    u2_0_cf = -(F * (1.0 - nu_u) / (2.0 * G * a)) * y

    sigma11_0 = 0.0
    sigma22_0 = -F / a

    gfp_old.Set(CoefficientFunction(p0_val))
    gfp.Set(CoefficientFunction(p0_val))

    gfu.Set(CoefficientFunction((u1_0_cf, u2_0_cf)))

    gfsigma.components[0].Set(CoefficientFunction((sigma11_0, 0.0)))
    gfsigma.components[1].Set(CoefficientFunction((0.0, sigma22_0)))

    gfsigma_old.vec.data = gfsigma.vec.data

    if verbose:
        print("\nInitialized exact benchmark state at t=0")
        print(f"  p0 = {p0_val:.8e}")
        print(f"  U2(top,0) = {U2_0:.8e}")

    nsteps = int(round(Tfinal / dt))
    iterations_each_step = []
    U2_used = []

    record_times     = list(record_times)
    record_times_set = set(float(ti) for ti in record_times)

    x_plot      = np.linspace(0.0, a, nx_plot)
    y_plot      = y_plot_factor * b
    y_plot_vec  = np.linspace(0.0, b, nx_plot)
    x_mid       = 0.5 * a

    snapshots_p   = {}
    snapshots_u1  = {}
    snapshots_u2  = {}
    snapshots_s11 = {}
    snapshots_s22 = {}
    snapshots_s12 = {}
    snapshots_s21 = {}
    snapshots_qx     = {}
    snapshots_qx_avg = {}

    if 0.0 in record_times_set:
        p_num = np.array([float(eval_gf_2d(gfp, mesh, xx, y_plot))
                          for xx in x_plot])
        p_ex  = p_initial_exact(a=a, F=F, B=B, nu_u=nu_u, x=x_plot)
        snapshots_p[0.0] = (x_plot.copy(), p_num, p_ex)

        u1_num = np.array([float(eval_gf_2d(gfu[0], mesh, xx, y_plot))
                           for xx in x_plot])
        u1_ex  = u1_initial_exact(x_plot, a=a, F=F, G=mu, nu_u=nu_u)
        snapshots_u1[0.0] = (x_plot.copy(), u1_num, u1_ex)

        u2_num = np.array([float(eval_gf_2d(gfu[1], mesh, x_mid, yy))
                           for yy in y_plot_vec])
        u2_ex  = u2_initial_exact(y_plot_vec, a=a, F=F, G=mu, nu_u=nu_u)
        snapshots_u2[0.0] = (y_plot_vec.copy(), u2_num, u2_ex)

        s11_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[0]
                            for xx in x_plot])
        s12_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[1]
                            for xx in x_plot])
        s21_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[2]
                            for xx in x_plot])
        s22_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[3]
                            for xx in x_plot])

        s11_ex = sigma11_initial_exact(x_plot, a=a, F=F, B=B, nu_u=nu_u,
                                       lam=lam, mu=mu, alpha=alpha)
        s22_ex = sigma22_initial_exact(x_plot, a=a, F=F, B=B, nu_u=nu_u,
                                       lam=lam, mu=mu, alpha=alpha)

        snapshots_s11[0.0] = (x_plot.copy(), s11_num, s11_ex)
        snapshots_s22[0.0] = (x_plot.copy(), s22_num, s22_ex)
        snapshots_s12[0.0] = (x_plot.copy(), s12_num, np.zeros_like(x_plot))
        snapshots_s21[0.0] = (x_plot.copy(), s21_num, np.zeros_like(x_plot))

        qx_num_0 = np.array([float(eval_gf_2d(gfw[0], mesh, xx, y_plot))
                              for xx in x_plot])
        qx_ex_0  = qx_initial_exact(x_plot, a=a, F=F, B=B, nu_u=nu_u,
                                    k_perm=k_perm, eta=eta,
                                    alphas=alphas, Cp=Cp)
        snapshots_qx[0.0] = (x_plot.copy(), qx_num_0, qx_ex_0)

        xm0, qx_cavg_num_0 = qx_cell_avg_numerical(gfw[0], mesh, x_edges, y_plot)
        xm0_ex, qx_cavg_ex_0 = qx_cell_avg_initial(x_edges)
        snapshots_qx_avg[0.0] = (xm0, qx_cavg_num_0, qx_cavg_ex_0)

    for nstep in range(1, nsteps + 1):
        t_now = nstep * dt
        if verbose:
            print(f"\n+====================+\n"
                  f"Time step {nstep}/{nsteps}  (t={t_now:g})")

        U2_value = U2_top_series(
            t_now, a=a, b=b, F=F, nu=nu, nu_u=nu_u, G=mu,
            c=c_diff, alphas=alphas, Cu=Cu
        )
        U2_used.append(U2_value)

        U2par.Set(U2_value)
        lf_u2top.Assemble()
        rhs_u2top = to_np(lf_u2top.vec)

        sigma_old_np = to_np(gfsigma_old.vec)
        p_old_np     = to_np(gfp_old.vec)

        sigma_prev = sigma_old_np.copy()
        p_prev     = p_old_np.copy()

        rhs1 = np.zeros(ndW + ndQ)
        rhs1[ndW:] = (Etr.dot(sigma_old_np) - Etr.dot(sigma_prev)
                      + Cmat.dot(p_old_np) + TP.dot(p_prev))
        rhs1 = apply_rhs_dirichlet(rhs1, free1)
        sol1 = B1.solve(rhs1)
        gfwq.vec.FV().NumPy()[:] = sol1
        p_np = to_np(gfp.vec)

        rhs2 = np.zeros(ndSO + ndU)
        rhs2[:ndS] = -(Etr.T.dot(p_np)) + rhs_u2top
        rhs2 = apply_rhs_dirichlet(rhs2, free2)
        sol2 = B2.solve(rhs2)
        gfmech.vec.FV().NumPy()[:] = sol2
        gfsigma.vec.data = gfmech.components[0].vec.data

        prev_sol1 = sol1.copy()
        prev_sol2 = sol2.copy()
        err1 = np.inf
        err2 = np.inf
        it   = 0

        sigma_prev = to_np(gfsigma.vec)
        p_prev     = p_np.copy()

        while (err1 > tol or err2 > tol) and it < max_iterations:
            it += 1

            rhs1 = np.zeros(ndW + ndQ)
            rhs1[ndW:] = (Etr.dot(sigma_old_np) - Etr.dot(sigma_prev)
                          + Cmat.dot(p_old_np) + TP.dot(p_prev))
            rhs1 = apply_rhs_dirichlet(rhs1, free1)
            sol1 = B1.solve(rhs1)
            gfwq.vec.FV().NumPy()[:] = sol1
            p_np = to_np(gfp.vec)

            rhs2 = np.zeros(ndSO + ndU)
            rhs2[:ndS] = -(Etr.T.dot(p_np)) + rhs_u2top
            rhs2 = apply_rhs_dirichlet(rhs2, free2)
            sol2 = B2.solve(rhs2)
            gfmech.vec.FV().NumPy()[:] = sol2
            gfsigma.vec.data = gfmech.components[0].vec.data

            err1 = np.linalg.norm(sol1 - prev_sol1) / (np.linalg.norm(prev_sol1) + 1e-14)
            err2 = np.linalg.norm(sol2 - prev_sol2) / (np.linalg.norm(prev_sol2) + 1e-14)
            prev_sol1 = sol1.copy()
            prev_sol2 = sol2.copy()

            sigma_prev = to_np(gfsigma.vec)
            p_prev     = p_np.copy()

            if verbose:
                print(f"  iter {it:3d} | "
                      f"flow={err1:.3e}  mech={err2:.3e} | "
                      f"U2(top)={U2_value:.6e}")

        iterations_each_step.append(it + 1)

        gfsigma_old.vec.data = gfsigma.vec.data
        gfp_old.vec.data     = gfp.vec.data

        if verbose:
            print(f"Converged in {iterations_each_step[-1]} iterations.  "
                  f"U2(top)={U2_value:.6e}")

        if float(t_now) in record_times_set:
            p_num = np.array([float(eval_gf_2d(gfp, mesh, xx, y_plot))
                              for xx in x_plot])
            p_ex  = p_analytic(x_plot, t_now, a=a, F=F, B=B, nu_u=nu_u,
                               c=c_diff, alphas=alphas, Cp=Cp, ca=ca)
            snapshots_p[float(t_now)] = (x_plot.copy(), p_num, p_ex)

            u1_num = np.array([float(eval_gf_2d(gfu[0], mesh, xx, y_plot))
                               for xx in x_plot])
            u1_ex  = u1_analytic_series(x_plot, t_now, a=a, F=F, G=mu,
                                        nu=nu, nu_u=nu_u, c=c_diff,
                                        alphas=alphas)
            snapshots_u1[float(t_now)] = (x_plot.copy(), u1_num, u1_ex)

            u2_num = np.array([float(eval_gf_2d(gfu[1], mesh, x_mid, yy))
                               for yy in y_plot_vec])
            u2_ex  = u2_analytic_linear_in_y(y_plot_vec,
                                             U2_top=U2_value, b=b)
            snapshots_u2[float(t_now)] = (y_plot_vec.copy(), u2_num, u2_ex)

            s11_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[0]
                                for xx in x_plot])
            s12_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[1]
                                for xx in x_plot])
            s21_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[2]
                                for xx in x_plot])
            s22_num = np.array([eval_sigma_components(gfsigma, mesh, xx, y_plot)[3]
                                for xx in x_plot])

            s11_ex = sigma11_analytic(
                x_plot, t_now, a=a, b=b, F=F, B=B, nu=nu, nu_u=nu_u,
                c=c_diff, alphas=alphas, Cp=Cp, Cu=Cu, ca=ca,
                lam=lam, mu=mu, alpha=alpha)
            s22_ex = sigma22_analytic(
                x_plot, t_now, a=a, b=b, F=F, B=B, nu=nu, nu_u=nu_u,
                c=c_diff, alphas=alphas, Cp=Cp, Cu=Cu, ca=ca,
                lam=lam, mu=mu, alpha=alpha)

            snapshots_s11[float(t_now)] = (x_plot.copy(), s11_num, s11_ex)
            snapshots_s22[float(t_now)] = (x_plot.copy(), s22_num, s22_ex)
            snapshots_s12[float(t_now)] = (x_plot.copy(), s12_num,
                                           np.zeros_like(x_plot))
            snapshots_s21[float(t_now)] = (x_plot.copy(), s21_num,
                                           np.zeros_like(x_plot))

            qx_num = np.array([float(eval_gf_2d(gfw[0], mesh, xx, y_plot))
                                for xx in x_plot])
            qx_ex  = qx_analytic(x_plot, t_now, a=a, F=F, B=B, nu_u=nu_u,
                                  c=c_diff, k_perm=k_perm, eta=eta,
                                  alphas=alphas, Cp=Cp)
            snapshots_qx[float(t_now)] = (x_plot.copy(), qx_num, qx_ex)

            xm, qx_cavg_num = qx_cell_avg_numerical(gfw[0], mesh, x_edges, y_plot)
            xm_ex, qx_cavg_ex = qx_cell_avg_analytic(
                x_edges, t_now, a=a, F=F, B=B, nu_u=nu_u, c=c_diff,
                k_perm=k_perm, eta=eta, alphas=alphas, Cp=Cp)
            snapshots_qx_avg[float(t_now)] = (xm, qx_cavg_num, qx_cavg_ex)

            eps_gf   = 0.5 * (Grad(gfu) + Grad(gfu).trans)
            divu     = Trace(Grad(gfu))
            sigma_tot_rec = (2 * mu * eps_gf
                             + lam * divu * Id(mesh.dim)
                             - alpha * gfp * Id(mesh.dim))
            s22_rec = np.array([
                float(sigma_tot_rec[1, 1](mesh(xx, y_plot)))
                for xx in x_plot
            ])
            s22_chk = np.array([
                eval_sigma_components(gfsigma, mesh, xx, y_plot)[3]
                for xx in x_plot
            ])
            print(f"  ||gfsigma22 - rec_tot22|| = "
                  f"{np.linalg.norm(s22_chk - s22_rec):.4e}  "
                  f"max abs = {np.max(np.abs(s22_chk - s22_rec)):.4e}")

    if verbose:
        print("\nIterations per step:", iterations_each_step)

    return {
        "mesh":         mesh,
        "gfp":          gfp,
        "gfw":          gfw,
        "gfsigma":      gfsigma,
        "gfu":          gfu,
        "gfomega":      gfomega,
        "iterations":   np.array(iterations_each_step, dtype=int),
        "U2_used":      np.array(U2_used, dtype=float),
        "snapshots_p":  snapshots_p,
        "snapshots_u1": snapshots_u1,
        "snapshots_u2": snapshots_u2,
        "snapshots_s11": snapshots_s11,
        "snapshots_s22": snapshots_s22,
        "snapshots_s12": snapshots_s12,
        "snapshots_s21": snapshots_s21,
        "snapshots_qx":      snapshots_qx,
        "snapshots_qx_avg":  snapshots_qx_avg,
        "params": dict(a=a, b=b, dt=dt, F=F, B=B,
                       nu=nu, nu_u=nu_u, c=c_diff,
                       G=mu, lam=lam, mu=mu, alpha=alpha,
                       k_perm=k_perm, eta=eta),
        "beta": beta,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def build_time_to_color(times):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    times = sorted(float(t) for t in times)
    return {t: colors[i % len(colors)] for i, t in enumerate(times)}


def _time_legend_inside_best(time_to_color, times, ncol=1, fontsize=9, frameon=False):
    ax = plt.gca()
    handles = []
    for t in sorted(float(tt) for tt in times):
        c = time_to_color[float(t)]
        handles.append(
            Line2D([0], [0], linestyle="-", linewidth=2.0, color=c, label=rf"$t={t:g}$")
        )
    leg = ax.legend(handles=handles, loc="best", ncol=ncol,
                    frameon=frameon, fontsize=fontsize,
                    handlelength=2.0, handletextpad=0.6,
                    labelspacing=0.3, borderaxespad=0.5)
    return leg


def _method_legend_top_row_fig(show_analytical=True, show_off=True, show_on=True):
    handles = []

    if show_analytical:
        handles.append(
            Line2D([0], [0], linestyle="-", linewidth=LINE_W,
                   color="black", label="analytical")
        )

    if show_off:
        handles.append(
            Line2D([0], [0], linestyle="None", marker="o", markersize=MS_O,
                   markerfacecolor="none", markeredgewidth=MEW, color="black",
                   label=r"fixed-stress $(\beta=0)$")
        )

    if show_on:
        handles.append(
            Line2D([0], [0], linestyle="None", marker="x", markersize=MS_X,
                   markeredgewidth=MEW, color="black",
                   label=r"fixed-stress $(\beta>0)$")
        )

    if handles:
        fig = plt.gcf()
        fig.legend(handles=handles,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.995),
                   ncol=len(handles),
                   frameon=True,
                   columnspacing=1.2,
                   handletextpad=0.6,
                   borderaxespad=0.0)
        fig.subplots_adjust(top=0.86)


def _plot_compare_generic(out_off, out_on, key, xmap, ymap,
                          xlabel, ylabel,
                          time_to_color,
                          annotate_times=True,
                          ann_frac=0.90, ann_dx=0.01, ann_dy_factor=0.01,
                          time_text_color="same",
                          show_time_legend=False,
                          time_legend_times=None,
                          time_legend_ncol=1,
                          time_legend_fontsize=9,
                          time_legend_frameon=False,
                          save_name=None,
                          xlim=(0.0, 1.0),
                          show_analytical=True,
                          show_off=True,
                          show_on=True,
                          markevery_off=3,
                          markevery_on=3):
    p_off = out_off["params"]
    p_on  = out_on["params"]

    snaps_off = out_off[key]
    snaps_on  = out_on[key]
    assert snaps_off.keys() == snaps_on.keys(), f"record_times mismatch for key={key}"

    plt.figure()

    for t in sorted(snaps_off.keys()):
        tt = float(t)
        if tt not in time_to_color:
            raise KeyError(f"time {tt} not found in time_to_color")
        c = time_to_color[tt]

        x_raw, yoff_num_raw, y_ex_raw = snaps_off[t]
        _,     yon_num_raw,  _        = snaps_on[t]

        x     = xmap(x_raw, p_off)
        y_ex  = ymap(y_ex_raw,     p_off)
        y_off = ymap(yoff_num_raw, p_off)
        y_on  = ymap(yon_num_raw,  p_on)

        if show_analytical:
            plt.plot(x, y_ex, linestyle="-", linewidth=LINE_W, color=c, zorder=1)

        if show_off:
            kwargs_off = {}
            if markevery_off is not None:
                kwargs_off["markevery"] = markevery_off
            plt.plot(x, y_off, linestyle="None", marker="o",
                     markersize=MS_O, markerfacecolor="none",
                     markeredgewidth=MEW, color=c, zorder=3, **kwargs_off)

        if show_on:
            kwargs_on = {}
            if markevery_on is not None:
                kwargs_on["markevery"] = markevery_on
            plt.plot(x, y_on, linestyle="None", marker="x",
                     markersize=MS_X, markeredgewidth=MEW,
                     color=c, zorder=4, **kwargs_on)

        if annotate_times:
            y_candidates = []
            if show_analytical:
                y_candidates.append(y_ex)
            if show_off:
                y_candidates.append(y_off)
            if show_on:
                y_candidates.append(y_on)

            if y_candidates:
                idx = int(ann_frac * (len(x) - 1))
                ymax = max(np.max(y) for y in y_candidates)
                ymin = min(np.min(y) for y in y_candidates)
                yr = ymax - ymin
                dy = ann_dy_factor * yr if yr > 0 else 0.0
                txtc = c if time_text_color == "same" else "black"

                if show_analytical:
                    yref = y_ex[idx]
                elif show_off:
                    yref = y_off[idx]
                else:
                    yref = y_on[idx]

                plt.text(x[idx] + ann_dx, yref + dy, rf"$t={tt:g}$",
                         fontsize=10, color=txtc)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(*xlim)
        plt.margins(x=0)

    _method_legend_top_row_fig(show_analytical=show_analytical,
                               show_off=show_off,
                               show_on=show_on)

    if show_time_legend:
        ax = plt.gca()
        leg1 = ax.get_legend()
        if leg1 is not None:
            ax.add_artist(leg1)

        if time_legend_times is None:
            time_legend_times = sorted(snaps_off.keys())

        _time_legend_inside_best(time_to_color, time_legend_times,
                                 ncol=time_legend_ncol,
                                 fontsize=time_legend_fontsize,
                                 frameon=time_legend_frameon)

    if save_name is not None:
        savefig_pub(save_name)

    plt.show()


def _interp_y(x, y, x_target):
    return float(np.interp(x_target, x, y))


def plot_compare_pressure_manual_labels(out_off, out_on,
                                        x_target=0.93,
                                        default_offset=(0.01, 0.0),
                                        label_offsets=None,
                                        time_text_color="same",
                                        time_to_color=None,
                                        save_name=None,
                                        show_analytical=True,
                                        show_off=True,
                                        show_on=True,
                                        markevery_off=3,
                                        markevery_on=3):
    if label_offsets is None:
        label_offsets = {}
    if time_to_color is None:
        raise ValueError("time_to_color must be provided")

    p_off = out_off["params"]
    p_on  = out_on["params"]

    snaps_off = out_off["snapshots_p"]
    snaps_on  = out_on["snapshots_p"]
    assert snaps_off.keys() == snaps_on.keys(), "record_times mismatch between off/on"

    plt.figure()

    for t in sorted(snaps_off.keys()):
        tt = float(t)
        if tt not in time_to_color:
            raise KeyError(f"time {tt} not found in time_to_color")
        c = time_to_color[tt]

        x_raw, pnum_off_raw, p_ex_raw = snaps_off[t]
        _,     pnum_on_raw,  _        = snaps_on[t]

        x    = x_raw / p_off["a"]
        p_ex = (p_off["a"] / p_off["F"]) * p_ex_raw
        p0   = (p_off["a"] / p_off["F"]) * pnum_off_raw
        pn   = (p_on["a"]  / p_on["F"])  * pnum_on_raw

        if show_analytical:
            plt.plot(x, p_ex, linestyle="-", linewidth=LINE_W, color=c, zorder=1)

        if show_off:
            kwargs_off = {}
            if markevery_off is not None:
                kwargs_off["markevery"] = markevery_off
            plt.plot(x, p0, linestyle="None", marker="o", markersize=MS_O,
                     markerfacecolor="none", markeredgewidth=MEW,
                     color=c, zorder=3, **kwargs_off)

        if show_on:
            kwargs_on = {}
            if markevery_on is not None:
                kwargs_on["markevery"] = markevery_on
            plt.plot(x, pn, linestyle="None", marker="x", markersize=MS_X,
                     markeredgewidth=MEW, color=c, zorder=4, **kwargs_on)

        dx, dy = label_offsets.get(tt, default_offset)
        txtc = c if time_text_color == "same" else "black"

        if show_analytical:
            yref = _interp_y(x, p_ex, x_target)
        elif show_off:
            yref = _interp_y(x, p0, x_target)
        elif show_on:
            yref = _interp_y(x, pn, x_target)
        else:
            continue

        plt.text(x_target + dx, yref + dy, rf"$t={tt:g}$", fontsize=10, color=txtc)

    plt.xlabel("x/a")
    plt.ylabel(r"$a\,p/F$")
    plt.xlim(0.0, 1.0)
    plt.margins(x=0)

    _method_legend_top_row_fig(show_analytical=show_analytical,
                               show_off=show_off,
                               show_on=show_on)

    if save_name is not None:
        savefig_pub(save_name)

    plt.show()


def plot_compare_u1(out_off, out_on, time_to_color, save_name=None,
                    show_analytical=True, show_off=True, show_on=True,
                    markevery_off=3, markevery_on=3):
    def xmap(x, prm):
        return x / prm["a"]

    def ymap(u1, prm):
        return (prm["G"] / prm["F"]) * u1

    _plot_compare_generic(
        out_off, out_on, "snapshots_u1", xmap, ymap,
        xlabel="x/a",
        ylabel=r"$(G/F)\,\mathbf{u}_1$",
        time_to_color=time_to_color,
        annotate_times=False,
        show_time_legend=True,
        time_legend_times=out_off["snapshots_p"].keys(),
        time_legend_ncol=1,
        time_legend_fontsize=9,
        time_legend_frameon=False,
        save_name=save_name,
        show_analytical=show_analytical,
        show_off=show_off,
        show_on=show_on,
        markevery_off=markevery_off,
        markevery_on=markevery_on
    )


def plot_compare_u2(out_off, out_on, time_to_color, save_name=None,
                    show_analytical=True, show_off=True, show_on=True,
                    markevery_off=3, markevery_on=3):
    def xmap(y, prm):
        return y / prm["b"]

    def ymap(u2, prm):
        return (prm["G"] * prm["a"] / (prm["F"] * prm["b"])) * u2

    _plot_compare_generic(
        out_off, out_on, "snapshots_u2", xmap, ymap,
        xlabel="y/b",
        ylabel=r"$(G\,a/(F\,b))\,\mathbf{u}_2$",
        time_to_color=time_to_color,
        annotate_times=False,
        show_time_legend=True,
        time_legend_times=out_off["snapshots_p"].keys(),
        time_legend_ncol=1,
        time_legend_fontsize=9,
        time_legend_frameon=False,
        save_name=save_name,
        show_analytical=show_analytical,
        show_off=show_off,
        show_on=show_on,
        markevery_off=markevery_off,
        markevery_on=markevery_on
    )


def plot_compare_sigma22(out_off, out_on, time_to_color, save_name=None,
                         show_analytical=True, show_off=True, show_on=True,
                         markevery_off=3, markevery_on=3):
    def xmap(x, prm):
        return x / prm["a"]

    def ymap(s, prm):
        return (prm["a"] / prm["F"]) * s

    _plot_compare_generic(
        out_off, out_on, "snapshots_s22", xmap, ymap,
        xlabel="x/a",
        ylabel=r"$a\,\underline{\boldsymbol{\sigma}}_{22}/F$",
        time_to_color=time_to_color,
        annotate_times=False,
        show_time_legend=True,
        time_legend_times=out_off["snapshots_p"].keys(),
        time_legend_ncol=1,
        time_legend_fontsize=9,
        time_legend_frameon=False,
        save_name=save_name,
        show_analytical=show_analytical,
        show_off=show_off,
        show_on=show_on,
        markevery_off=markevery_off,
        markevery_on=markevery_on
    )


def plot_compare_qx(out_off, out_on, time_to_color, save_name=None,
                    show_analytical=True, show_off=True, show_on=True,
                    markevery_off=3, markevery_on=3):
    def xmap(x, prm):
        return x / prm["a"]

    def ymap(qx, prm):
        return (prm["eta"] * prm["a"]**2 / (prm["k_perm"] * prm["F"])) * qx

    _plot_compare_generic(
        out_off, out_on, "snapshots_qx", xmap, ymap,
        xlabel="x/a",
        ylabel=r"$(\eta\,a^2/(kF))\,\mathbf{w}_1$",
        time_to_color=time_to_color,
        annotate_times=False,
        show_time_legend=True,
        time_legend_times=out_off["snapshots_p"].keys(),
        time_legend_ncol=1,
        time_legend_fontsize=9,
        time_legend_frameon=False,
        save_name=save_name,
        show_analytical=show_analytical,
        show_off=show_off,
        show_on=show_on,
        markevery_off=markevery_off,
        markevery_on=markevery_on
    )


def plot_compare_qx_avg(out_off, out_on, time_to_color, save_name=None,
                        show_analytical=True, show_off=True, show_on=True,
                        markevery_off=None, markevery_on=None):
    p_off = out_off["params"]
    p_on  = out_on["params"]

    snaps_off = out_off["snapshots_qx_avg"]
    snaps_on  = out_on["snapshots_qx_avg"]
    assert snaps_off.keys() == snaps_on.keys(), "record_times mismatch for qx_avg"

    plt.figure()

    for t in sorted(snaps_off.keys()):
        tt = float(t)
        if tt not in time_to_color:
            raise KeyError(f"time {tt} not found in time_to_color")
        c = time_to_color[tt]

        xm_raw, qxoff_raw, qxex_raw = snaps_off[t]
        _,      qxon_raw,  _        = snaps_on[t]

        x     = xm_raw / p_off["a"]
        y_ex  = (p_off["eta"] * p_off["a"]**2 / (p_off["k_perm"] * p_off["F"])) * qxex_raw
        y_off = (p_off["eta"] * p_off["a"]**2 / (p_off["k_perm"] * p_off["F"])) * qxoff_raw
        y_on  = (p_on["eta"]  * p_on["a"]**2  / (p_on["k_perm"]  * p_on["F"]))  * qxon_raw

        if show_analytical:
            plt.plot(x, y_ex, linestyle='-', linewidth=LINE_W, color=c, zorder=1)

        if show_off:
            kwargs_off = {}
            if markevery_off is not None:
                kwargs_off["markevery"] = markevery_off
            plt.plot(x, y_off, linestyle='None', marker='o',
                     markersize=MS_O, markerfacecolor='none',
                     markeredgewidth=MEW, color=c, zorder=3, **kwargs_off)

        if show_on:
            kwargs_on = {}
            if markevery_on is not None:
                kwargs_on["markevery"] = markevery_on
            plt.plot(x, y_on, linestyle='None', marker='x',
                     markersize=MS_X, markeredgewidth=MEW, color=c, zorder=4, **kwargs_on)

    plt.xlabel(r"$x_{\mathrm{mid}}/a$")
    plt.ylabel(r"$(\eta\,a^2/(kF))\,\langle \mathbf{w}_1\rangle_{\mathrm{cell}}$")

    _method_legend_top_row_fig(show_analytical=show_analytical,
                               show_off=show_off,
                               show_on=show_on)
    _time_legend_inside_best(time_to_color, sorted(snaps_off.keys()),
                             ncol=1, fontsize=9, frameon=False)
    plt.xlim(0.0, 1.0)
    plt.margins(x=0)
    if save_name is not None:
        savefig_pub(save_name)

    plt.show()


# =============================================================================
# Driver
# =============================================================================

def run_or_load_case(case_name, tuning, force_rerun=False, **kwargs):
    out_path = os.path.join(RESULTS_DIR, f"{case_name}.pkl")
    if (not force_rerun) and os.path.isfile(out_path):
        print(f"Loading saved result: {out_path}")
        return load_pickle(out_path)

    print(f"Running case: {case_name}")
    out = mandel_splitting_U2BC(tuning=tuning, **kwargs)
    save_pickle(out, out_path)
    print(f"Saved result: {out_path}")
    return out


def main():
    print("=== Mandel test (quarter domain), BCs (37)-(40), u2 = U2(top,t) imposed ===")

    record_times = (10, 50, 1000, 5000, 10000, 20000, 50000) 

    common_kwargs = dict(
        k=1,
        a=100.0,
        b=10.0,
        hx=2.5,
        hy=0.25,
        dt=10.0,
        Tfinal=50000.0,
        tol=1e-6,
        max_iterations=300,
        F=1.0,  # load scaling; plots are F-normalized and iteration counts are F-invariant (Table 3: F = 6.0e8)
        record_times=record_times,
        verbose=True,
    )

    out_off = run_or_load_case(
        case_name="mandel_profiles_off",
        tuning="off",
        force_rerun=False,
        **common_kwargs
    )

    out_on = run_or_load_case(
        case_name="mandel_profiles_on",
        tuning="on",
        force_rerun=False,
        **common_kwargs
    )

    out_on1d = run_or_load_case(
        case_name="mandel_profiles_on1d",
        tuning="on1d",
        force_rerun=False,
        **common_kwargs
    )

    # plots remain only off vs on
    time_to_color = build_time_to_color(out_off["snapshots_p"].keys())

    plot_compare_u1(out_off, out_on, time_to_color=time_to_color, save_name="mandel_u1")
    plot_compare_u2(out_off, out_on, time_to_color=time_to_color, save_name="mandel_u2")
    plot_compare_sigma22(out_off, out_on, time_to_color=time_to_color, save_name="mandel_sigma22")
    plot_compare_qx(out_off, out_on, time_to_color=time_to_color, save_name="mandel_wx")
    plot_compare_qx_avg(out_off, out_on, time_to_color=time_to_color, save_name="mandel_wx_avg")

    label_offsets = {
        10.0:    (-0.029,  0.02),
        50.0:    (-0.13,   0.05),
        1000.0:  (-0.46,   0.25),
        5000.0:  (-0.75,   0.25),
        10000.0: (-0.75,   0.15),
        20000.0: (-0.75,   0.05),
        50000.0: (-0.75,  -0.018),
    }

    plot_compare_pressure_manual_labels(
        out_off, out_on,
        x_target=0.93,
        default_offset=(0.01, 0.0),
        label_offsets=label_offsets,
        time_to_color=time_to_color,
        save_name="mandel_pressure"
    )

    # iteration-summary outputs for off / on / on1d
    write_iteration_summary(
        {
            "off": out_off,
            "on": out_on,
            "on1d": out_on1d,
        },
        csv_path=ITER_TABLE_CSV,
        tex_path=ITER_TABLE_TEX,
    )


if __name__ == "__main__":
    main()