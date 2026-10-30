#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mandel's problem: scaling study of the fixed-stress splitting
(Section 6.2, Table 4 of the paper).

  E = gamma_3 * E_ref,  F = gamma_3 * F_ref,  E_ref = 5.94e9,  F_ref = 6.0e8,
  gamma_3 in {1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3},  k in {1, 2, 3},
  destabilization beta = theta * beta_max,  theta in {0, 1/4, 1/2, 3/4, 1},
  beta_max = M/2,  M = d alpha^2 / (2 mu + d lambda)   (Lemma 3, paper sign beta >= 0),
  time window [0, 100] (10 steps of dt = 10); reported: average iterations per step.

  theta = 0 is the plain fixed-stress split, theta = 1 the theoretical optimum beta_max.
  Coupling strength tau = M / c0 (= 4 * gamma_3^{-1} for the parameters of Table 3).

All other material data (Table 3) are kept fixed; the Skempton coefficient B, the undrained
Poisson ratio nu_u and the Mandel diffusivity c are derived consistently from (E, nu, alpha, M_b).

Outputs (results/):
  * mandel_scaling_summary.csv : one row per (gamma_3, k, theta), incl. per-step iterations
  * mandel_scaling_table.tex   : Table 4 (booktabs; per-row minimum in bold)
Set TABLE_ONLY = True to rebuild the table from the csv without recomputing.
Set COMPUTE_ERRORS = True to additionally record the (time-L2) discretization errors.

Author: Tugay Dagli
"""

from ngsolve import *
import ngsolve
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse import hstack, vstack
from scipy.optimize import brentq
import csv
import os


# =============================================================================
# Paths and settings
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_CSV = "mandel_scaling_summary.csv"
TABLE_TEX = "mandel_scaling_table.tex"

E_REF = 5.94e9
F_REF = 6.0e8
GAMMA3_VALUES = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
K_VALUES = [1, 2, 3]
THETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

TABLE_ONLY = False       # True: skip computation, rebuild the table from results/mandel_scaling_summary.csv
COMPUTE_ERRORS = False   # True: also compute per-step errors and time-L2 errors (expensive; not in Table 4)
VERBOSE = False          # True: print every fixed-stress iteration

SOLVER_KWARGS = dict(
    a=100.0,
    b=10.0,
    hx=2.5,
    hy=0.25,
    dt=10.0,
    Tfinal=100.0,
    tol=1e-6,
    max_iterations=300,
    nu=0.2,
    alpha=1.0,
    M=1.65e10,
    n_terms=80,
    graded=False,
    time_error_window=(10.0, 100.0),
)


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


def eval_gf_2d(gf, mesh, xval, yval):
    """Point evaluation helper."""
    try:
        return gf(mesh(xval, yval))
    except Exception:
        return gf(xval, yval)


def eval_sigma_components(gfsigma, mesh, xval, yval):
    """
    Evaluate total stress components at a point.
    gfsigma on Sigma = [row1,row2], row1=(s11,s12), row2=(s21,s22).
    """
    pt = mesh(xval, yval)
    r1 = gfsigma.components[0](pt)
    r2 = gfsigma.components[1](pt)
    return float(r1[0]), float(r1[1]), float(r2[0]), float(r2[1])


def safe_rel(err, ref, eps=1e-30):
    return err / max(ref, eps)


def scaled_load_from_E(Eyoung, E_ref=5.94e9, F_ref=6.0e8):
    """
    Scale the applied load F linearly with Eyoung using the reference pair
        (E_ref, F_ref).

    Thus:
        F = F_ref * (Eyoung / E_ref)
    """
    if E_ref <= 0.0:
        raise ValueError("E_ref must be positive.")
    return F_ref * (Eyoung / E_ref)


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
    x_split = n_coarse * hx_coarse
    L_fine = a - x_split

    H = L_fine * (1.0 - r) / (1.0 - r**n_fine)
    widths = np.array([H * r**i for i in range(n_fine)])
    hx_fine = widths[-1]

    x_coarse = np.linspace(0.0, x_split, n_coarse + 1)
    x_fine = x_split + np.concatenate([[0.0], np.cumsum(widths)])
    x_nodes = np.concatenate([x_coarse, x_fine[1:]])
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
# Consistent isotropic poroelastic parameters
# =============================================================================

def isotropic_biot_derived_params(Eyoung, nu, alpha, M):
    """
    Derived isotropic poroelastic parameters from (E, nu, alpha, M).

    Returns
    -------
    lam    : Lamé parameter
    mu     : shear modulus
    Kdr    : drained bulk modulus
    Ku     : undrained bulk modulus
    B      : Skempton coefficient
    nu_u   : undrained Poisson ratio
    """
    lam = Eyoung * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = Eyoung / (2.0 * (1.0 + nu))

    Kdr = lam + 2.0 * mu / 3.0
    Ku  = Kdr + alpha**2 * M

    B = alpha * M / Ku
    nu_u = (3.0 * Ku - 2.0 * mu) / (2.0 * (3.0 * Ku + mu))

    return lam, mu, Kdr, Ku, B, nu_u


def mandel_diffusivity_isotropic(mu, lam, M, k_perm, eta, alpha=1.0):
    """
    Isotropic Mandel diffusivity:
        c1 = (k_perm/eta) * M * M11 / M11u
    with
        M11  = 2*mu + lam
        M11u = M11 + alpha^2 * M
    """
    M11 = 2.0 * mu + lam
    M11u = M11 + alpha**2 * M
    return (k_perm / eta) * M * M11 / M11u


def check_physical_params(Eyoung, nu, alpha, M, k_perm, eta):
    """Basic sanity checks."""
    if Eyoung <= 0.0:
        raise ValueError("Eyoung must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must satisfy -1 < nu < 0.5.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if M <= 0.0:
        raise ValueError("M must be positive.")
    if k_perm <= 0.0:
        raise ValueError("k_perm must be positive.")
    if eta <= 0.0:
        raise ValueError("eta must be positive.")


# =============================================================================
# Mandel analytical helpers
# =============================================================================

def mandel_roots(n_terms, nu, nu_u):
    """
    Characteristic roots alpha_i satisfying tan(alpha_i) = m*alpha_i,
    where m = (1-nu)/(nu_u - nu).
    """
    m = (1.0 - nu) / (nu_u - nu)
    eps = 1e-12
    alphas = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        aL = (n - 1) * np.pi + eps
        aR = (n - 0.5) * np.pi - eps
        f = lambda aval: np.tan(aval) - m * aval
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


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------

def p_analytic(xval, t, a, F, B, nu_u, c, alphas, Cp, ca):
    xval = np.asarray(xval, dtype=float)
    p0 = (B * (1.0 + nu_u) * F) / (3.0 * a)
    cos_ax = np.cos(np.outer(alphas, xval / a))
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    s = (Cp[:, None] * (cos_ax - ca[:, None]) * expo[:, None]).sum(axis=0)
    return 2.0 * p0 * s


def p_initial_exact(a, F, B, nu_u, xval):
    xval = np.asarray(xval, dtype=float)
    p0 = (B * (1.0 + nu_u) * F) / (3.0 * a)
    return p0 * np.ones_like(xval)


# ---------------------------------------------------------------------------
# Displacements
# ---------------------------------------------------------------------------

def U2_top_series(t, a, b, F, nu, nu_u, G, c, alphas, Cu):
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    bracket = (-(F * (1.0 - nu)) / (2.0 * G * a)
               + (F * (1.0 - nu_u)) / (G * a) * np.sum(Cu * expo))
    return bracket * b


def u1_analytic_series(xval, t, a, F, G, nu, nu_u, c, alphas):
    xval = np.asarray(xval, dtype=float)
    sa = np.sin(alphas)
    ca = np.cos(alphas)
    denom = alphas - sa * ca
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    sum1 = np.sum((sa * ca / denom) * expo)
    sum2 = np.sum((ca / denom)[:, None]
                  * np.sin(np.outer(alphas, xval / a))
                  * expo[:, None], axis=0)
    return (F * nu / (2 * G * a) - F * nu_u / (G * a) * sum1) * xval + (F / G) * sum2


def u2_analytic_linear_in_y(yval, U2_top, b):
    yval = np.asarray(yval, dtype=float)
    return U2_top * (yval / b)


def u1_initial_exact(xval, a, F, G, nu_u):
    xval = np.asarray(xval, dtype=float)
    return (F * nu_u / (2.0 * G * a)) * xval


def u2_initial_exact(yval, a, F, G, nu_u):
    yval = np.asarray(yval, dtype=float)
    return -(F * (1.0 - nu_u) / (2.0 * G * a)) * yval


# ---------------------------------------------------------------------------
# Total stresses
# ---------------------------------------------------------------------------

def sigma22_analytic(xval, t, a, F, B, nu, nu_u, c,
                     alphas, Cp, Cu, ca, alpha=1.0):
    xval = np.asarray(xval, dtype=float)
    A1, A2 = mandel_A1_A2(B, nu, nu_u, alpha)

    expo = np.exp(-(alphas**2) * c * t / (a**2))
    cos_ax = np.cos(np.outer(alphas, xval / a))

    sum_cos = np.sum(Cp[:, None] * cos_ax * expo[:, None], axis=0)
    sum_cu = np.sum(Cu * expo)

    return -F / a - (2.0 * F * A2) / (a * A1) * sum_cos + (2.0 * F / a) * sum_cu


def sigma11_analytic(xval):
    xval = np.asarray(xval, dtype=float)
    return np.zeros_like(xval)


def sigma22_initial_exact(xval, a, F):
    xval = np.asarray(xval, dtype=float)
    return np.full_like(xval, -F / a)


def sigma11_initial_exact(xval):
    xval = np.asarray(xval, dtype=float)
    return np.zeros_like(xval)


# ---------------------------------------------------------------------------
# Darcy velocity
# ---------------------------------------------------------------------------

def qx_analytic(xval, t, a, F, B, nu_u, c, k_perm, eta, alphas, Cp):
    xval = np.asarray(xval, dtype=float)
    A1 = 3.0 / (B * (1.0 + nu_u))
    expo = np.exp(-(alphas**2) * c * t / (a**2))
    sin_ax = np.sin(np.outer(alphas, xval / a))
    return ((2.0 * F * k_perm) / (a**2 * eta * A1)
            * np.sum(alphas[:, None] * Cp[:, None] * sin_ax * expo[:, None],
                     axis=0))


def qx_initial_exact(xval):
    xval = np.asarray(xval, dtype=float)
    return np.zeros_like(xval)


# =============================================================================
# Exact coefficient functions for integrated errors
# =============================================================================

def p_exact_cf(t, a, F, B, nu_u, c, alphas, Cp, ca):
    p0 = (B * (1.0 + nu_u) * F) / (3.0 * a)
    s = 0
    for i, ai in enumerate(alphas):
        s += Cp[i] * (cos(ai * x / a) - ca[i]) * exp(-(ai**2) * c * t / (a**2))
    return 2.0 * p0 * s


def u_exact_cf(t, a, b, F, mu, nu, nu_u, c, alphas):
    sa = np.sin(alphas)
    ca = np.cos(alphas)
    denom = alphas - sa * ca
    expo = np.exp(-(alphas**2) * c * t / (a**2))

    sum1 = np.sum((sa * ca / denom) * expo)

    ux_series = 0
    for i, ai in enumerate(alphas):
        ux_series += (ca[i] / denom[i]) * sin(ai * x / a) * expo[i]

    ux = (F * nu / (2 * mu * a) - F * nu_u / (mu * a) * sum1) * x + (F / mu) * ux_series
    uy_factor = (-F * (1.0 - nu) / (2.0 * mu * a)
                 + F * (1.0 - nu_u) / (mu * a) * sum1)
    uy = uy_factor * y

    return CoefficientFunction((ux, uy))


def sigma22_exact_cf(t, a, F, B, nu, nu_u, c, alphas, Cp, ca):
    s = 0
    factor = (nu_u - nu) / (1.0 - nu)
    for i, ai in enumerate(alphas):
        s += Cp[i] * (factor * cos(ai * x / a) - ca[i]) * exp(-(ai**2) * c * t / (a**2))
    return -F / a - (2.0 * F / a) * s


def sigma_exact_rows_cf(t, a, F, B, nu, nu_u, c, alphas, Cp, ca):
    s22 = sigma22_exact_cf(t, a, F, B, nu, nu_u, c, alphas, Cp, ca)
    row1 = CoefficientFunction((0.0, 0.0))
    row2 = CoefficientFunction((0.0, s22))
    return row1, row2


def div_sigma_exact_rows_cf():
    """
    For the Mandel exact solution in this formulation:
      row1 = (0, 0), row2 = (0, sigma22(x,t))
    and sigma22 is independent of y, hence both row divergences are exactly zero.
    """
    return CoefficientFunction(0.0), CoefficientFunction(0.0)


def q_exact_cf(t, a, F, B, nu_u, c, k_perm, eta, alphas, Cp):
    A1 = 3.0 / (B * (1.0 + nu_u))
    pref = (2.0 * F * k_perm) / (a**2 * eta * A1)

    qx_cf = 0
    for i, ai in enumerate(alphas):
        qx_cf += ai * Cp[i] * sin(ai * x / a) * exp(-(ai**2) * c * t / (a**2))
    qx_cf *= pref

    return CoefficientFunction((qx_cf, 0.0))


def div_q_exact_cf(t, a, F, B, nu_u, c, k_perm, eta, alphas, Cp):
    A1 = 3.0 / (B * (1.0 + nu_u))
    pref = (2.0 * F * k_perm) / (a**3 * eta * A1)

    dqdx_cf = 0
    for i, ai in enumerate(alphas):
        dqdx_cf += (ai**2) * Cp[i] * cos(ai * x / a) * exp(-(ai**2) * c * t / (a**2))
    dqdx_cf *= pref

    return dqdx_cf


# =============================================================================
# Norms and errors
# =============================================================================

def l2_norm_scalar(cf, mesh, order=None):
    val = Integrate(cf * cf, mesh, order=order)
    return np.sqrt(abs(val))


def l2_norm_vector(cf, mesh, order=None):
    val = Integrate(InnerProduct(cf, cf), mesh, order=order)
    return np.sqrt(abs(val))


def hdiv_norm_vector(v_cf, div_v_cf, mesh, order=None):
    val = Integrate(InnerProduct(v_cf, v_cf) + div_v_cf * div_v_cf, mesh, order=order)
    return np.sqrt(abs(val))


def stress_hdiv_error(gfsigma, row1_ex, row2_ex, div_row1_ex, div_row2_ex, mesh, order=None):
    """
    True row-wise H(div) error:
      ||row1-row1_ex||^2 + ||div(row1-row1_ex)||^2
    + ||row2-row2_ex||^2 + ||div(row2-row2_ex)||^2
    """
    e1 = gfsigma.components[0] - row1_ex
    e2 = gfsigma.components[1] - row2_ex

    div_e1 = div(gfsigma.components[0]) - div_row1_ex
    div_e2 = div(gfsigma.components[1]) - div_row2_ex

    val = Integrate(
        InnerProduct(e1, e1) + InnerProduct(e2, e2)
        + div_e1 * div_e1 + div_e2 * div_e2,
        mesh, order=order
    )
    return np.sqrt(abs(val))


def stress_hdiv_norm_exact(row1_ex, row2_ex, div_row1_ex, div_row2_ex, mesh, order=None):
    val = Integrate(
        InnerProduct(row1_ex, row1_ex) + InnerProduct(row2_ex, row2_ex)
        + div_row1_ex * div_row1_ex + div_row2_ex * div_row2_ex,
        mesh, order=order
    )
    return np.sqrt(abs(val))


def compute_error_summary_at_time(
    mesh, gfp, gfu, gfw, gfsigma,
    t, a, b, F, B, nu, nu_u, c_used, mu, k_perm, eta,
    alphas, Cp, Cu, ca,
    order=None
):
    p_ex = p_exact_cf(t, a, F, B, nu_u, c_used, alphas, Cp, ca)
    u_ex = u_exact_cf(t, a, b, F, mu, nu, nu_u, c_used, alphas)
    s1_ex, s2_ex = sigma_exact_rows_cf(t, a, F, B, nu, nu_u, c_used, alphas, Cp, ca)
    div_s1_ex, div_s2_ex = div_sigma_exact_rows_cf()
    q_ex = q_exact_cf(t, a, F, B, nu_u, c_used, k_perm, eta, alphas, Cp)
    div_q_ex = div_q_exact_cf(t, a, F, B, nu_u, c_used, k_perm, eta, alphas, Cp)

    p_abs = l2_norm_scalar(gfp - p_ex, mesh, order=order)
    p_ref = l2_norm_scalar(p_ex, mesh, order=order)
    p_rel = safe_rel(p_abs, p_ref)

    u_abs = l2_norm_vector(gfu - u_ex, mesh, order=order)
    u_ref = l2_norm_vector(u_ex, mesh, order=order)
    u_rel = safe_rel(u_abs, u_ref)

    sigma_abs = stress_hdiv_error(
        gfsigma, s1_ex, s2_ex, div_s1_ex, div_s2_ex, mesh, order=order
    )
    sigma_ref = stress_hdiv_norm_exact(
        s1_ex, s2_ex, div_s1_ex, div_s2_ex, mesh, order=order
    )
    sigma_rel = safe_rel(sigma_abs, sigma_ref)

    q_err_cf = gfw - q_ex
    div_q_err_cf = div(gfw) - div_q_ex
    q_abs = hdiv_norm_vector(q_err_cf, div_q_err_cf, mesh, order=order)

    q_ref = hdiv_norm_vector(q_ex, div_q_ex, mesh, order=order)
    q_rel = safe_rel(q_abs, q_ref)

    return {
        "p_L2_abs": p_abs,
        "p_L2_rel": p_rel,
        "u_L2_abs": u_abs,
        "u_L2_rel": u_rel,
        "sigma_Hdiv_abs": sigma_abs,
        "sigma_Hdiv_rel": sigma_rel,
        "q_Hdiv_abs": q_abs,
        "q_Hdiv_rel": q_rel,
        "p_ref": p_ref,
        "u_ref": u_ref,
        "sigma_ref": sigma_ref,
        "q_ref": q_ref,
    }


def compute_time_L2_from_history(error_history, dt, time_window=None):
    if time_window is None:
        times = sorted(error_history.keys())
    else:
        t0, t1 = time_window
        times = sorted([t for t in error_history.keys() if t0 <= t <= t1])

    if len(times) == 0:
        return {}

    fields_abs = ["p_L2_abs", "u_L2_abs", "sigma_Hdiv_abs", "q_Hdiv_abs"]

    out = {}
    for f in fields_abs:
        out[f.replace("_abs", "_timeL2_abs")] = np.sqrt(
            dt * sum(error_history[t][f]**2 for t in times)
        )

    out["p_L2_timeL2_rel"] = safe_rel(
        out["p_L2_timeL2_abs"],
        np.sqrt(dt * sum(error_history[t]["p_ref"]**2 for t in times))
    )
    out["u_L2_timeL2_rel"] = safe_rel(
        out["u_L2_timeL2_abs"],
        np.sqrt(dt * sum(error_history[t]["u_ref"]**2 for t in times))
    )
    out["sigma_Hdiv_timeL2_rel"] = safe_rel(
        out["sigma_Hdiv_timeL2_abs"],
        np.sqrt(dt * sum(error_history[t]["sigma_ref"]**2 for t in times))
    )
    out["q_Hdiv_timeL2_rel"] = safe_rel(
        out["q_Hdiv_timeL2_abs"],
        np.sqrt(dt * sum(error_history[t]["q_ref"]**2 for t in times))
    )

    out["time_window_start"] = times[0]
    out["time_window_end"] = times[-1]
    out["num_times_used"] = len(times)
    return out


# =============================================================================
# Mandel solver
# =============================================================================

def mandel_splitting_U2BC(
    theta=0.0,
    k=1,
    a=100.0,
    b=10.0,
    hx=2.5,
    hy=0.25,
    nx=None,
    ny=None,
    dt=10.0,
    Tfinal=100.0,
    tol=1e-6,
    max_iterations=300,
    F=None,
    scale_F_with_E=True,
    E_ref_for_F=E_REF,
    F_ref_for_E=F_REF,
    Eyoung=E_REF,
    nu=0.2,
    alpha=1.0,
    M=1.65e10,
    n_terms=80,
    verbose=False,
    graded=False,
    graded_n_fine=12,
    graded_r=0.75,
    compute_space_errors=False,
    time_error_window=(10.0, 100.0),
):
    # ---- geometry-independent fixed material data
    k_perm = 0.1 * 9.869233e-13
    eta = 1.0e-3
    check_physical_params(Eyoung, nu, alpha, M, k_perm, eta)

    # ---- applied load scaling with E: F = gamma_3 * F_ref
    if scale_F_with_E:
        F = scaled_load_from_E(
            Eyoung=Eyoung,
            E_ref=E_ref_for_F,
            F_ref=F_ref_for_E
        )
    else:
        if F is None:
            raise ValueError("If scale_F_with_E is False, you must provide F explicitly.")

    # ---- resolve nx, ny and build x-node array
    if graded:
        x_nodes, n_coarse, hx_fine = make_graded_x_nodes(
            a, hx_coarse=hx, n_fine=graded_n_fine, r=graded_r
        )
        nx = len(x_nodes) - 1
    else:
        if nx is None:
            nx = int(round(a / hx))
        x_nodes = None

    if ny is None:
        ny = int(round(b / hy))

    mesh = make_structured_rect_mesh(a, b, nx, ny, x_nodes=x_nodes)

    # ---- physical parameters
    c0 = 1.0 / M
    K_over_eta = k_perm / eta
    eta_over_k = 1.0 / K_over_eta

    # derive all consistent poroelastic parameters from E
    lam, mu, Kdr, Ku, B, nu_u = isotropic_biot_derived_params(
        Eyoung=Eyoung, nu=nu, alpha=alpha, M=M
    )

    # exact Mandel diffusivity, updated consistently with E
    c_mandel = mandel_diffusivity_isotropic(
        mu=mu, lam=lam, M=M, k_perm=k_perm, eta=eta, alpha=alpha
    )
    c_used = c_mandel

    d = 2
    cr = (d * alpha**2) / (2 * mu + d * lam)   # M_paper = d alpha^2 / (2 mu + d lam)
    C1 = cr / (d * alpha)
    C2 = c0 + cr

    # destabilization in the paper's sign convention (beta >= 0, Lemma 3);
    # the assembled flow block is (c0~ - beta_paper) * M_p
    beta_max = 0.5 * cr
    beta_paper = theta * beta_max
    beta = -beta_paper

    # coupling strength tau = M / c0 = d alpha^2 / (c0 (2 mu + d lam))
    coupling_strength = cr / c0

    print(f"gamma_3 = {Eyoung / E_ref_for_F:g}, k = {k}, theta = {theta:g}: "
          f"E = {Eyoung:.4e}, F = {F:.4e}, beta = {beta_paper:.6e} (beta_max = {beta_max:.6e}), "
          f"tau = {coupling_strength:.6e}, B = {B:.6f}, nu_u = {nu_u:.6f}, c = {c_mandel:.6e}")

    # ---- analytic precompute
    alphas = mandel_roots(n_terms, nu=nu, nu_u=nu_u)
    Cp, Cu, ca = mandel_coeffs(alphas)

    # ---- FE spaces (flow)
    W = HDiv(mesh, order=k, RT=True, dirichlet="left|bottom|top")
    Qp = L2(mesh, order=k)
    WQ = FESpace([W, Qp])
    gfwq = GridFunction(WQ)
    gfw, gfp = gfwq.components
    gfp_old = GridFunction(Qp)

    # ---- FE spaces (mechanics)
    Sigma_row1 = HDiv(mesh, order=k, RT=True, dirichlet="bottom|top|right")
    Sigma_row2 = HDiv(mesh, order=k, RT=True, dirichlet="left|right")
    Sigma = FESpace([Sigma_row1, Sigma_row2])
    Omega = H1(mesh, order=k)
    U = VectorL2(mesh, order=k)

    Mech = FESpace([Sigma, Omega, U])
    gfmech = GridFunction(Mech)
    gfsigma, gfomega, gfu = gfmech.components
    gfsigma_old = GridFunction(Sigma)

    # ---- Trial / test functions
    w, v = W.TnT()
    p, q = Qp.TnT()
    (sigma1, sigma2) = Sigma.TrialFunction()
    (tau1, tau2) = Sigma.TestFunction()
    ome, gam = Omega.TnT()
    u, z = U.TnT()
    nrm = specialcf.normal(mesh.dim)

    def tr_rows(s1, s2):
        return s1[0] + s2[1]

    A_sigma1 = (1.0 / (2.0 * mu)) * (
        sigma1
        - (lam / (2.0 * mu + d * lam)) * tr_rows(sigma1, sigma2)
          * CoefficientFunction((1.0, 0.0))
    )
    A_sigma2 = (1.0 / (2.0 * mu)) * (
        sigma2
        - (lam / (2.0 * mu + d * lam)) * tr_rows(sigma1, sigma2)
          * CoefficientFunction((0.0, 1.0))
    )

    # ---- FLOW matrices
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
    TP = beta * Mp

    eSigP = BilinearForm(trialspace=Sigma, testspace=Qp)
    eSigP += ((sigma1[0] + sigma2[1]) * q) * dx
    eSigP.Assemble()
    Etr = C1 * to_csr(eSigP.mat)

    ndW = W.ndof
    ndQ = Qp.ndof

    LHS1 = vstack([
        hstack([A,             Bmat]),
        hstack([-dt * Bmat.T,  Cmat + TP])
    ]).tocsc()

    freeW = bitarray_to_bool(W.FreeDofs())
    free1 = np.ones(ndW + ndQ, dtype=bool)
    free1[:ndW] = freeW

    LHS1_bc = apply_row_dirichlet(LHS1, free1)
    B1 = sp.linalg.splu(LHS1_bc)

    # ---- MECH matrices
    Pbf = BilinearForm(Sigma)
    Pbf += (A_sigma1 * tau1 + A_sigma2 * tau2) * dx
    Pbf.Assemble()
    Pmat = to_csr(Pbf.mat)

    rbf = BilinearForm(trialspace=U, testspace=Sigma)
    rbf += (div(tau1) * u[0] + div(tau2) * u[1]) * dx
    rbf.Assemble()
    R = to_csr(rbf.mat)

    sbf = BilinearForm(trialspace=Sigma, testspace=Omega)
    sbf += ((sigma2[0] - sigma1[1]) * gam) * dx
    sbf.Assemble()
    S = to_csr(sbf.mat)

    ndS = Sigma.ndof
    ndO = Omega.ndof
    ndU = U.ndof
    ndSO = ndS + ndO

    ZERO_OO = sp.csr_matrix((ndO, ndO))
    ZERO_UU = sp.csr_matrix((ndU, ndU))

    PTILDE = vstack([
        hstack([Pmat,  S.T]),
        hstack([-S,    ZERO_OO])
    ]).tocsr()

    RTILDE = vstack([
        R,
        sp.csr_matrix((ndO, ndU))
    ]).tocsr()

    LHS2 = vstack([
        hstack([PTILDE,    RTILDE]),
        hstack([-RTILDE.T, ZERO_UU])
    ]).tocsc()

    freeSigma = bitarray_to_bool(Sigma.FreeDofs())
    free2 = np.ones(ndSO + ndU, dtype=bool)
    free2[:ndS] = freeSigma

    LHS2_bc = apply_row_dirichlet(LHS2, free2)
    B2 = sp.linalg.splu(LHS2_bc)

    # ---- Boundary linear form: u2 = U2(b,t) on top
    U2par = Parameter(0.0)
    lf_u2top = LinearForm(Sigma)
    lf_u2top += U2par * hdiv_normal_trace(tau2, nrm) * ds("top")

    # ---- Exact initial conditions at t = 0
    p0_val = (B * (1.0 + nu_u) * F) / (3.0 * a)
    G = mu

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

    # ---- Time stepping
    nsteps = int(round(Tfinal / dt))
    iterations_each_step = []
    hit_cap = False
    error_history = {}

    for nstep in range(1, nsteps + 1):
        t_now = nstep * dt

        U2_value = U2_top_series(
            t_now, a=a, b=b, F=F, nu=nu, nu_u=nu_u, G=mu,
            c=c_used, alphas=alphas, Cu=Cu
        )

        U2par.Set(U2_value)
        lf_u2top.Assemble()
        rhs_u2top = to_np(lf_u2top.vec)

        sigma_old_np = to_np(gfsigma_old.vec)
        p_old_np = to_np(gfp_old.vec)

        sigma_prev = sigma_old_np.copy()
        p_prev = p_old_np.copy()

        # first iteration (i = 1, initialized with previous time step data)
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
        it = 0

        sigma_prev = to_np(gfsigma.vec)
        p_prev = p_np.copy()

        # fixed-point iterations
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

            # stopping criterion (47): relative update of successive iterates
            err1 = np.linalg.norm(sol1 - prev_sol1) / (np.linalg.norm(prev_sol1) + 1e-14)
            err2 = np.linalg.norm(sol2 - prev_sol2) / (np.linalg.norm(prev_sol2) + 1e-14)
            prev_sol1 = sol1.copy()
            prev_sol2 = sol2.copy()

            sigma_prev = to_np(gfsigma.vec)
            p_prev = p_np.copy()

            if verbose:
                print(f"  iter {it:3d} | flow={err1:.3e}  mech={err2:.3e}")

        if err1 > tol or err2 > tol:
            hit_cap = True
            print(f"!! WARNING: iteration cap {max_iterations} reached at time step {nstep} (t={t_now:g})")

        iterations_each_step.append(it + 1)

        gfsigma_old.vec.data = gfsigma.vec.data
        gfp_old.vec.data = gfp.vec.data

        # ---- optional space errors at every time step
        if compute_space_errors and t_now >= 10.0:
            err_order = max(8, 2 * k + 6)
            error_history[float(t_now)] = compute_error_summary_at_time(
                mesh=mesh,
                gfp=gfp,
                gfu=gfu,
                gfw=gfw,
                gfsigma=gfsigma,
                t=t_now,
                a=a, b=b, F=F, B=B, nu=nu, nu_u=nu_u,
                c_used=c_used, mu=mu, k_perm=k_perm, eta=eta,
                alphas=alphas, Cp=Cp, Cu=Cu, ca=ca,
                order=err_order
            )

    its = np.array(iterations_each_step, dtype=int)
    print(f"   -> iterations per step {its.tolist()}, mean {its.mean():.1f}"
          + ("   !! hit iteration cap" if hit_cap else ""))

    time_L2_errors = compute_time_L2_from_history(
        error_history=error_history,
        dt=dt,
        time_window=time_error_window
    ) if compute_space_errors else {}

    return {
        "iterations": its,
        "avg_iterations": float(np.mean(its)),
        "max_iterations_step": int(np.max(its)),
        "hit_cap": hit_cap,
        "theta": float(theta),
        "beta": beta_paper,
        "beta_max": beta_max,
        "coupling_strength": coupling_strength,
        "Eyoung": Eyoung,
        "F_used": F,
        "k_order": k,
        "c_mandel": c_mandel,
        "B_derived": B,
        "nu_u_derived": nu_u,
        "time_L2_errors": time_L2_errors,
    }


# =============================================================================
# CSV utilities
# =============================================================================

ERROR_KEYS = [
    "p_L2_timeL2_abs", "p_L2_timeL2_rel",
    "u_L2_timeL2_abs", "u_L2_timeL2_rel",
    "sigma_Hdiv_timeL2_abs", "sigma_Hdiv_timeL2_rel",
    "q_Hdiv_timeL2_abs", "q_Hdiv_timeL2_rel",
]


def save_summary_csv(summary_rows, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    fieldnames = list(summary_rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {filepath}")


def load_summary_csv(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run with TABLE_ONLY = False first.")
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Table 4 (booktabs; requires \usepackage{booktabs} and \usepackage{multirow})
# =============================================================================

def sci_notation_latex(xval, digits=3):
    if xval == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(xval))))
    mantissa = xval / (10 ** exponent)
    return rf"{mantissa:.{digits}f} \times 10^{{{exponent}}}"


def gamma3_label(gamma3):
    exponent = int(round(np.log10(gamma3)))
    return rf"$10^{{{exponent}}}$"


def theta_header(theta):
    return f"{theta:g}"


def write_mandel_scaling_table(summary_rows, filename=TABLE_TEX):
    """
    Rows: gamma_3 (with coupling strength tau) x k; columns: theta = beta/beta_max.
    The per-row minimum (at the printed precision, ties included) is set in bold.
    """
    filepath = os.path.join(RESULTS_DIR, filename)

    row_map = {}
    for row in summary_rows:
        key = (round(float(row["gamma3"]), 12), int(row["k"]), round(float(row["theta"]), 12))
        row_map[key] = row

    def get(g, k, th):
        key = (round(g, 12), k, round(th, 12))
        if key not in row_map:
            raise KeyError(f"missing run gamma_3={g:g}, k={k}, theta={th:g}")
        return row_map[key]

    capped = [r for r in summary_rows if str(r["hit_cap"]) == "True"]
    if capped:
        print(f"!! WARNING: {len(capped)} run(s) hit the iteration cap; their averages are not converged counts.")

    n_theta = len(THETA_VALUES)
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{ccc" + "c" * n_theta + "}")
    lines.append(r"\toprule")
    lines.append(r"$\gamma_3$ & coupling strength $\tau$ & $k$ & "
                 rf"\multicolumn{{{n_theta}}}{{c}}{{$\theta = \beta/\beta_{{\max}}$}} \\")
    lines.append(rf"\cmidrule(lr){{4-{3 + n_theta}}}")
    lines.append(r"& & & " + " & ".join(theta_header(th) for th in THETA_VALUES) + r" \\")
    lines.append(r"\midrule")

    for ig, g in enumerate(GAMMA3_VALUES):
        tau = float(get(g, K_VALUES[0], THETA_VALUES[0])["coupling_strength"])
        nk = len(K_VALUES)
        for ik, k in enumerate(K_VALUES):
            vals = [float(get(g, k, th)["avg_iterations"]) for th in THETA_VALUES]
            printed = [f"{v:.1f}" for v in vals]
            vmin = min(float(s) for s in printed)
            cells = [rf"\textbf{{{s}}}" if float(s) == vmin else s for s in printed]
            if ik == 0:
                first = (rf"\multirow{{{nk}}}{{*}}{{{gamma3_label(g)}}} & "
                         rf"\multirow{{{nk}}}{{*}}{{${sci_notation_latex(tau)}$}}")
            else:
                first = "& "
            lines.append(f"{first} & {k} & " + " & ".join(cells) + r" \\")
        if ig < len(GAMMA3_VALUES) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Mandel's problem: average number of iterations per time step on $[0, 100]$ for varying "
        r"$\gamma_3 = E/E_{\mathrm{ref}} = F/F_{\mathrm{ref}}$ versus the destabilization fraction "
        r"$\theta = \beta/\beta_{\max}$. Here $\theta = 0$ is the plain split and $\theta = 1$ the theoretical "
        r"optimum $\beta_{\max}$; the per-row minimum is in bold.}"
    )
    lines.append(r"\label{tab:mandel_scaling_iterations}")
    lines.append(r"\end{table}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {filepath}")


def print_table_console(summary_rows):
    row_map = {(round(float(r["gamma3"]), 12), int(r["k"]), round(float(r["theta"]), 12)): r
               for r in summary_rows}
    print("\n gamma_3 |     tau     | k | " + " | ".join(f"th={th:<4g}" for th in THETA_VALUES))
    for g in GAMMA3_VALUES:
        for k in K_VALUES:
            rows = [row_map[(round(g, 12), k, round(th, 12))] for th in THETA_VALUES]
            tau = float(rows[0]["coupling_strength"])
            print(f" {g:7g} | {tau:11.4e} | {k} | "
                  + " | ".join(f"{float(r['avg_iterations']):7.1f}" for r in rows))


# =============================================================================
# Parameter sweep
# =============================================================================

def run_mandel_scaling_sweep():
    summary_rows = []
    for g in GAMMA3_VALUES:
        Eyoung = g * E_REF
        for k in K_VALUES:
            for theta in THETA_VALUES:
                out = mandel_splitting_U2BC(
                    theta=theta,
                    k=k,
                    Eyoung=Eyoung,
                    E_ref_for_F=E_REF,
                    F_ref_for_E=F_REF,
                    scale_F_with_E=True,
                    verbose=VERBOSE,
                    compute_space_errors=COMPUTE_ERRORS,
                    **SOLVER_KWARGS
                )
                row = {
                    "gamma3": g,
                    "k": k,
                    "theta": theta,
                    "beta": out["beta"],
                    "beta_max": out["beta_max"],
                    "Eyoung": out["Eyoung"],
                    "F_used": out["F_used"],
                    "coupling_strength": out["coupling_strength"],
                    "avg_iterations": out["avg_iterations"],
                    "max_iterations_step": out["max_iterations_step"],
                    "iterations_each_step": ",".join(map(str, out["iterations"].tolist())),
                    "hit_cap": out["hit_cap"],
                    "c_mandel": out["c_mandel"],
                    "B_derived": out["B_derived"],
                    "nu_u_derived": out["nu_u_derived"],
                }
                for key in ERROR_KEYS:
                    row[key] = out["time_L2_errors"].get(key, np.nan)
                row["ngsolve_version"] = ngsolve.__version__
                row["scipy_version"] = scipy.__version__
                summary_rows.append(row)

    save_summary_csv(summary_rows, SUMMARY_CSV)
    return summary_rows


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Mandel scaling study (Table 4): gamma_3 x k x theta on [0, 100] ===")
    print(f"NGSolve {ngsolve.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__}")

    if TABLE_ONLY:
        print("TABLE_ONLY = True: rebuilding the table from the stored csv.")
        summary_rows = load_summary_csv(SUMMARY_CSV)
    else:
        summary_rows = run_mandel_scaling_sweep()

    print_table_console(summary_rows)
    write_mandel_scaling_table(summary_rows, TABLE_TEX)


if __name__ == "__main__":
    main()