# splitting_4_arbitrary_mfem_biot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Reproducibility repository for the paper

> F. Bertrand, J. W. Both, T. Dağlı,
> *Splitting-strategies for arbitrary-order fully mixed finite element discretizations of the Biot equations*,
> submitted to Computers and Mathematics with Applications (2026). Preprint: [arXiv:2603.18241](https://arxiv.org/abs/2603.18241).

The repository contains the scripts and result files for all numerical experiments of Section 6:
a five-field, weakly symmetric fully mixed finite element discretization of the Biot equations
(Raviart–Thomas stress rows and Darcy flux, discontinuous displacement and pressure, continuous rotation)
of arbitrary order k, solved with the fixed-stress splitting with destabilization parameter β.

## Contents: figures and tables of the paper

| Paper | Experiment | Script |
|---|---|---|
| Figure 1 | 2D manufactured solution, spatial convergence (Section 6.1) | `2D/manufactured_solution/discretization_conv_rates/main.py` |
| Tables 1–2 | 2D manufactured solution, iteration counts for varying γ₁ and γ₂ (Section 6.1) | `2D/manufactured_solution/parameter_scaling_iterative_method_test/main.py` |
| Figure 3 | Mandel's problem, time profiles of the dimensionless solution (Section 6.2) | `2D/mandel/time_profiles/main.py` |
| Table 4 | Mandel's problem, iteration counts for varying γ₃ and θ (Section 6.2) | `2D/mandel/scaling_iterative_test/main.py` |
| Figure 4 | 3D manufactured solution, spatial convergence (Section 6.3) | `3D/manufactured_solution/discretization_conv_rates/main.py` |
| Tables 5–6 | 3D manufactured solution, iteration counts for varying γ₁ and γ₂ (Section 6.3) | `3D/manufactured_solution/parameter_scaling_iterative_method_test/main.py` |
| Tables 7–8 | 3D footing problem with the parameters of Both et al. (arXiv:1907.03134), iteration counts for varying E and θ (Section 6.4) | `3D/footing_benchmark/main.py` |

Each folder also contains the result files (csv, LaTeX tables, figures) produced by the script,
so that every entry of the paper can be checked without rerunning the computations.

## Notation used in the code

The scripts follow the notation of the paper:

- coupling modulus M = d α² / (2μ + d λ) and coupling strength τ = M / c₀;
- destabilization β ≥ 0, entering the flow block as (c̃₀ − β) M_p; the maximal destabilization of Lemma 3 is β_max = M/2;
- destabilization fraction θ = β / β_max, so θ = 0 is the plain fixed-stress split (β_FS = 0) and θ = 1 is β_max;
- stopping criterion (47): relative update of the flow variables (w, p) and of the mechanics variables (σ, ξ, u), tolerance 10⁻⁶;
- reported iteration counts are averages over the time steps.

## Software environment

All scripts use NGSolve for the finite element assembly and SciPy sparse matrices for the splitting iteration,
plus the standard Python modules `csv`, `os` and `pickle`. The computations were performed on two machines.

### Laptop — Sections 6.1, 6.2 and 6.4 (Figures 1, 3; Tables 1–4, 7–8)

- Ubuntu 22.04.5 LTS
- Python 3.10.12
- NGSolve 6.2.2505-144-gac1b9a527
- NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.6
- PETSc 3.23.6 with petsc4py 3.23.6 (Section 6.4 only)
- MUMPS 5.8.1 (via PETSc), ordering: METIS 5.1.0 (nested dissection)

### Server — Section 6.3 (Figure 4; Tables 5–6)

- Debian GNU/Linux 13 (trixie)
- Python 3.11.14
- NGSolve 6.2.2506-93-g4fb22b00e
- NumPy 2.3.3, SciPy 1.16.2, Matplotlib 3.10.7
- PETSc 3.24.0 with petsc4py 3.24.0
- MUMPS 5.8.1 (via PETSc), ordering: METIS 5.1.0 (nested dissection)

The two-dimensional computations (Sections 6.1–6.2) use SciPy's SuperLU and do not require PETSc.
The three-dimensional computations (Sections 6.3–6.4) use PETSc/MUMPS through petsc4py
(`SOLVER = "mumps"`, exact direct solve: KSP `preonly` + PC `lu`); `SOLVER = "splu"` switches to SuperLU,
which is feasible only for small meshes. Each 3D result file records the NGSolve, NumPy, SciPy and PETSc
versions and the MUMPS ordering that was used.

**Note on Figure 3.** The Mandel time profiles are plotted from stored pickle files
(`mandel_profiles_*.pkl`) produced with NGSolve 6.2.2505. These pickles contain NGSolve objects
and cannot be loaded with other NGSolve versions (e.g. 6.2.2608 fails). With a different version,
set `force_rerun=True` to recompute them (full time interval [0, 50000]; long run).

## Running the experiments

Each script is run from its own folder, e.g.

```bash
cd 3D/footing_benchmark
python main.py
```

and writes its results into the subfolders `results/` and `figures/` of the script's folder
(3D scripts: `results_3d/` and `figures_3d/`).
Switches are set at the top of each script:

- `CHEAP_TEST = True` (3D scripts): quick smoke test on a coarse mesh; the numbers are **not** those of the paper.
- `TABLES_ONLY = True` (Table 4, Tables 5–8): rebuild the LaTeX tables from the stored csv files without recomputing.
- `force_rerun` (Figure 3): `False` plots from the stored pickles, `True` recomputes the time profiles.
- `MESH_CHECK` (Tables 5–6): optional additional runs of the most strongly coupled cases on several meshes; not part of the paper, off by default.
- `SOLVER` (3D scripts): `"mumps"` (setting of the paper) or `"splu"`.

Computational cost: the 2D scripts and Figure 3 (from the pickles) run on a laptop in minutes to hours.
The 3D scripts are considerably more expensive; the footing problem (Tables 7–8, k = 2 on the N = 8 mesh)
needs about 16 GB of memory and several hours on a laptop, dominated by the slowly converging runs
with θ = 1 at strong coupling.

## License

This code is released under the MIT license; see `LICENSE`.

## How to cite

If you use this code or results obtained with it, **please cite the paper**:

```bibtex
@misc{BertrandBothDagli2026,
  author        = {Bertrand, Fleurianne and Both, Jakub Wiktor and Da{\u{g}}l{\i}, Tugay},
  title         = {Splitting-strategies for arbitrary-order fully mixed finite element
                   discretizations of the {B}iot equations},
  year          = {2026},
  eprint        = {2603.18241},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  note          = {Submitted to Computers and Mathematics with Applications}
}
```

When referring to the code itself, please additionally cite the archived version of this repository:

```bibtex
@software{Dagli2026_splitting_biot_code,
  author    = {Da{\u{g}}l{\i}, Tugay},
  title     = {Code for ``Splitting-strategies for arbitrary-order fully mixed finite element
               discretizations of the {B}iot equations''},
  year      = {2026},
  publisher = {Zenodo},
  version   = {vX.Y.Z},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```
