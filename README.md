# ABCA parquet superconductivity

Code for [arXiv:2610.33808](https://arxiv.org/abs/2610.33808).

A self-contained calculation at **one tunable displacement ΔD and carrier density n** in rhombohedral tetralayer graphene. It starts from the free eight-band Hamiltonian, computes constrained RPA screening, solves the coupled particle-hole parquet equations, and evaluates the linearized and nonlinear odd intravalley superconducting gap.

No original project folders, Hartree–Fock files, external notebooks, or saved vertices are needed. The numerical dependency modules are bundled here. Density scans and self-energy calculations are outside this package's scope.

## Install

Use Python 3.11 or newer. Clone the repository, then install it:

```bash
git clone https://github.com/KernelW/arXiv-2610.33808.git
cd arXiv-2610.33808
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook,test]"
python -m ipykernel install --user --name abca-sc --display-name "Python (ABCA SC)"
```

On Windows, activate with `.venv\Scripts\activate` instead. The kernel installation command is optional for command-line use. Core dependencies are NumPy, SciPy, Matplotlib, and threadpoolctl; Jupyter and pytest are optional extras. No GPU is required.

## Start with an installation check

```bash
python -m pytest -q
abca-sc --smoke-test --output results/smoke
```

The smoke case runs the complete free-band → cRPA → parquet at g=1 → pairing pipeline on M=2 at kBT=5 meV. It tests the software; its coarse mesh and high temperature are **not** representative superconducting results.

## Run a physical point

Default: ΔD=110 meV, n=6.375 × 10¹¹ cm⁻², gate distance 25 nm, dielectric constant 15, and kBT=0.1 meV.

```bash
abca-sc --delta-d-mev 110 --density 6.375 --preview
abca-sc --delta-d-mev 110 --density 6.375 --low-temperature-mev 0.01
```

`python -m parquet_superconductivity` is equivalent to `abca-sc`. Change the two physical parameters directly, for example:

```bash
abca-sc --delta-d-mev 125 --density 7 --preview
abca-sc --delta-d-mev 125 --density 7
```

The second example specifies a different point; it does not guarantee superconductivity or parquet convergence there. `--low-temperature-mev` optionally reuses the solved vertex while changing the pairing temperature, with the active-band population fixed. Omit it for a calculation at the vertex temperature only.

For Jupyter, run `jupyter lab fixed_point_superconductivity.ipynb`, select **Python (ABCA SC)**, and edit the parameter cell. `DELTA_D_MEV` is in meV and `DENSITY_1E11_CM2` is in units of 10¹¹ cm⁻². The notebook and CLI use the same numerical workflow.

## Memory, convergence, and resuming

The default M=10 mesh has 331 momentum points and 1261 transfer sectors. Solver arrays require approximately 12 GB, plus Python/BLAS overhead; provide additional free memory. A full complex vertex alone is approximately 266 MB. Check `--preview` before starting; runtime depends strongly on the mesh and proximity to a particle-hole pole. Use the smoke test for a quick installation check. Do not run the full notebook and CLI simultaneously in the same output directory.

Outputs default to `results/<parameter-and-config-hash>` in the current directory. A rerun with identical controls resumes the last converged parquet checkpoint. Changed configurations use different default directories; an explicit `--output` cannot overwrite a run with different settings. Moving or editing code changes its provenance hashes; start a new output directory after changing source code.

Stages can also be run separately with identical parameters/configuration:

```bash
abca-sc --stage screening
abca-sc --stage vertex
abca-sc --stage pairing --low-temperature-mev 0.01
```

Advanced controls are in `configs/reference_D110_n6p375.json`:

```bash
abca-sc --config configs/reference_D110_n6p375.json --preview
abca-sc --config configs/reference_D110_n6p375.json
```

The default parquet acceptance tolerance is **5 × 10⁻³**, matching the reference notebook's controls. The saved reference actually attained a residual of approximately **8.13 × 10⁻⁵**. These are equation residuals, not energy error bars. For quantitative conclusions, check the actual parquet and BSE residuals and tighten tolerance/refine the mesh. The default adaptive continuation does not relax tolerance or feed failed iterates into pairing. If it cannot reach g=1, it stops and keeps its checkpoint for diagnosis. 

A converged normal solution is a legitimate result when the Cooper eigenvalue is subcritical. Failed gap/number solves are labeled explicitly. Winding is assigned only when a nonzero gap and a resolved closed contour permit it.

## Outputs

| Location under a run | Contents |
| --- | --- |
| `config.json`, `provenance.json` | Full controls and bundled source hashes |
| `01_screening/` | Polarization channels and cRPA interaction |
| `02_vertex/checkpoints/` | Last converged parquet state for resuming |
| `02_vertex/continuation.json` | Coupling history and solver diagnostics |
| `02_vertex/full_gamma.json`, `gamma_full_*.npy` | Explicit full vertex, checksum, residuals, symmetries |
| `03_superconductivity/` | Cooper matrix, leading eigenmode, nonlinear gap, summary, figures |
| `04_fixed_vertex_temperature/` | Optional second-temperature gap and phase winding |

Plots are exported as PNG and vector PDF. The Re/Im plots use the solver's band gauge, with an overall phase fixed for display. Phase winding uses the separately specified A1-positive band gauge; see [methods](docs/METHODS.md).

## Source layout

- `parquet_superconductivity/`: cRPA, screened-source parquet, pairing, workflow and figures.
- `free_fermion_gamma/`: free eight-band model, mesh, polarization and interaction utilities.
- `static_parquet_memory/`: all-transfer problem construction and Newton–Krylov solver.
- `tests/`: screening, crossing, normalization, checkpoint, gap/number, and workflow checks.
- `fixed_point_superconductivity.ipynb`: single-point notebook.
- `docs/VALIDATION.md`: checks performed when preparing this snapshot.

## Repository

Source: [KernelW/arXiv-2610.33808](https://github.com/KernelW/arXiv-2610.33808).
Generated results, caches, environments and large arrays are excluded by `.gitignore`.
