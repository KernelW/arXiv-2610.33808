# Validation of the standalone snapshot

Prepared 2026-11-15. See `validated_environment.json` for exact local package versions and platform, and `reference_results.json` for the measured reference values.

## Completed checks

- Built and installed a standard wheel in a temporary environment, using the local numerical libraries. Ran from a directory outside both the source package and the production project with Python isolated mode (`-I`). All three numerical packages imported from the installed wheel; there was no production-project import fallback.
- **25 pytest checks passed**: finite-Q cRPA band-pair sum, excluded active pair, sequential screening identity, symmetries and cache validation; direct/exchange source, crossed-transfer residual and BSE identities; checkpoint/source validation and failed/subtarget rejection; Cooper routing, factor 1/2, quadrature weight, asymmetric dispersion, odd projection, number constraint, analytic separable-gap solution; staged continuation and resume; CLI parameter units and no-write preview; gauge transformation and known winding/node checks.
- CLI **full-coupling g=1 smoke run**: M=2, 19 momenta, 61 transfer blocks, kBT=5 meV, requested parquet residual tolerance 1e-7. All numerical stages completed. The Cooper eigenvalue was −0.00457341293 and the converged state was normal (zero gap), as appropriate for this software check.
- Executed **all six code cells of the notebook** in an actual temporary Jupyter kernel, changing only `SMOKE_TEST` to `True`. The delivered notebook is clean: no saved outputs or execution counts.
- Default M=10 preview: 331 momenta, 1261 transfer blocks, estimated solver arrays 11.815 GB; one full vertex 0.265632 GB.
- Recomputed **M=10 cRPA from scratch** at ΔD=110 meV, n=6.375 × 10¹¹ cm⁻²; it matched the saved constrained-screening reference.
- Loaded a temporary copy of the existing converged **full Γ**, extracted the Cooper matrix with the bundled code, then ran the packaged pairing workflow at kBT=0.1 meV and the optional fixed-vertex workflow at 0.01 meV. Both nonlinear gap/number solves converged and reproduced the reference.
- Exported PNG/PDF gap and phase-winding figures for both temperatures. The A1-positive-reference winding was **−3** at both temperatures; the low-temperature interpolated minimum gap on the normal Fermi contour was 0.216788 meV.

| Reference quantity | Recomputed value |
| --- | ---: |
| Leading Cooper eigenvalue at kBT=0.1 meV | 1.0436268208205803 |
| Maximum gap at kBT=0.1 meV | 0.288557008968008 meV |
| Gap residual at kBT=0.1 meV | 9.24794 × 10⁻⁹ |
| Leading Cooper eigenvalue at kBT=0.01 meV, fixed vertex | 1.039777844266199 |
| Maximum gap at kBT=0.01 meV, fixed vertex | 0.313776392191014 meV |
| Gap residual at kBT=0.01 meV | 9.45002 × 10⁻⁹ |
| Filling residual at kBT=0.01 meV | 2.22711 × 10⁻¹³ |

## Scope of validation

The expensive M=10 parquet continuation was **not rerun from scratch** while assembling this package. Its existing full vertex was used read-only for the M=10 reference checks. The full three-stage pipeline was run from scratch on the small smoke mesh. The bundled all-transfer numerical solver is copied from the working source; its equations are exercised by the automated tests. No large saved reference arrays are included in the repository, and no precomputed result is required to run the package.

These checks establish software functionality and consistency with the existing reference. They do not establish momentum-mesh convergence, convergence at every tunable ΔD/n, or exact physical accuracy. The default reference controls accept a parquet residual below 5e-3; the reference used here had an actual residual 8.13129e-5 and full-vertex BSE residual 5.50324e-4.

The GitHub Actions workflow is provided for future runs; it has not been executed on GitHub as part of this preparation.

## Packaging changes

`upstream_source_hashes.json` records the original numerical-source hashes. The standalone snapshot bundles the required free-band and all-transfer modules, uses minimal package initializers, removes external path discovery, extracts the progress formatter, and hashes its own installed sources for provenance. Default controls match the reference notebook. Workflow changes add saved-gap provenance for phase plots and close full-vertex memory maps. The single-point CLI, clean notebook, plotting exporter, documentation and interface/phase tests are supplied here. The original project files were left untouched.
