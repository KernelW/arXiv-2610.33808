"""Run one configurable displacement and density, with resumable stages."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

# These run before this entry point imports NumPy. threadpoolctl also handles
# environments where a notebook or shell already loaded a BLAS library.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-d-mev", type=float, help="Outer-layer displacement in meV")
    parser.add_argument("--density", type=float, help="Selected-flavor density in 1e11 cm^-2")
    parser.add_argument("--mesh-radius", type=int, help="M; sets dk=0.12/M")
    parser.add_argument("--temperature-mev", type=float, help="kBT for cRPA, parquet and pairing")
    parser.add_argument("--gate-nm", type=float)
    parser.add_argument("--dielectric", type=float)
    parser.add_argument("--config", type=Path, help="Advanced JSON configuration")
    parser.add_argument("--output", type=Path, help="Default: results/<parameter hash> in current directory")
    parser.add_argument("--stage", choices=("all", "screening", "vertex", "pairing"), default="all")
    parser.add_argument("--preview", action="store_true", help="Print parameters and memory estimate; do not solve")
    parser.add_argument("--smoke-test", action="store_true", help="M=2, kBT=5 meV full-coupling installation check")
    parser.add_argument("--low-temperature-mev", type=float, help="Optional second pairing solve with fixed vertex")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test and (args.config or any(v is not None for v in (
        args.delta_d_mev, args.density, args.mesh_radius, args.temperature_mev,
        args.gate_nm, args.dielectric, args.low_temperature_mev))):
        parser.error("--smoke-test uses its own documented settings; do not combine with physical overrides")
    if args.low_temperature_mev is not None:
        if args.low_temperature_mev <= 0 or args.stage not in ("all", "pairing"):
            parser.error("--low-temperature-mev must be positive and used with --stage all or pairing")
    from .config import default_config, point_config, smoke_config, preview, result_directory, make_system
    from threadpoolctl import threadpool_limits
    config = smoke_config() if args.smoke_test else (
        json.loads(args.config.read_text()) if args.config else default_config())
    model = config["model"]
    # Validate the convenient physical overrides using the same notebook API.
    checked = point_config(
        args.delta_d_mev if args.delta_d_mev is not None else 1000 * model["band_params"]["delta_d"],
        args.density if args.density is not None else model["carrier_density_cm2"] / 1e11,
        mesh_radius=args.mesh_radius if args.mesh_radius is not None else model["radius"],
    )
    if args.delta_d_mev is not None:
        model["band_params"]["delta_d"] = checked["model"]["band_params"]["delta_d"]
    if args.density is not None:
        model["carrier_density_cm2"] = checked["model"]["carrier_density_cm2"]
        model["target_particles_per_momentum"] = None
    if args.mesh_radius is not None:
        model.update(radius=checked["model"]["radius"], spacing=checked["model"]["spacing"])
    for value, key, factor in ((args.temperature_mev, "temperature_eV", .001),
                               (args.gate_nm, "gate_distance_nm", 1),
                               (args.dielectric, "dielectric_constant", 1)):
        if value is not None:
            if not __import__("math").isfinite(value) or value <= 0:
                parser.error(f"{key} must be positive and finite")
            model[key] = value * factor
    output = args.output or result_directory(config)
    with threadpool_limits(limits=1):
        print(json.dumps({"delta_d_meV": 1000 * model["band_params"]["delta_d"],
                          "density_1e11_cm2": model["carrier_density_cm2"] / 1e11,
                          "temperature_meV": 1000 * model["temperature_eV"],
                          "parquet_tolerance": config["solver"]["tolerance"],
                          **preview(config)}, indent=2))
        print("Output:", output, flush=True)
        if args.preview:
            return 0
        from .workflow import run
        result = run(config, output, stage=args.stage, progress=not args.quiet,
                     resume=not args.no_resume)
        if args.stage in ("all", "pairing"):
            _, sc, output = result
            if not sc.linearized.converged or (sc.nonlinear is not None and not sc.nonlinear.converged):
                print("Pairing did not converge. Inspect the saved residuals in summary.json.")
                return 2
            print(json.dumps({"lambda_Cooper": float(sc.linearized.leading_eigenvalue),
                              "classification": sc.linearized.classification,
                              "max_gap_meV": None if sc.nonlinear is None else 1000 * sc.nonlinear.max_gap}, indent=2))
            if not args.no_plots:
                from .reporting import save_pairing_figures
                save_pairing_figures(make_system(config), sc, output / "03_superconductivity")
            if args.low_temperature_mev is not None:
                from .fixed_vertex_temperature import solve_fixed_vertex_temperature
                system, low, summary, destination = solve_fixed_vertex_temperature(
                    output, args.low_temperature_mev / 1000, progress=not args.quiet,
                    nonlinear_kwargs=config["pairing"].get("nonlinear_kwargs"))
                print(json.dumps({k: summary[k] for k in ("lambda_max", "status", "max_gap_meV")}, indent=2))
                if not args.no_plots:
                    from .reporting import save_pairing_figures
                    save_pairing_figures(system, low, destination)
        elif args.stage == "vertex":
            result[3].flat._mmap.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
