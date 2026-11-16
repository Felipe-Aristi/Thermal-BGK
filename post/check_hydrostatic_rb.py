#!/usr/bin/env python3
"""Analyze the unperturbed hydrostatic-conduction test of Thermal-BGK.

Usage (from repository root):
  python post/check_hydrostatic_rb.py \
    --data results/rb/NX_200_NY_100/Ra_500/vti \
    --m -0.9806 --delta-t-ratio 0.6

The diagnostics compare exported hydrodynamic moments with the continuum
polytropic equilibrium. They do not by themselves prove kinetic wall accuracy.
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CS2 = 1.0 / (1.19697977039307435897238846385 ** 2)
GHOST_LAYERS = 3  # D2V37 maxAbsC: exclude these near-wall fluid rows from bulk errors


def load_vti(path: Path):
    root = ET.parse(path).getroot()
    image = root.find("ImageData")
    pointdata = root.find(".//PointData")
    if image is None or pointdata is None:
        raise ValueError(f"Invalid VTI structure: {path}")

    extent = list(map(int, image.attrib["WholeExtent"].split()))
    nx, ny = extent[1] - extent[0] + 1, extent[3] - extent[2] + 1
    values = {}
    for entry in pointdata.findall("DataArray"):
        name = entry.attrib.get("Name")
        if name in ("Density", "Temperature", "Pressure", "Velocity"):
            values[name] = np.fromstring(entry.text or "", sep=" ")

    required = ("Density", "Temperature", "Velocity")
    for name in required:
        if name not in values:
            raise ValueError(f"Missing {name} in {path}")

    if len(values["Density"]) != nx * ny or len(values["Temperature"]) != nx * ny:
        raise ValueError(f"Incorrect scalar array lengths in {path}")
    if len(values["Velocity"]) != nx * ny * 3:
        raise ValueError(f"Incorrect velocity array length in {path}")

    rho = values["Density"].reshape(ny, nx)
    temp = values["Temperature"].reshape(ny, nx)
    velocity = values["Velocity"].reshape(ny, nx, 3)[..., :2]
    pressure = values.get("Pressure", rho.ravel() * temp.ravel()).reshape(ny, nx)

    if not all(np.isfinite(v).all() for v in (rho, temp, velocity, pressure)):
        raise ValueError(f"NaN or infinite value in {path}")
    if (rho <= 0).any() or (temp <= 0).any():
        raise ValueError(f"Non-positive density or temperature in {path}")
    return nx, ny, rho, temp, pressure, velocity


def snapshot_step(path: Path):
    match = re.search(r"_(\d+)\.vti$", path.name)
    if match is None:
        raise ValueError(f"Could not determine timestep from filename {path.name}")
    return int(match.group(1))


def save_profile(path, eta, analytic, initial, final, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(analytic, eta, "-", label="Analytical hydrostatic profile")
    ax.plot(initial, eta, ":", label="First saved snapshot")
    ax.plot(final, eta, "--", label="Last saved snapshot")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_history(path, steps, values, label, use_log=False):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    v = np.asarray(values, dtype=float)
    if use_log:
        ax.semilogy(steps, np.maximum(v, 1.0e-30), "o-", markersize=3)
    else:
        ax.plot(steps, v, "o-", markersize=3)
    ax.set(xlabel="Saved iteration (step 1 is the initial pre-evolution output)", ylabel=label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing rb_*.vti")
    parser.add_argument("--m", type=float, required=True, help="Polytropic index used in constants.cuh")
    parser.add_argument("--delta-t-ratio", type=float, default=0.6,
                        help="DeltaT / referenceTemperature (default 0.6)")
    parser.add_argument("--reference-temperature", type=float, default=CS2,
                        help="Middle-cell reference temperature; default D2V37::cs2")
    parser.add_argument("--reference-density", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = sorted(args.data.glob("rb_*.vti"), key=snapshot_step)
    if not paths:
        parser.error(f"No rb_*.vti in {args.data}")

    output = args.output or (args.data.parent / "hydrostatic_validation")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    first = last = None
    expected_shape = None

    for path in paths:
        step = snapshot_step(path)
        nx, ny, rho, temp, pressure, velocity = load_vti(path)
        if expected_shape is not None and (nx, ny) != expected_shape:
            raise ValueError(f"Mesh dimensions changed in {path}")
        expected_shape = nx, ny

        # Wall at eta=0, 1; node centers at eta=(j+1/2)/NYPhysical.
        eta = (np.arange(ny, dtype=float) + 0.5) / ny
        dtemp = args.delta_t_ratio * args.reference_temperature
        temperature_analytic = args.reference_temperature + dtemp * (0.5 - eta)
        density_analytic = args.reference_density * (
            temperature_analytic / args.reference_temperature
        ) ** args.m
        pressure_analytic = density_analytic * temperature_analytic
        gravity = (args.m + 1.0) * dtemp / ny

        if ny <= 2 * GHOST_LAYERS + 2:
            raise ValueError("Too few vertical nodes for bulk diagnostics")
        bulk = slice(GHOST_LAYERS, ny - GHOST_LAYERS)
        pbar = pressure.mean(axis=1)
        tbar = temp.mean(axis=1)
        rhobar = rho.mean(axis=1)
        speed = np.linalg.norm(velocity, axis=2)
        vel_fluct = velocity - velocity.mean(axis=1, keepdims=True)

        # Pressure gradient and gravity should nearly balance in the stationary bulk.
        j = np.arange(GHOST_LAYERS + 1, ny - GHOST_LAYERS - 1)

        dp_dy = 0.5 * (pbar[j + 1] - pbar[j - 1])

        hydro_res = dp_dy + gravity * rhobar[j]
        hydro_scale = max(abs(gravity) * args.reference_density, 1e-30)
        kinetic_energy = 0.5 * np.mean(rho * speed**2)
        fluctuating_energy = 0.5 * np.mean(rho * np.sum(vel_fluct**2, axis=2))
        profile_temperature_error = np.max(np.abs(tbar[bulk] - temperature_analytic[bulk])) / dtemp
        profile_density_error = np.max(np.abs(rhobar[bulk] - density_analytic[bulk])) / args.reference_density
        profile_pressure_error = np.max(np.abs(pbar[bulk] - pressure_analytic[bulk])) / (
            args.reference_density * args.reference_temperature
        )
        wall_error_bottom = (tbar[0] - temperature_analytic[0]) / dtemp
        wall_error_top = (tbar[-1] - temperature_analytic[-1]) / dtemp
        max_speed_bulk = float(np.max(speed[bulk]))
        max_speed_wall = float(max(np.max(speed[:GHOST_LAYERS]), np.max(speed[-GHOST_LAYERS:])))
        x_variation_t = float(np.max(temp.max(axis=1) - temp.min(axis=1))) / dtemp
        total_mass = float(np.sum(rho))

        rows.append({
            "step": step, "mass": total_mass,
            "mass_drift_from_first": 0.0, "mass_error_vs_analytic":
                (total_mass - nx * np.sum(density_analytic)) / (nx * np.sum(density_analytic)),
            "max_speed": float(np.max(speed)), "max_speed_bulk": max_speed_bulk,
            "max_speed_wall_3_rows": max_speed_wall,
            "kinetic_energy_total": float(kinetic_energy),
            "kinetic_energy_x_fluct": float(fluctuating_energy),
            "temperature_bulk_Linf_over_deltaT": float(profile_temperature_error),
            "density_bulk_Linf_over_rho_m": float(profile_density_error),
            "pressure_bulk_Linf_over_p_m": float(profile_pressure_error),
            "pressure_hydrostatic_residual_over_rho_m_g": float(np.max(np.abs(hydro_res)) / hydro_scale),
            "lower_first_cell_T_error_over_deltaT": float(wall_error_bottom),
            "upper_first_cell_T_error_over_deltaT": float(wall_error_top),
            "max_horizontal_T_span_over_deltaT": x_variation_t,
        })
        fields = (eta.copy(), temperature_analytic.copy(), density_analytic.copy(),
                  pressure_analytic.copy(), tbar.copy(), rhobar.copy(), pbar.copy())
        if first is None:
            first = fields
        last = fields

    initial_mass = rows[0]["mass"]
    for row in rows:
        row["mass_drift_from_first"] = (row["mass"] - initial_mass) / initial_mass

    csv_path = output / "hydrostatic_history.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eta, ta, ra, pa, ti, ri, pi = first
    _, _, _, _, tf, rf, pf = last
    save_profile(output / "temperature_profile.png", eta, ta, ti, tf, "Hydrodynamic temperature", "Height / H")
    save_profile(output / "density_profile.png", eta, ra, ri, rf, "Density", "Height / H")
    save_profile(output / "pressure_profile.png", eta, pa, pi, pf, "Pressure", "Height / H")
    steps = [r["step"] for r in rows]
    save_history(output / "max_speed.png", steps, [r["max_speed"] for r in rows], "Maximum hydrodynamic speed", True)
    save_history(output / "temperature_bulk_error.png", steps,
                 [r["temperature_bulk_Linf_over_deltaT"] for r in rows], "Bulk max |T-T0| / DeltaT", True)
    save_history(output / "mass_drift.png", steps,
                 [r["mass_drift_from_first"] for r in rows], "(M(t)-M(first))/M(first)")
    save_history(output / "wall_temperature_error.png", steps,
                 [r["lower_first_cell_T_error_over_deltaT"] for r in rows],
                 "Lower first-fluid-node T deviation / DeltaT")

    print(f"Analyzed {len(rows)} snapshots from {paths[0].name} to {paths[-1].name}")
    print(f"NX, NY: {expected_shape}; m={args.m}; DeltaT/T_m={args.delta_t_ratio}")
    print("The FIRST VTI is reconstructed after one pull of post-collision initial populations.")
    print("Therefore, it should not be interpreted as an exact t=0 raw-population snapshot.")
    print("\nLast snapshot:")
    for key, val in rows[-1].items():
        if key != "step":
            print(f"  {key:52s} {val:+.9e}")
    print(f"\nHistory CSV and diagnostic plots: {output}")
    print("No automatic PASS/FAIL: assess stationarity, wall error, and mesh refinement.")


if __name__ == "__main__":
    main()
