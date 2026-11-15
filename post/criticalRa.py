
#!/usr/bin/env python3
"""
Estimate the linear growth or decay rate of Rayleigh-Benard
perturbations from TBGK ASCII VTI files.
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# D2V37 reference temperature
# ============================================================

CS2 = 1.0 / (1.19697977039307435897238846385 ** 2)


# ============================================================
# Read VTI file and calculate perturbation kinetic energy
# ============================================================

def read_energy(path: Path):

    root = ET.parse(path).getroot()

    image = root.find("ImageData")

    if image is None:
        raise ValueError(f"Missing ImageData in {path}")

    extent = [int(v) for v in image.attrib["WholeExtent"].split()]

    nx = extent[1] - extent[0] + 1
    ny = extent[3] - extent[2] + 1

    n = nx * ny

    pointdata = root.find(".//PointData")

    if pointdata is None:
        raise ValueError(f"Missing PointData in {path}")

    values = {}

    for array in pointdata.findall("DataArray"):

        if array.attrib.get("Name") in ("Density", "Velocity", "Temperature"):

            values[array.attrib["Name"]] = np.fromstring(
                array.text or "", sep=" "
            )

    density = values["Density"].reshape(ny, nx)

    velocity = (
        values["Velocity"]
        .reshape(n, 3)[:, :2]
        .reshape(ny, nx, 2)
    )

    temperature = values["Temperature"].reshape(ny, nx)

    # ========================================================
    # Validate data
    # ========================================================

    if not (
        np.all(np.isfinite(density))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(temperature))
    ):
        raise ValueError(f"NaN/Inf in {path}")

    if np.any(density <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError(f"Non-positive density/temperature in {path}")

    # ========================================================
    # Remove horizontal mean velocity at each height
    #
    # u'(x,y) = u(x,y) - <u>_x(y)
    # ========================================================

    fluctuating = velocity - velocity.mean(axis=1, keepdims=True)

    # ========================================================
    # Perturbation kinetic energy
    #
    # E = 0.5 * <rho * (ux'^2 + uy'^2)>
    # ========================================================

    energy = 0.5 * np.mean(
        density * np.sum(fluctuating**2, axis=2)
    )

    return (
        float(energy),
        float(density.sum()),
        float(temperature.min()),
        float(temperature.max()),
        nx,
        ny
    )


# ============================================================
# Reference thermal diffusivity
#
# Ra = g * beta * DeltaT_SA * H^3 / (nu * alpha)
#
# alpha = nu / Pr
# ============================================================

def reference_diffusivity(ra, height, polytropic_index, deltaT_ratio, pr):

    t0 = CS2
    dt = deltaT_ratio * t0
    cp = 2.0

    g = (polytropic_index + 1.0) * dt / height

    dt_superadiabatic = dt - g * height / cp

    if dt_superadiabatic <= 0:
        raise ValueError("The chosen background is not superadiabatic.")

    nu = np.sqrt(
        pr * g * dt_superadiabatic * height**3 / (t0 * ra)
    )

    return nu / pr


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("results/rb/vti"),
        help="VTI directory for one mesh and Rayleigh number "
             "(e.g. results/rb/NX_200_NY_100/Ra_1300/vti); "
             "defaults to the legacy results/rb/vti directory"
    )

    parser.add_argument("--fit-start", type=float, required=True)
    parser.add_argument("--fit-end", type=float, required=True)

    parser.add_argument("--m", type=float, default=0.98)
    parser.add_argument("--deltaT-ratio", type=float, default=0.6)
    parser.add_argument("--pr", type=float, default=1.0)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rb/critical")
    )

    args = parser.parse_args()

    if args.fit_end <= args.fit_start:
        parser.error("--fit-end must be greater than --fit-start")

    # ========================================================
    # Find VTI snapshots
    # ========================================================

    paths = sorted(args.data.glob("rb_*.vti"))

    if not paths:
        parser.error(f"No rb_*.vti files in {args.data}")

    tagged_pattern = re.compile(
        r"rb_Ra_(?P<ra>[^_]+)_NX_(?P<nx>\d+)_NY_(?P<ny>\d+)_(?P<step>\d+)\.vti"
    )
    legacy_pattern = re.compile(r"rb_(?P<step>\d+)\.vti")

    tagged_paths = [
        (path, tagged_pattern.fullmatch(path.name))
        for path in paths
    ]
    recognized_tagged_paths = [
        (path, match)
        for path, match in tagged_paths
        if match is not None
    ]

    if recognized_tagged_paths:
        paths = [
            (path, match)
            for path, match in recognized_tagged_paths
            if float(match.group("ra")) == args.ra
        ]
    else:
        paths = [
            (path, legacy_pattern.fullmatch(path.name))
            for path in sorted(args.data.glob("rb_*.vti"))
            if legacy_pattern.fullmatch(path.name) is not None
        ]

    if not paths:
        parser.error(f"No VTI snapshots found for Ra={args.ra:g} in {args.data}")

    steps = []
    energies = []
    masses = []
    tmins = []
    tmaxs = []

    nx0 = None
    ny0 = None

    # ========================================================
    # Calculate energy history
    # ========================================================

    for path, match in paths:
        step = int(match.group("step"))

        energy, mass, tmin, tmax, nx, ny = read_energy(path)

        if nx0 is not None and nx != nx0:
            raise ValueError("Inconsistent width across VTI files")

        if ny0 is not None and ny != ny0:
            raise ValueError("Inconsistent height across VTI files")

        nx0 = nx
        ny0 = ny

        steps.append(step)
        energies.append(energy)
        masses.append(mass)
        tmins.append(tmin)
        tmaxs.append(tmax)

    steps = np.asarray(steps, dtype=np.float64)
    energies = np.asarray(energies)
    masses = np.asarray(masses)
    tmins = np.asarray(tmins)
    tmaxs = np.asarray(tmaxs)

    # ========================================================
    # Dimensionless diffusive time
    #
    # t* = alpha0 * t / H^2
    #
    # In our current main.cu, the first VTI (step 1)
    # corresponds to the initialized state at t = 0.
    # ========================================================

    alpha = reference_diffusivity(
        args.ra,
        ny0,
        args.m,
        args.deltaT_ratio,
        args.pr
    )

    tstar = alpha * (steps - 1.0) / ny0**2

    # ========================================================
    # Select linear fitting interval
    # ========================================================

    mask = (
        (tstar >= args.fit_start)
        & (tstar <= args.fit_end)
        & (energies > 0.0)
    )

    if np.count_nonzero(mask) < 4:
        parser.error(
            f"Only {np.count_nonzero(mask)} positive-energy samples "
            "in fit window. Change fit times or run longer."
        )

    # ========================================================
    # Fit ln(E) = gamma * t* + C
    # ========================================================

    gamma, intercept = np.polyfit(
        tstar[mask],
        np.log(energies[mask]),
        1
    )

    yfit = gamma * tstar[mask] + intercept

    residual = np.log(energies[mask]) - yfit
    rms = np.sqrt(np.mean(residual**2))

    # ========================================================
    # Output directory
    # ========================================================

    args.output.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Save energy and mass history
    # ========================================================

    out_csv = args.output / f"Ra_{args.ra:g}_history.csv"

    with out_csv.open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "step",
            "tstar",
            "kinetic_energy_fluct",
            "total_mass",
            "Tmin",
            "Tmax"
        ])

        writer.writerows(
            zip(steps.astype(np.int64), tstar, energies, masses, tmins, tmaxs)
        )

    # ========================================================
    # Plot logarithmic energy growth
    # ========================================================

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    valid = energies > 0.0

    ax.plot(
        tstar[valid],
        np.log(energies[valid]),
        label="Numerical data"
    )

    ax.plot(
        tstar[mask],
        yfit,
        "--",
        label=fr"Linear fit: $\gamma^*={gamma:.4g}$"
    )

    ax.axvspan(
        args.fit_start,
        args.fit_end,
        alpha=0.10,
        label="Fitting window"
    )

    ax.set(
        xlabel=r"Diffusive time $t^*=\alpha_0 t/H^2$",
        ylabel=r"$\ln E_k'$",
        title=fr"TBGK: $Ra={args.ra:g}$"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_png = args.output / f"Ra_{args.ra:g}_growth.png"

    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # ========================================================
    # Save fitted growth rate
    # ========================================================

    fit_csv = args.output / f"Ra_{args.ra:g}_fit.csv"

    with fit_csv.open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Ra",
            "gamma",
            "fit_start_tstar",
            "fit_end_tstar",
            "rms_residual",
            "n_fit",
            "mass_rel_change"
        ])

        writer.writerow([
            args.ra,
            gamma,
            args.fit_start,
            args.fit_end,
            rms,
            int(np.count_nonzero(mask)),
            (masses[-1] - masses[0]) / masses[0]
        ])

    # ========================================================
    # Diagnostics
    # ========================================================

    print(f"Ra                       : {args.ra:g}")
    print(f"Grid height H            : {ny0}")
    print(f"Reference alpha0         : {alpha:.10g}")
    print(f"Time range t*            : {tstar.min():.5g} to {tstar.max():.5g}")
    print(f"Fitting window t*        : {args.fit_start:g} to {args.fit_end:g}")
    print(f"Growth rate gamma*       : {gamma:+.9g}")
    print(f"Fit RMS residual         : {rms:.5g}")
    print(f"Relative mass change     : {(masses[-1]-masses[0])/masses[0]:+.6g}")
    print(f"Energy history CSV       : {out_csv}")
    print(f"Growth-rate plot         : {out_png}")
    print(f"Growth-rate fit CSV      : {fit_csv}")

    print(
        "Interpretation           :",
        "growing" if gamma > 0 else "decaying"
    )

    print("NOTE: confirm a clean linear regime before interpreting the sign.")


if __name__ == "__main__":
    main()
