
#!/usr/bin/env python3
"""
Estimate the numerical critical Rayleigh number from
previously fitted perturbation growth rates.
"""

import argparse
import csv

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/rb/critical")
    )

    args = parser.parse_args()

    values = []

    # ========================================================
    # Read the fitted growth rates
    # ========================================================

    for path in args.data.glob("Ra_*_fit.csv"):

        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        if len(rows) == 1:
            values.append((
                float(rows[0]["Ra"]),
                float(rows[0]["gamma"])
            ))

    values.sort()

    if len(values) < 2:
        parser.error("At least two Rayleigh runs are needed.")

    r = np.array([v[0] for v in values])
    gamma = np.array([v[1] for v in values])

    # ========================================================
    # Find sign changes
    #
    # gamma < 0: stable
    # gamma > 0: unstable
    # ========================================================

    brackets = [
        (i, i + 1)
        for i in range(len(r) - 1)
        if gamma[i] * gamma[i + 1] <= 0
    ]

    if not brackets:

        print("No zero crossing found.")
        print("Test more Rayleigh numbers to bracket Ra_critical.")

        for ri, gi in values:
            print(f"Ra={ri:g}, gamma*={gi:+.7g}")

        return

    # ========================================================
    # Choose the narrowest bracket
    # ========================================================

    i, j = min(
        brackets,
        key=lambda p: r[p[1]] - r[p[0]]
    )

    # ========================================================
    # Linear interpolation
    # ========================================================

    rac = r[i] - gamma[i] * (
        r[j] - r[i]
    ) / (
        gamma[j] - gamma[i]
    )

    print(
        f"Bracket: Ra={r[i]:g} "
        f"(gamma*={gamma[i]:+.6g}) to "
        f"Ra={r[j]:g} "
        f"(gamma*={gamma[j]:+.6g})"
    )

    print(f"Interpolated numerical critical Ra = {rac:.6g}")

    # ========================================================
    # Plot growth rate against Rayleigh number
    # ========================================================

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(
        r,
        gamma,
        "o",
        label="Measured rates"
    )

    ax.plot(
        [r[i], r[j]],
        [gamma[i], gamma[j]],
        "--",
        label="Local linear interpolation"
    )

    ax.axhline(0, color="0.4", lw=1)

    ax.axvline(
        rac,
        color="0.4",
        lw=1,
        ls=":",
        label=fr"$Ra_c \approx {rac:.4g}$"
    )

    ax.set(
        xlabel="Rayleigh number",
        ylabel=r"Growth rate $\gamma^*$",
        title="TBGK linear-stability threshold"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path = args.data / "critical_rayleigh.png"

    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"Plot saved: {path}")


if __name__ == "__main__":
    main()
