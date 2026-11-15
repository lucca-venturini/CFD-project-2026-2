#!/usr/bin/env python3
"""
Real-time transient wake & Courant number dashboard for OpenFOAM (Part III: pimpleFoam).
Monitors postProcessing/ (courantExtrema, wakeProbes, inletPressure, massFlow, yPlus).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CASE_DIR = "."
POST_DIR = os.path.join(CASE_DIR, "postProcessing")
POLL_INTERVAL_MS = 1500
RHO_WATER = 997.0

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

if fig.canvas.manager is not None:
    fig.canvas.manager.set_window_title("OpenFOAM Part III: Transient Wake & PIMPLE Monitor (pimpleFoam)")

LAST_MTIMES = {}
INITIAL_DRAW = True


def get_latest_data_file(func_name, file_candidates):
    if isinstance(file_candidates, str):
        file_candidates = [file_candidates]
    func_dir = os.path.join(POST_DIR, func_name)
    if not os.path.isdir(func_dir):
        return None
    time_dirs = [d for d in os.listdir(func_dir) if os.path.isdir(os.path.join(func_dir, d))]
    if not time_dirs:
        return None
    try:
        sorted_dirs = sorted(time_dirs, key=lambda x: float(x))
    except ValueError:
        sorted_dirs = sorted(time_dirs)
    for fname in file_candidates:
        candidate_path = os.path.join(func_dir, sorted_dirs[-1], fname)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def read_field_minmax(filepath):
    """Parses fieldMinMax.dat for Co, U, p, k."""
    if not filepath or not os.path.isfile(filepath):
        return {}
    data = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) >= 4:
                    t = float(parts[0])
                    field = parts[1].strip()
                    vmin = float(parts[2])
                    # max is after min location or in column
                    # Find all floats outside parentheses
                    clean = re.sub(r"\([^)]*\)", "", line)
                    tokens = clean.split()
                    if len(tokens) >= 4:
                        t = float(tokens[0])
                        field = tokens[1]
                        vmin = float(tokens[2])
                        vmax = float(tokens[4]) if len(tokens) >= 5 else float(tokens[3])
                        if field not in data:
                            data[field] = {"Time": [], "min": [], "max": []}
                        data[field]["Time"].append(t)
                        data[field]["min"].append(vmin)
                        data[field]["max"].append(vmax)
    except Exception:
        return {}
    for k in data:
        for col in data[k]:
            data[k][col] = np.array(data[k][col], dtype=float)
    return data


def read_vector_probes(filepath):
    """Parses OpenFOAM vector probe file: t (Ux0 Uy0 Uz0) (Ux1 Uy1 Uz1) ..."""
    if not filepath or not os.path.isfile(filepath):
        return None, None
    times = []
    vecs = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                matches = re.findall(r"\(([^)]+)\)", line)
                if matches:
                    t_str = line.split("(")[0].strip()
                    times.append(float(t_str))
                    row = [[float(x) for x in m.split()] for m in matches]
                    vecs.append(row)
    except Exception:
        return None, None
    if not times:
        return None, None
    return np.array(times), np.array(vecs)  # shape (N_t, N_probes, 3)


def read_scalar_probes(filepath):
    """Parses OpenFOAM scalar probe file: t p0 p1 p2 ..."""
    if not filepath or not os.path.isfile(filepath):
        return None, None
    rows = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append([float(x) for x in line.split()])
    except Exception:
        return None, None
    if not rows:
        return None, None
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def read_simple_surface(filepath):
    if not filepath or not os.path.isfile(filepath):
        return None, None
    t_list, v_list = [], []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    t_list.append(float(parts[0]))
                    v_list.append(float(parts[1]))
    except Exception:
        return None, None
    if not t_list:
        return None, None
    return np.array(t_list), np.array(v_list)


def read_yplus_by_patch(filepath):
    if not filepath or not os.path.isfile(filepath):
        return {}
    patch_data = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    t = float(parts[0])
                    patch = parts[1]
                    if patch not in patch_data:
                        patch_data[patch] = {"Time": [], "min": [], "max": [], "average": []}
                    patch_data[patch]["Time"].append(t)
                    patch_data[patch]["min"].append(float(parts[2]))
                    patch_data[patch]["max"].append(float(parts[3]))
                    patch_data[patch]["average"].append(float(parts[4]))
    except Exception:
        return {}
    for p in patch_data:
        for k in patch_data[p]:
            patch_data[p][k] = np.array(patch_data[p][k], dtype=float)
    return patch_data


def has_new_data():
    changed = False
    targets = [
        ("courantExtrema", ["fieldMinMax.dat"]),
        ("wakeProbes", ["U", "p", "k"]),
        ("inletPressure", ["surfaceFieldValue.dat"]),
        ("massFlowInlet", ["surfaceFieldValue.dat"]),
        ("massFlowOutlet", ["surfaceFieldValue.dat"]),
        ("yPlus", ["yPlus.dat"]),
    ]
    for func, candidates in targets:
        fpath = get_latest_data_file(func, candidates)
        if fpath:
            try:
                mtime = os.path.getmtime(fpath)
                if fpath not in LAST_MTIMES or LAST_MTIMES[fpath] != mtime:
                    LAST_MTIMES[fpath] = mtime
                    changed = True
            except OSError:
                continue
    return changed


def update_plots(frame):
    global INITIAL_DRAW
    if not INITIAL_DRAW and not has_new_data():
        return
    INITIAL_DRAW = False

    for ax in axes.flat:
        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

    # 1. Courant Number & Max Velocity History [0, 0]
    ax_co = axes[0, 0]
    mm_file = get_latest_data_file("courantExtrema", ["fieldMinMax.dat"])
    mm_data = read_field_minmax(mm_file)
    if "Co" in mm_data and len(mm_data["Co"]["Time"]) > 0:
        t_co = mm_data["Co"]["Time"]
        co_max = mm_data["Co"]["max"]
        ax_co.plot(t_co, co_max, color="tab:red", linewidth=1.6, label=r"$\mathrm{Co}_{\mathrm{max}}$")
        ax_co.axhline(0.8, color="black", linestyle="--", linewidth=1.0, label="Target Limit (0.8)")
        ax_co.set_title(f"Courant Number History (Current Max: {co_max[-1]:.3f})", fontsize=11, fontweight="bold")
        ax_co.set_xlabel("Physical Time t [s]")
        ax_co.set_ylabel("Max Courant Number")
        ax_co.legend(loc="upper right", fontsize=8.5)
    else:
        ax_co.text(0.5, 0.5, "Waiting for courantExtrema...", ha="center", va="center", color="gray")
        ax_co.set_title("Courant Number History", fontsize=11, fontweight="bold")

    # 2. Wake Probe Streamwise Velocity Uz(t) [0, 1]
    ax_u = axes[0, 1]
    u_probe_file = get_latest_data_file("wakeProbes", ["U"])
    t_pu, u_vecs = read_vector_probes(u_probe_file)
    if t_pu is not None and u_vecs.shape[1] >= 7:
        probe_labels = [
            (2, "P2: Roof Shear Layer (z=0.425m)", "tab:purple"),
            (3, "P3: Near Wake (0.5 Hc)", "tab:red"),
            (4, "P4: Wake Core (2.0 Hc)", "tab:orange"),
            (5, "P5: Reattachment (4.0 Hc)", "tab:green"),
            (6, "P6: Far Wake (8.0 Hc)", "tab:blue"),
        ]
        for idx_p, lbl, col in probe_labels:
            ax_u.plot(t_pu, u_vecs[:, idx_p, 2], label=lbl, color=col, linewidth=1.4)
        ax_u.set_title("Wake Probes: Streamwise Velocity Uz(t)", fontsize=11, fontweight="bold")
        ax_u.set_xlabel("Physical Time t [s]")
        ax_u.set_ylabel("Uz [m/s]")
        ax_u.legend(loc="best", fontsize=7.5)
    else:
        ax_u.text(0.5, 0.5, "Waiting for wakeProbes/U...", ha="center", va="center", color="gray")
        ax_u.set_title("Wake Probes: Streamwise Velocity Uz(t)", fontsize=11, fontweight="bold")

    # 3. Wake Probe Transverse/Vertical Velocity Fluctuations Uy(t) [0, 2]
    ax_v = axes[0, 2]
    if t_pu is not None and u_vecs.shape[1] >= 7:
        for idx_p, lbl, col in [
            (2, "P2: Roof Shear Uy", "tab:purple"),
            (3, "P3: Near Wake Uy", "tab:red"),
            (4, "P4: Wake Core Uy", "tab:orange"),
            (5, "P5: Reattachment Uy", "tab:green"),
        ]:
            ax_v.plot(t_pu, u_vecs[:, idx_p, 1], label=lbl, color=col, linewidth=1.4)
        ax_v.set_title("Wake Probes: Vertical Velocity Uy(t) (Shedding Check)", fontsize=11, fontweight="bold")
        ax_v.set_xlabel("Physical Time t [s]")
        ax_v.set_ylabel("Uy [m/s]")
        ax_v.legend(loc="best", fontsize=8)
    else:
        ax_v.text(0.5, 0.5, "Waiting for wakeProbes/U...", ha="center", va="center", color="gray")
        ax_v.set_title("Wake Probes: Vertical Velocity Uy(t)", fontsize=11, fontweight="bold")

    # 4. Probe Static Pressure Histories [1, 0]
    ax_pp = axes[1, 0]
    p_probe_file = get_latest_data_file("wakeProbes", ["p"])
    t_pp, p_vals = read_scalar_probes(p_probe_file)
    if t_pp is not None and p_vals.shape[1] >= 7:
        for idx_p, lbl, col in [
            (0, "P0: Upstream (z=0.30m)", "tab:blue"),
            (2, "P2: Roof Shear (z=0.425m)", "tab:purple"),
            (3, "P3: Near Wake (0.5 Hc)", "tab:red"),
            (4, "P4: Wake Core (2.0 Hc)", "tab:orange"),
            (6, "P6: Far Wake (8.0 Hc)", "tab:green"),
        ]:
            ax_pp.plot(t_pp, p_vals[:, idx_p] * RHO_WATER, label=lbl, color=col, linewidth=1.4)
        ax_pp.set_title("Wake Probes: Static Pressure P(t) [Pa]", fontsize=11, fontweight="bold")
        ax_pp.set_xlabel("Physical Time t [s]")
        ax_pp.set_ylabel("Gauge Pressure [Pa]")
        ax_pp.legend(loc="best", fontsize=7.8)
    else:
        ax_pp.text(0.5, 0.5, "Waiting for wakeProbes/p...", ha="center", va="center", color="gray")
        ax_pp.set_title("Wake Probes: Static Pressure P(t)", fontsize=11, fontweight="bold")

    # 5. Continuity Imbalance % [1, 1]
    ax_imb = axes[1, 1]
    in_file = get_latest_data_file("massFlowInlet", "surfaceFieldValue.dat")
    out_file = get_latest_data_file("massFlowOutlet", "surfaceFieldValue.dat")
    t_in, phi_in = read_simple_surface(in_file)
    t_out, phi_out = read_simple_surface(out_file)
    if t_in is not None and t_out is not None:
        n_m = min(len(t_in), len(t_out))
        err_pct = np.abs(phi_in[:n_m] + phi_out[:n_m]) / (np.abs(phi_in[:n_m]) + 1e-12) * 100.0
        ax_imb.semilogy(t_in[:n_m], np.clip(err_pct, 1e-10, None), color="tab:purple", linewidth=1.6, label=r"$\varepsilon_m$ (%)")
        ax_imb.set_title(f"Continuity Imbalance (Current: {err_pct[-1]:.2e}%)", fontsize=11, fontweight="bold")
        ax_imb.set_xlabel("Physical Time t [s]")
        ax_imb.set_ylabel("Imbalance [%]")
        ax_imb.legend(loc="upper right", fontsize=8.5)
    else:
        ax_imb.text(0.5, 0.5, "Waiting for massFlow...", ha="center", va="center", color="gray")
        ax_imb.set_title("Continuity Imbalance (%)", fontsize=11, fontweight="bold")

    # 6. yPlus Resolution [1, 2]
    ax_yp = axes[1, 2]
    yp_file = get_latest_data_file("yPlus", "yPlus.dat")
    yp_patches = read_yplus_by_patch(yp_file)
    if yp_patches:
        if "walls" in yp_patches:
            ax_yp.plot(yp_patches["walls"]["Time"], yp_patches["walls"]["average"], marker="s", color="tab:blue", label="Walls y+ Avg")
        if "cube" in yp_patches:
            ax_yp.plot(yp_patches["cube"]["Time"], yp_patches["cube"]["average"], marker="^", color="tab:orange", label="Cube y+ Avg")
        ax_yp.axhline(1.0, color="black", linestyle="--", label="y+ = 1")
        ax_yp.set_title("Near-Wall Resolution y+(t)", fontsize=11, fontweight="bold")
        ax_yp.set_xlabel("Physical Time t [s]")
        ax_yp.set_ylabel("y+")
        ax_yp.legend(loc="best", fontsize=8.5)
    else:
        ax_yp.text(0.5, 0.5, "Waiting for yPlus.dat (writes every 0.25s)...", ha="center", va="center", color="gray")
        ax_yp.set_title("Near-Wall Resolution y+(t)", fontsize=11, fontweight="bold")

    fig.tight_layout()


def main():
    _ = FuncAnimation(fig, update_plots, interval=POLL_INTERVAL_MS, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
