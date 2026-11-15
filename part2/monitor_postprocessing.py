#!/usr/bin/env python3
"""
Real-time convergence and postProcessing dashboard for OpenFOAM (simpleFoam).
Monitors postProcessing/ and updates plots upon file modification.
Features 6-panel monitoring with warmup filtering (ignores t <= 100 for y-scaling).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CASE_DIR = "."
POST_DIR = os.path.join(CASE_DIR, "postProcessing")
POLL_INTERVAL_MS = 1500  # Refresh rate in milliseconds
RHO_WATER = 997.0        # Density in kg/m3 for physical stress calculation
WARMUP_ITERS = 0       # Iterations to exclude from vertical y-axis scaling

# Styling and Figure Initialization (2x3 Grid)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

if fig.canvas.manager is not None:
    fig.canvas.manager.set_window_title("OpenFOAM Live Convergence & Boundary Layer Monitor")

LAST_MTIMES = {}
INITIAL_DRAW = True


def get_latest_data_file(func_name, file_candidates):
    """Finds the latest time directory inside postProcessing/<func_name>/."""
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


def read_openfoam_table(filepath):
    """Parses scalar tabulated OpenFOAM files without breaking functional tokens."""
    if not filepath or not os.path.isfile(filepath):
        return None, None

    headers = []
    rows = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    stripped = line.lstrip("#").strip()
                    tokens = stripped.split()
                    if tokens and tokens[0].lower() in ["time", "timestep"]:
                        headers = tokens
                else:
                    parts = line.split()
                    rows.append(parts)
    except Exception:
        return None, None

    if not rows:
        return None, None

    num_cols = len(rows[0])
    if not headers or len(headers) != num_cols:
        if headers and len(headers) < num_cols:
            headers = headers + [f"Col_{i}" for i in range(len(headers), num_cols)]
        else:
            headers = ["Time"] + [f"Field_{i}" for i in range(1, num_cols)]

    col_data = {}
    for i, col_name in enumerate(headers):
        if i >= num_cols:
            break
        raw_vals = [r[i] for r in rows if len(r) > i]
        try:
            col_data[col_name] = np.array(raw_vals, dtype=float)
        except ValueError:
            continue

    return headers, col_data


def read_yplus_by_patch(filepath):
    """
    Parses yPlus.dat and separates series by patch name (e.g. 'walls' and 'cube').
    Returns dict: {patch_name: {'Time': arr, 'min': arr, 'max': arr, 'average': arr}}
    """
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
                    y_min = float(parts[2])
                    y_max = float(parts[3])
                    y_avg = float(parts[4])
                    if patch not in patch_data:
                        patch_data[patch] = {"Time": [], "min": [], "max": [], "average": []}
                    patch_data[patch]["Time"].append(t)
                    patch_data[patch]["min"].append(y_min)
                    patch_data[patch]["max"].append(y_max)
                    patch_data[patch]["average"].append(y_avg)
    except Exception:
        return {}

    for p in patch_data:
        for k in patch_data[p]:
            patch_data[p][k] = np.array(patch_data[p][k], dtype=float)
    return patch_data


def read_wall_shear_stress(filepath):
    """
    Robustly parses wallShearStress.dat handling (tau_x tau_y tau_z) vector format.
    Separates peak shear stress on 'walls' and 'cube' and scales by density (Pa).
    """
    if not filepath or not os.path.isfile(filepath):
        return None, None, None

    by_time = {}

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                vec_matches = re.findall(r"\(([^)]+)\)", line)
                if len(vec_matches) >= 2:
                    parts = line.split()
                    t = float(parts[0])
                    patch = parts[1]

                    v_min = [float(x) for x in vec_matches[0].split()]
                    v_max = [float(x) for x in vec_matches[1].split()]

                    peak_val = max(abs(v_min[2]), abs(v_max[2])) * RHO_WATER
                    if t not in by_time:
                        by_time[t] = {"walls": np.nan, "cube": np.nan}
                    if patch in by_time[t]:
                        by_time[t][patch] = peak_val
                    else:
                        by_time[t]["walls"] = peak_val
    except Exception:
        return None, None, None

    if not by_time:
        return None, None, None

    sorted_t = sorted(by_time.keys())
    times = np.array(sorted_t)
    tau_walls = np.array([by_time[t]["walls"] for t in sorted_t])
    tau_cube = np.array([by_time[t]["cube"] for t in sorted_t])

    return times, tau_walls, tau_cube


def has_new_data():
    """Checks whether any monitored function object output has updated on disk."""
    changed = False
    targets = [
        ("residuals", ["solverInfo.dat", "residuals.dat"]),
        ("inletPressure", ["surfaceFieldValue.dat"]),
        ("massFlowInlet", ["surfaceFieldValue.dat"]),
        ("massFlowOutlet", ["surfaceFieldValue.dat"]),
        ("wallShearStress", ["wallShearStress.dat"]),
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


def apply_adaptive_limits(ax, x_full, y_filtered, is_log=False, margin=0.08):
    """
    Applies x-limits based on full time history, and y-limits based exclusively
    on data past the warmup phase (y_filtered).
    """
    if len(x_full) == 0:
        return

    valid_x = x_full[np.isfinite(x_full)]
    if len(valid_x) == 0:
        return

    x_min, x_max = np.min(valid_x), np.max(valid_x)
    if x_min == x_max:
        ax.set_xlim(x_min - 1, x_max + 1)
    else:
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.02)

    # Indicate warmup boundary line if crossed
    if x_max > WARMUP_ITERS:
        ax.axvline(WARMUP_ITERS, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    if len(y_filtered) == 0:
        return

    valid_y = y_filtered[np.isfinite(y_filtered)]
    if len(valid_y) == 0:
        return

    if is_log:
        positive_y = valid_y[valid_y > 0]
        if len(positive_y) > 0:
            log_min = np.floor(np.log10(np.min(positive_y)))
            log_max = np.ceil(np.log10(np.max(positive_y)))
            if log_min == log_max:
                log_min -= 0.5
                log_max += 0.5
            ax.set_ylim(10 ** (log_min - 0.2), 10 ** (log_max + 0.2))
    else:
        y_min, y_max = np.min(valid_y), np.max(valid_y)
        if y_min == y_max:
            offset = 1.0 if y_min == 0 else abs(y_min) * 0.05
            ax.set_ylim(y_min - offset, y_max + offset)
        else:
            delta = y_max - y_min
            ax.set_ylim(y_min - delta * margin, y_max + delta * margin)


def get_warmup_mask(time_arr):
    """Returns a boolean mask selecting iterations >= WARMUP_ITERS if past warmup."""
    if len(time_arr) == 0:
        return np.array([], dtype=bool)
    if np.max(time_arr) > WARMUP_ITERS:
        mask = time_arr >= WARMUP_ITERS
        # Fallback if sparse writes left no points past warmup
        return mask if np.any(mask) else np.ones(len(time_arr), dtype=bool)
    return np.ones(len(time_arr), dtype=bool)


def update_plots(frame):
    """Periodic animation callback."""
    global INITIAL_DRAW
    if not INITIAL_DRAW and not has_new_data():
        return
    INITIAL_DRAW = False

    for ax in axes.flat:
        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------------------
    # 1. Residuals Plot (Top-Left: [0, 0])
    # -------------------------------------------------------------------------
    ax_res = axes[0, 0]
    res_file = get_latest_data_file("residuals", ["solverInfo.dat", "residuals.dat"])
    _, res_dict = read_openfoam_table(res_file)

    if res_dict and "Time" in res_dict:
        time = res_dict["Time"]
        mask = get_warmup_mask(time)
        target_initials = ["Ux_initial", "Uy_initial", "Uz_initial", "k_initial", "omega_initial", "p_initial"]
        scaling_vals = []

        for key in target_initials:
            if key in res_dict:
                clean_label = key.replace("_initial", "")
                y_vals = res_dict[key]
                ax_res.semilogy(time, y_vals, label=clean_label, linewidth=1.5)
                scaling_vals.extend(y_vals[mask])

        if not scaling_vals:
            for key, y_vals in res_dict.items():
                if key != "Time":
                    ax_res.semilogy(time, y_vals, label=key, linewidth=1.5)
                    scaling_vals.extend(y_vals[mask])

        ax_res.axhline(1e-6, color="crimson", linestyle="--", linewidth=1.0, alpha=0.7, label="Target (1e-6)")
        ax_res.set_title("Equation Residuals (Initial)", fontsize=11, fontweight="bold")
        ax_res.set_xlabel("Iteration")
        ax_res.set_ylabel("Initial Residual")
        ax_res.legend(loc="upper right", fontsize=8, ncol=2)

        apply_adaptive_limits(ax_res, time, np.array(scaling_vals), is_log=True)
    else:
        ax_res.text(0.5, 0.5, "Waiting for residuals...", ha="center", va="center", color="gray")
        ax_res.set_title("Equation Residuals (Initial)", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # 2. Inlet Pressure Evolution (Top-Middle: [0, 1])
    # -------------------------------------------------------------------------
    ax_p = axes[0, 1]
    p_file = get_latest_data_file("inletPressure", "surfaceFieldValue.dat")
    _, p_dict = read_openfoam_table(p_file)

    if p_dict and "Time" in p_dict and len(p_dict) >= 2:
        time_p = p_dict["Time"]
        p_keys = [k for k in p_dict.keys() if k != "Time"]
        p_val = p_dict[p_keys[0]]
        mask_p = get_warmup_mask(time_p)

        ax_p.plot(time_p, p_val, color="tab:blue", linewidth=1.8, label=r"$\langle p \rangle_{\mathrm{inlet}}$")
        ax_p.set_title(f"Inlet Pressure Drop (Current: {p_val[-1]:.5e} m²/s²)", fontsize=11, fontweight="bold")
        ax_p.set_xlabel("Iteration")
        ax_p.set_ylabel("Kinematic Pressure [m²/s²]")
        ax_p.legend(loc="lower right", fontsize=9)

        apply_adaptive_limits(ax_p, time_p, p_val[mask_p], is_log=False, margin=0.08)
    else:
        ax_p.text(0.5, 0.5, "Waiting for inlet pressure...", ha="center", va="center", color="gray")
        ax_p.set_title("Inlet Pressure Drop Plateau", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # 3. Mass Flow Rates: Combined on Same Graph (Top-Right: [0, 2])
    # -------------------------------------------------------------------------
    ax_flows = axes[0, 2]
    in_file = get_latest_data_file("massFlowInlet", "surfaceFieldValue.dat")
    out_file = get_latest_data_file("massFlowOutlet", "surfaceFieldValue.dat")
    _, in_dict = read_openfoam_table(in_file)
    _, out_dict = read_openfoam_table(out_file)

    if in_dict and out_dict and "Time" in in_dict and "Time" in out_dict:
        in_keys = [k for k in in_dict.keys() if k != "Time"]
        out_keys = [k for k in out_dict.keys() if k != "Time"]

        if in_keys and out_keys:
            min_len = min(len(in_dict["Time"]), len(out_dict["Time"]))
            time_f = in_dict["Time"][:min_len]
            phi_in_mag = np.abs(in_dict[in_keys[0]][:min_len])
            phi_out = out_dict[out_keys[0]][:min_len]
            mask_f = get_warmup_mask(time_f)

            ax_flows.plot(time_f, phi_in_mag, color="tab:green", linestyle="--", linewidth=2.0, label=r"Inlet $|\sum \phi_{\mathrm{in}}|$")
            ax_flows.plot(time_f, phi_out, color="tab:blue", linewidth=1.5, alpha=0.85, label=r"Outlet $\sum \phi_{\mathrm{out}}$")

            ax_flows.set_title(f"Mass Flux Balance (Out: {phi_out[-1]:.6f} m³/s)", fontsize=11, fontweight="bold")
            ax_flows.set_xlabel("Iteration")
            ax_flows.set_ylabel("Volumetric Flux [m³/s]")
            ax_flows.legend(loc="center right", fontsize=8.5)

            scaling_flows = np.concatenate([phi_in_mag[mask_f], phi_out[mask_f]])
            apply_adaptive_limits(ax_flows, time_f, scaling_flows, is_log=False, margin=0.15)
    else:
        ax_flows.text(0.5, 0.5, "Waiting for mass flow monitors...", ha="center", va="center", color="gray")
        ax_flows.set_title("Mass Flow Rates (Combined)", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # 4. Mass Continuity Imbalance % (Bottom-Left: [1, 0])
    # -------------------------------------------------------------------------
    ax_imb = axes[1, 0]
    if in_dict and out_dict and "Time" in in_dict and "Time" in out_dict:
        in_keys = [k for k in in_dict.keys() if k != "Time"]
        out_keys = [k for k in out_dict.keys() if k != "Time"]

        if in_keys and out_keys:
            min_len = min(len(in_dict["Time"]), len(out_dict["Time"]))
            time_flux = in_dict["Time"][:min_len]
            phi_in = in_dict[in_keys[0]][:min_len]
            phi_out = out_dict[out_keys[0]][:min_len]
            mask_imb = get_warmup_mask(time_flux)

            error_pct = np.abs(phi_in + phi_out) / (np.abs(phi_in) + 1e-12) * 100.0
            error_pct_plot = np.clip(error_pct, 1e-10, None)

            ax_imb.semilogy(time_flux, error_pct_plot, color="tab:purple", linewidth=1.8, label="Net Imbalance (%)")
            ax_imb.axhline(0.01, color="red", linestyle=":", label="0.01% Criterion")
            ax_imb.set_title(f"Continuity Error (Current: {error_pct[-1]:.2e}%)", fontsize=11, fontweight="bold")
            ax_imb.set_xlabel("Iteration")
            ax_imb.set_ylabel("Imbalance [%]")
            ax_imb.legend(loc="upper right", fontsize=8.5)

            apply_adaptive_limits(ax_imb, time_flux, error_pct_plot[mask_imb], is_log=True)
    else:
        ax_imb.text(0.5, 0.5, "Waiting for mass balance...", ha="center", va="center", color="gray")
        ax_imb.set_title("Continuity Imbalance (%)", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # 5. Wall Shear Stress Evolution (Bottom-Middle: [1, 1])
    # -------------------------------------------------------------------------
    ax_tau = axes[1, 1]
    tau_file = get_latest_data_file("wallShearStress", "wallShearStress.dat")
    t_tau, tau_walls, tau_cube = read_wall_shear_stress(tau_file)

    if t_tau is not None and len(t_tau) > 0:
        mask_tau = get_warmup_mask(t_tau)

        ax_tau.semilogy(t_tau, tau_walls, color="tab:red", marker="o", markersize=3.5, linewidth=1.5, label=r"Peak $|\tau_{w,z}|$ (Duct Walls)")
        ax_tau.semilogy(t_tau, tau_cube, color="tab:blue", marker="s", markersize=3.5, linewidth=1.5, label=r"Peak $|\tau_{w,z}|$ (Cube)")

        latest_cube = tau_cube[-1] if np.isfinite(tau_cube[-1]) else 0.0
        ax_tau.set_title(f"Peak Wall Shear (Walls: {tau_walls[-1]:.2f}, Cube: {latest_cube:.2f} Pa)", fontsize=10.5, fontweight="bold")
        ax_tau.set_xlabel("Iteration")
        ax_tau.set_ylabel(r"Peak Streamwise $|\tau_{w,z}|$ [Pa]")
        ax_tau.legend(loc="best", fontsize=8.5)

        scaling_tau = np.concatenate([tau_walls[mask_tau], tau_cube[mask_tau]])
        apply_adaptive_limits(ax_tau, t_tau, scaling_tau, is_log=True)
    else:
        ax_tau.text(0.5, 0.5, "Waiting for wallShearStress.dat...", ha="center", va="center", color="gray")
        ax_tau.set_title("Wall Shear Stress", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # 6. yPlus Boundary Layer Resolution (Bottom-Right: [1, 2])
    # -------------------------------------------------------------------------
    ax_yp = axes[1, 2]
    yp_file = get_latest_data_file("yPlus", "yPlus.dat")
    yp_patches = read_yplus_by_patch(yp_file)

    if yp_patches:
        scaling_list = []
        all_times = []
        title_parts = []

        if "walls" in yp_patches and len(yp_patches["walls"]["Time"]) > 0:
            tw = yp_patches["walls"]["Time"]
            w_avg = yp_patches["walls"]["average"]
            w_max = yp_patches["walls"]["max"]
            mw = get_warmup_mask(tw)
            ax_yp.plot(tw, w_avg, label=r"Walls $y^+_{\mathrm{avg}}$", color="tab:blue", marker="s", markersize=4, linewidth=1.8)
            ax_yp.plot(tw, w_max, label=r"Walls $y^+_{\mathrm{max}}$", color="tab:cyan", linestyle=":", marker="o", markersize=3, linewidth=1.2)
            scaling_list.extend([w_avg[mw], w_max[mw]])
            all_times.append(tw)
            title_parts.append(f"Walls Avg: {w_avg[-1]:.2f}")

        if "cube" in yp_patches and len(yp_patches["cube"]["Time"]) > 0:
            tc = yp_patches["cube"]["Time"]
            c_avg = yp_patches["cube"]["average"]
            c_max = yp_patches["cube"]["max"]
            mc = get_warmup_mask(tc)
            ax_yp.plot(tc, c_avg, label=r"Cube $y^+_{\mathrm{avg}}$", color="tab:orange", marker="^", markersize=4, linewidth=1.8)
            ax_yp.plot(tc, c_max, label=r"Cube $y^+_{\mathrm{max}}$", color="crimson", linestyle="--", marker="d", markersize=3, linewidth=1.2)
            scaling_list.extend([c_avg[mc], c_max[mc]])
            all_times.append(tc)
            title_parts.append(f"Cube Avg: {c_avg[-1]:.2f}")

        ax_yp.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Low-Re Limit (y⁺ = 1)")

        title_str = " | ".join(title_parts) if title_parts else "yPlus"
        ax_yp.set_title(f"Near-Wall y⁺ ({title_str})", fontsize=10.5, fontweight="bold")
        ax_yp.set_xlabel("Iteration")
        ax_yp.set_ylabel("y⁺ Value")
        ax_yp.legend(loc="upper right", fontsize=7.5, ncol=2)

        if all_times and scaling_list:
            t_combined = np.concatenate(all_times)
            y_combined = np.concatenate(scaling_list)
            apply_adaptive_limits(ax_yp, t_combined, y_combined, is_log=False, margin=0.15)
            cur_ymin, cur_ymax = ax_yp.get_ylim()
            ax_yp.set_ylim(min(0.0, cur_ymin), max(1.2, cur_ymax))
    else:
        ax_yp.text(0.5, 0.5, "Waiting for yPlus.dat...", ha="center", va="center", color="gray")
        ax_yp.set_title("Near-Wall Resolution (yPlus)", fontsize=11, fontweight="bold")

    fig.tight_layout()


def main():
    if not os.path.exists(POST_DIR):
        print(f"[*] Note: '{POST_DIR}' not detected yet. Waiting for simpleFoam initialization...")

    _ = FuncAnimation(fig, update_plots, interval=POLL_INTERVAL_MS, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()