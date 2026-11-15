#!/usr/bin/env python3
"""
Real-time convergence and thermal balance dashboard for OpenFOAM Part IV
(chtMultiRegionSimpleFoam: fluid + solid regions).
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CASE_DIR = "."
POST_DIR = os.path.join(CASE_DIR, "postProcessing")

CP_WATER = 4184.0       # J/(kg K)
M_DOT_NOMINAL = 40.0    # kg/s
Q_FVOPTIONS_MW = 3.350  # q''' * V_s = 2.68e10 * 1.25e-4 = 3.350 MW
Q_REF_MW = 3.3472       # Analytical 20 K mixed-mean target: m_dot * Cp * 20
DELTA_T_TARGET = 20.0   # K

LAST_MTIMES = {}
INITIAL_DRAW = True


def get_all_data_files(region, func_name, file_candidates):
    """Returns chronologically sorted data files inside postProcessing/<region>/<func_name>/."""
    if isinstance(file_candidates, str):
        file_candidates = [file_candidates]

    func_dir = os.path.join(POST_DIR, region, func_name)
    if not os.path.isdir(func_dir):
        return []

    time_dirs = [d for d in os.listdir(func_dir) if os.path.isdir(os.path.join(func_dir, d))]
    if not time_dirs:
        return []

    try:
        sorted_dirs = sorted(time_dirs, key=lambda x: float(x))
    except ValueError:
        sorted_dirs = sorted(time_dirs)

    files = []
    for tdir in sorted_dirs:
        for fname in file_candidates:
            candidate = os.path.join(func_dir, tdir, fname)
            if os.path.isfile(candidate):
                files.append(candidate)
                break
    return files


def _as_file_list(filepaths):
    if not filepaths:
        return []
    return [filepaths] if isinstance(filepaths, str) else list(filepaths)


def read_numeric_table(filepaths):
    """Parses and merges scalar OpenFOAM tables chronologically across restart segments."""
    by_time = {}
    col_names_ref = None

    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
        headers = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        tokens = line.lstrip("#").strip().split()
                        if tokens and tokens[0].lower() in ("time", "timestep"):
                            headers = tokens
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if not headers or len(headers) != len(parts):
                        headers = ["Time"] + [f"Col_{i}" for i in range(1, len(parts))]
                    if col_names_ref is None:
                        col_names_ref = headers[1:]
                    try:
                        t_val = float(parts[0])
                        row_dict = {}
                        for idx, name in enumerate(headers[1:], start=1):
                            row_dict[name] = float(parts[idx])
                        by_time[t_val] = row_dict
                    except ValueError:
                        continue
        except OSError:
            continue

    if not by_time or not col_names_ref:
        return {}

    sorted_t = sorted(by_time.keys())
    data = {"Time": np.array(sorted_t, dtype=float)}
    for name in col_names_ref:
        vals = [by_time[t].get(name, np.nan) for t in sorted_t]
        data[name] = np.array(vals, dtype=float)
    return data


def read_solver_residuals(filepaths):
    """Parses and merges solverInfo.dat chronologically across restart segments."""
    by_time = {}
    all_fields = set()

    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
        headers = []
        init_cols = {}
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        tokens = line.lstrip("#").strip().split()
                        if tokens and tokens[0].lower() == "time":
                            headers = tokens
                            init_cols = {
                                h[:-len("_initial")]: idx
                                for idx, h in enumerate(headers)
                                if h.endswith("_initial")
                            }
                            all_fields.update(init_cols.keys())
                        continue
                    if not init_cols:
                        continue
                    parts = line.split()
                    if len(parts) <= max(init_cols.values()):
                        continue
                    try:
                        t_val = float(parts[0])
                        by_time[t_val] = {k: float(parts[idx]) for k, idx in init_cols.items()}
                    except ValueError:
                        continue
        except OSError:
            continue

    if not by_time or not all_fields:
        return {}

    sorted_t = sorted(by_time.keys())
    out = {"Time": np.array(sorted_t, dtype=float)}
    for k in sorted(all_fields):
        out[k] = np.array([by_time[t].get(k, np.nan) for t in sorted_t], dtype=float)
    return out


def read_field_minmax(filepaths, target_field="T"):
    """Parses and merges fieldMinMax.dat for a target field across restart segments."""
    by_time = {}
    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cleaned = re.sub(r"\([^)]*\)", "", line)
                    tokens = cleaned.split()
                    if len(tokens) >= 5 and tokens[1] == target_field:
                        by_time[float(tokens[0])] = (float(tokens[2]), float(tokens[4]))
        except (OSError, ValueError):
            continue

    if not by_time:
        return None, None, None
    sorted_t = sorted(by_time.keys())
    vmins = [by_time[t][0] for t in sorted_t]
    vmaxs = [by_time[t][1] for t in sorted_t]
    return np.array(sorted_t, dtype=float), np.array(vmins, dtype=float), np.array(vmaxs, dtype=float)


def read_wall_heat_flux(filepaths, patch_name="fluid_to_solid"):
    """Parses and merges wallHeatFlux.dat chronologically across restart segments."""
    by_time = {}
    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 5 and parts[1] == patch_name:
                        by_time[float(parts[0])] = float(parts[4])
        except (OSError, ValueError):
            continue

    if not by_time:
        return None, None
    sorted_t = sorted(by_time.keys())
    return np.array(sorted_t, dtype=float), np.array([by_time[t] for t in sorted_t], dtype=float)


def read_yplus_by_patch(filepaths):
    """Parses and merges yPlus.dat across restart segments."""
    by_patch = {}
    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
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
                        if patch not in by_patch:
                            by_patch[patch] = {}
                        by_patch[patch][t] = (float(parts[3]), float(parts[4]))
        except (OSError, ValueError):
            continue

    patch_data = {}
    for patch, tmap in by_patch.items():
        sorted_t = sorted(tmap.keys())
        patch_data[patch] = {
            "Time": np.array(sorted_t, dtype=float),
            "max": np.array([tmap[t][0] for t in sorted_t], dtype=float),
            "average": np.array([tmap[t][1] for t in sorted_t], dtype=float),
        }
    return patch_data


def has_new_data():
    """Checks whether any monitored Part IV postProcessing file has updated."""
    changed = False
    targets = [
        ("fluid", "residualsFluid", ["solverInfo.dat"]),
        ("solid", "residualsSolid", ["solverInfo.dat"]),
        ("fluid", "wallHeatFlux", ["wallHeatFlux.dat"]),
        ("fluid", "inletTemperatureMix", ["surfaceFieldValue.dat"]),
        ("fluid", "outletTemperatureMix", ["surfaceFieldValue.dat"]),
        ("solid", "solidTempExtrema", ["fieldMinMax.dat"]),
        ("solid", "solidTempAvg", ["volFieldValue.dat"]),
        ("fluid", "fluidTempExtrema", ["fieldMinMax.dat"]),
        ("fluid", "interfaceTempFluid", ["surfaceFieldValue.dat"]),
        ("fluid", "inletPressure", ["surfaceFieldValue.dat"]),
        ("fluid", "outletPressure", ["surfaceFieldValue.dat"]),
        ("fluid", "pressureUpstreamCube", ["surfaceFieldValue.dat"]),
        ("fluid", "pressureDownstreamCube", ["surfaceFieldValue.dat"]),
        ("fluid", "massFlowInlet", ["surfaceFieldValue.dat"]),
        ("fluid", "massFlowOutlet", ["surfaceFieldValue.dat"]),
        ("fluid", "yPlus", ["yPlus.dat"]),
    ]

    for region, func, candidates in targets:
        for fpath in get_all_data_files(region, func, candidates):
            try:
                mtime = os.path.getmtime(fpath)
                if fpath not in LAST_MTIMES or LAST_MTIMES[fpath] != mtime:
                    LAST_MTIMES[fpath] = mtime
                    changed = True
            except OSError:
                continue
    return changed


def draw_dashboard(fig, axes):
    """Updates all 6 Part IV convergence and thermal monitoring panels."""
    for ax in axes.flat:
        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

    # [0, 0] Fluid & Solid Initial Residuals
    ax_res = axes[0, 0]
    res_f = read_solver_residuals(get_all_data_files("fluid", "residualsFluid", ["solverInfo.dat"]))
    res_s = read_solver_residuals(get_all_data_files("solid", "residualsSolid", ["solverInfo.dat"]))
    plotted_res = False
    if res_f and "Time" in res_f:
        t_rf = res_f["Time"]
        style_map = [
            ("Ux", "Ux (fluid)", "tab:blue", "-"),
            ("Uy", "Uy (fluid)", "tab:cyan", "--"),
            ("Uz", "Uz (fluid)", "tab:green", "-."),
            ("p_rgh", "p_rgh (fluid)", "tab:red", "-"),
            ("k", "k (fluid)", "tab:purple", ":"),
            ("omega", "omega (fluid)", "tab:brown", ":"),
            ("h", "h (fluid)", "tab:orange", "-"),
        ]
        for key, lbl, col, ls in style_map:
            if key in res_f and len(res_f[key]) > 0:
                ax_res.semilogy(t_rf, np.clip(res_f[key], 1e-12, None),
                                label=lbl, color=col, linestyle=ls, linewidth=1.3)
                plotted_res = True
    if res_s and "h" in res_s and len(res_s["h"]) > 0:
        ax_res.semilogy(res_s["Time"], np.clip(res_s["h"], 1e-12, None),
                        label="h (solid)", color="black", linestyle="--", linewidth=1.6)
        plotted_res = True
    ax_res.set_title("Fluid & Solid Initial Residuals", fontsize=11, fontweight="bold")
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Initial Residual")
    if plotted_res:
        ax_res.legend(loc="upper right", ncol=2, fontsize=7.5)
    else:
        ax_res.text(0.5, 0.5, "Waiting for solverInfo.dat...", ha="center", va="center", color="gray")

    # Shared fluid inlet/outlet mixed-mean & mass-flow data for Energy Balance & DeltaT
    t_in_tbl = read_numeric_table(get_all_data_files("fluid", "inletTemperatureMix", ["surfaceFieldValue.dat"]))
    t_out_tbl = read_numeric_table(get_all_data_files("fluid", "outletTemperatureMix", ["surfaceFieldValue.dat"]))
    m_in_tbl = read_numeric_table(get_all_data_files("fluid", "massFlowInlet", ["surfaceFieldValue.dat"]))
    m_out_tbl = read_numeric_table(get_all_data_files("fluid", "massFlowOutlet", ["surfaceFieldValue.dat"]))

    m_dot_val = M_DOT_NOMINAL
    if m_in_tbl and "sum(phi)" in m_in_tbl and len(m_in_tbl["sum(phi)"]) > 0:
        m_dot_val = float(np.abs(m_in_tbl["sum(phi)"][-1]))

    # [0, 1] Steady Energy Balance [MW]
    ax_eb = axes[0, 1]
    t_whf, q_int_w = read_wall_heat_flux(get_all_data_files("fluid", "wallHeatFlux", ["wallHeatFlux.dat"]))
    ax_eb.axhline(Q_FVOPTIONS_MW, color="black", linestyle="-", linewidth=1.4,
                  label=f"Q_fvOptions ({Q_FVOPTIONS_MW:.3f} MW)")
    ax_eb.axhline(Q_REF_MW, color="dimgray", linestyle="--", linewidth=1.1,
                  label=f"Q_ref Target ({Q_REF_MW:.4f} MW)")

    eps_q_str = "--"
    if t_whf is not None:
        q_int_mw = q_int_w / 1e6
        ax_eb.plot(t_whf, q_int_mw, color="tab:red", linewidth=1.6,
                   label=f"Q_interface ({q_int_mw[-1]:.3f} MW)")
    if (t_in_tbl and t_out_tbl and "weightedAverage(T)" in t_in_tbl
            and "weightedAverage(T)" in t_out_tbl):
        n_q = min(len(t_in_tbl["Time"]), len(t_out_tbl["Time"]))
        t_q = t_out_tbl["Time"][:n_q]
        dT_mix = t_out_tbl["weightedAverage(T)"][:n_q] - t_in_tbl["weightedAverage(T)"][:n_q]
        q_rem_mw = (m_dot_val * CP_WATER * dT_mix) / 1e6
        ax_eb.plot(t_q, q_rem_mw, color="tab:blue", marker="o", markersize=3, linewidth=1.5,
                   label=f"Q_removed ({q_rem_mw[-1]:.3f} MW)")
        eps_q = abs(Q_FVOPTIONS_MW - q_rem_mw[-1]) / Q_FVOPTIONS_MW * 100.0
        eps_q_str = f"{eps_q:.2f}%"

    ax_eb.set_title(f"Steady Energy Balance (eps_Q = {eps_q_str})", fontsize=11, fontweight="bold")
    ax_eb.set_xlabel("Iteration")
    ax_eb.set_ylabel("Thermal Power [MW]")
    ax_eb.legend(loc="lower right", fontsize=8)

    # [0, 2] Mixed-Mean Water Temperature Rise
    ax_dt = axes[0, 2]
    if (t_in_tbl and t_out_tbl and "weightedAverage(T)" in t_in_tbl
            and "weightedAverage(T)" in t_out_tbl):
        n_q = min(len(t_in_tbl["Time"]), len(t_out_tbl["Time"]))
        t_q = t_out_tbl["Time"][:n_q]
        dT_mix = t_out_tbl["weightedAverage(T)"][:n_q] - t_in_tbl["weightedAverage(T)"][:n_q]
        ax_dt.plot(t_q, dT_mix, color="tab:orange", marker="s", markersize=3, linewidth=1.6,
                   label=f"Delta T_mix ({dT_mix[-1]:.2f} K)")
        ax_dt.axhline(DELTA_T_TARGET, color="black", linestyle="--", linewidth=1.2,
                      label="Analytical Target (20.0 K)")
        ax_dt.set_title(f"Mixed-Mean Water Delta T_mix (Current: {dT_mix[-1]:.2f} K)",
                        fontsize=11, fontweight="bold")
        ax_dt.legend(loc="lower right", fontsize=8.5)
    else:
        ax_dt.text(0.5, 0.5, "Waiting for TemperatureMix...", ha="center", va="center", color="gray")
        ax_dt.set_title("Mixed-Mean Water Delta T_mix", fontsize=11, fontweight="bold")
    ax_dt.set_xlabel("Iteration")
    ax_dt.set_ylabel("T_out,mix - T_in,mix [K]")

    # [1, 0] Solid, Fluid & Interface Temperatures [K]
    ax_t = axes[1, 0]
    t_smax, _, s_max = read_field_minmax(
        get_all_data_files("solid", "solidTempExtrema", ["fieldMinMax.dat"]), "T"
    )
    t_fmax, _, f_max = read_field_minmax(
        get_all_data_files("fluid", "fluidTempExtrema", ["fieldMinMax.dat"]), "T"
    )
    s_avg_tbl = read_numeric_table(get_all_data_files("solid", "solidTempAvg", ["volFieldValue.dat"]))
    if_tbl = read_numeric_table(get_all_data_files("fluid", "interfaceTempFluid", ["surfaceFieldValue.dat"]))

    plotted_t = False
    if t_smax is not None:
        ax_t.plot(t_smax, s_max, color="tab:red", linewidth=1.6,
                  label=f"Cube T_max ({s_max[-1]:.0f} K)")
        plotted_t = True
    if s_avg_tbl and "volAverage(T)" in s_avg_tbl and len(s_avg_tbl["volAverage(T)"]) > 0:
        ax_t.plot(s_avg_tbl["Time"], s_avg_tbl["volAverage(T)"], color="tab:orange", linewidth=1.5,
                  label=f"Cube T_volAvg ({s_avg_tbl['volAverage(T)'][-1]:.0f} K)")
        plotted_t = True
    if t_fmax is not None:
        ax_t.plot(t_fmax, f_max, color="tab:purple", linestyle="--", linewidth=1.5,
                  label=f"Fluid T_max ({f_max[-1]:.0f} K)")
        plotted_t = True
    if if_tbl and "areaAverage(T)" in if_tbl and len(if_tbl["areaAverage(T)"]) > 0:
        ax_t.plot(if_tbl["Time"], if_tbl["areaAverage(T)"], color="tab:blue", linewidth=1.5,
                  label=f"Interface T_avg ({if_tbl['areaAverage(T)'][-1]:.0f} K)")
        plotted_t = True

    ax_t.set_title("Solid, Fluid & Interface Temperatures", fontsize=11, fontweight="bold")
    ax_t.set_xlabel("Iteration")
    ax_t.set_ylabel("Temperature [K]")
    if plotted_t:
        ax_t.legend(loc="best", fontsize=8)
    else:
        ax_t.text(0.5, 0.5, "Waiting for temperature fields...", ha="center", va="center", color="gray")

    # [1, 1] Duct & Cube Pressure Drop [Pa]
    ax_dp = axes[1, 1]
    p_in = read_numeric_table(get_all_data_files("fluid", "inletPressure", ["surfaceFieldValue.dat"]))
    p_out = read_numeric_table(get_all_data_files("fluid", "outletPressure", ["surfaceFieldValue.dat"]))
    p_up = read_numeric_table(get_all_data_files("fluid", "pressureUpstreamCube", ["surfaceFieldValue.dat"]))
    p_dn = read_numeric_table(get_all_data_files("fluid", "pressureDownstreamCube", ["surfaceFieldValue.dat"]))

    plotted_dp = False
    if p_in and p_out and "areaAverage(p)" in p_in and "areaAverage(p)" in p_out:
        n_p = min(len(p_in["Time"]), len(p_out["Time"]))
        dp_duct = p_in["areaAverage(p)"][:n_p] - p_out["areaAverage(p)"][:n_p]
        ax_dp.plot(p_in["Time"][:n_p], dp_duct, color="tab:blue", linewidth=1.6,
                   label=f"Total Duct dP ({dp_duct[-1]:.1f} Pa)")
        plotted_dp = True
    if p_up and p_dn and "areaAverage(p)" in p_up and "areaAverage(p)" in p_dn:
        n_c = min(len(p_up["Time"]), len(p_dn["Time"]))
        dp_cube = p_up["areaAverage(p)"][:n_c] - p_dn["areaAverage(p)"][:n_c]
        ax_dp.plot(p_up["Time"][:n_c], dp_cube, color="tab:green", marker="^", markersize=4, linewidth=1.4,
                   label=f"Local Cube dP ({dp_cube[-1]:.1f} Pa)")
        plotted_dp = True

    ax_dp.set_title("Duct & Local Cube Pressure Drop", fontsize=11, fontweight="bold")
    ax_dp.set_xlabel("Iteration")
    ax_dp.set_ylabel("Delta P [Pa]")
    if plotted_dp:
        ax_dp.legend(loc="best", fontsize=8.5)
    else:
        ax_dp.text(0.5, 0.5, "Waiting for pressure monitors...", ha="center", va="center", color="gray")

    # [1, 2] Continuity Imbalance (%) & yPlus status
    ax_imb = axes[1, 2]
    yp_patches = read_yplus_by_patch(get_all_data_files("fluid", "yPlus", ["yPlus.dat"]))
    yp_note = ""
    if "walls" in yp_patches and "fluid_to_solid" in yp_patches:
        yp_note = (f" | y+: walls={yp_patches['walls']['average'][-1]:.2f}, "
                   f"if={yp_patches['fluid_to_solid']['average'][-1]:.2f}")

    if m_in_tbl and m_out_tbl and "sum(phi)" in m_in_tbl and "sum(phi)" in m_out_tbl:
        n_m = min(len(m_in_tbl["Time"]), len(m_out_tbl["Time"]))
        phi_in = m_in_tbl["sum(phi)"][:n_m]
        phi_out = m_out_tbl["sum(phi)"][:n_m]
        err_pct = np.abs(phi_in + phi_out) / (np.abs(phi_in) + 1e-12) * 100.0
        ax_imb.semilogy(m_in_tbl["Time"][:n_m], np.clip(err_pct, 1e-10, None),
                        color="tab:purple", linewidth=1.6, label=r"$\varepsilon_m$ (%)")
        ax_imb.set_title(f"Continuity eps_m={err_pct[-1]:.2e}%{yp_note}", fontsize=10.5, fontweight="bold")
        ax_imb.legend(loc="upper right", fontsize=8.5)
    else:
        ax_imb.text(0.5, 0.5, "Waiting for massFlow...", ha="center", va="center", color="gray")
        ax_imb.set_title(f"Continuity Imbalance (%){yp_note}", fontsize=10.5, fontweight="bold")
    ax_imb.set_xlabel("Iteration")
    ax_imb.set_ylabel("Imbalance [%]")

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Part IV live convergence & CHT monitor.")
    parser.add_argument("--interval", type=float, default=1.5, help="Poll interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no GUI loop).")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save snapshot image.")
    args = parser.parse_args()

    if args.once:
        plt.switch_backend("Agg")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    if not args.once and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("OpenFOAM Part IV: Steady CHT Monitor (chtMultiRegionSimpleFoam)")

    if args.once:
        draw_dashboard(fig, axes)
        if args.save:
            fig.savefig(args.save, dpi=150)
        plt.close(fig)
        return

    def _animate(_frame):
        global INITIAL_DRAW
        if not INITIAL_DRAW and not has_new_data():
            return
        INITIAL_DRAW = False
        draw_dashboard(fig, axes)

    _anim = FuncAnimation(fig, _animate, interval=int(args.interval * 1000), cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
