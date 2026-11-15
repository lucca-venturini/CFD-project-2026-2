#!/usr/bin/env python3
"""
Real-time transient thermal, boiling-threshold, and energy-balance dashboard
for OpenFOAM Part V (chtMultiRegionFoam: fluid + solid regions).
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CASE_DIR = "."
POST_DIR = os.path.join(CASE_DIR, "postProcessing")

# Thermophysical constants matching OpenFOAM constant/{fluid,solid}/thermophysicalProperties
RHO_F = 1000.0          # kg/m^3
CP_F = 4184.0           # J/(kg K)
V_F = 0.059875          # m^3 (0.2*0.2*1.5 - 0.05^3)
M_DOT_NOMINAL = 40.0    # kg/s

RHO_S = 8000.0          # kg/m^3
CP_S = 500.0            # J/(kg K)
V_S = 1.25e-4           # m^3 (0.05^3)
Q_DOT = 2.68e10         # W/m^3
Q_GEN_W = Q_DOT * V_S   # 3.350e6 W (3.350 MW)
Q_GEN_MW = Q_GEN_W / 1e6
T0_K = 298.15           # Initial temperature [K]

# Water saturation temperature constants (Clausius-Clapeyron anchored at 101325 Pa -> 373.15 K)
P_ATM_PA = 101325.0
T_SAT_ATM_K = 373.15
R_VAPOR = 461.52        # J/(kg K)
H_FG = 2.2564e6         # J/kg

LAST_MTIMES = {}
INITIAL_DRAW = True


def compute_tsat_k(p_pa):
    """Computes water saturation temperature [K] via Clausius-Clapeyron around 1 atm."""
    p_safe = max(float(p_pa), 1.0e4)
    inv_t = (1.0 / T_SAT_ATM_K) - (R_VAPOR / H_FG) * np.log(p_safe / P_ATM_PA)
    return 1.0 / inv_t


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


def read_field_minmax(filepaths, target_field="T", return_locations=False):
    """Parses and merges fieldMinMax.dat for a target field across restart segments."""
    by_time = {}
    loc_by_time = {}
    tuple_re = re.compile(r"\(([^)]*)\)")

    for filepath in _as_file_list(filepaths):
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    locs = tuple_re.findall(line)
                    cleaned = tuple_re.sub("", line)
                    tokens = cleaned.split()
                    if len(tokens) >= 5 and tokens[1] == target_field:
                        t_val = float(tokens[0])
                        by_time[t_val] = (float(tokens[2]), float(tokens[4]))
                        if len(locs) >= 2:
                            coords = tuple(float(c) for c in locs[1].split()[:3])
                            if len(coords) == 3:
                                loc_by_time[t_val] = coords
        except (OSError, ValueError):
            continue

    if not by_time:
        return (None, None, None, None) if return_locations else (None, None, None)

    sorted_t = sorted(by_time.keys())
    vmins = np.array([by_time[t][0] for t in sorted_t], dtype=float)
    vmaxs = np.array([by_time[t][1] for t in sorted_t], dtype=float)
    t_arr = np.array(sorted_t, dtype=float)
    if return_locations:
        locs_arr = [loc_by_time.get(t, None) for t in sorted_t]
        return t_arr, vmins, vmaxs, locs_arr
    return t_arr, vmins, vmaxs


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


def has_new_data():
    """Checks whether any monitored Part V postProcessing file has updated."""
    changed = False
    targets = [
        ("fluid", "fluidTempExtrema", ["fieldMinMax.dat"]),
        ("fluid", "interfaceTempFluid", ["surfaceFieldValue.dat"]),
        ("fluid", "interfaceTempFluidMax", ["surfaceFieldValue.dat"]),
        ("fluid", "fluidTempAvg", ["volFieldValue.dat"]),
        ("fluid", "wallHeatFlux", ["wallHeatFlux.dat"]),
        ("fluid", "inletTemperatureMix", ["surfaceFieldValue.dat"]),
        ("fluid", "outletTemperatureMix", ["surfaceFieldValue.dat"]),
        ("fluid", "massFlowInlet", ["surfaceFieldValue.dat"]),
        ("fluid", "massFlowOutlet", ["surfaceFieldValue.dat"]),
        ("fluid", "inletPressure", ["surfaceFieldValue.dat"]),
        ("fluid", "outletPressure", ["surfaceFieldValue.dat"]),
        ("fluid", "courantExtrema", ["fieldMinMax.dat"]),
        ("fluid", "residualsFluid", ["solverInfo.dat"]),
        ("solid", "solidTempExtrema", ["fieldMinMax.dat"]),
        ("solid", "solidTempAvg", ["volFieldValue.dat"]),
        ("solid", "residualsSolid", ["solverInfo.dat"]),
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


def _time_derivative_mw(t_sec, energy_j):
    """Computes smooth dE/dt [MW] using central/one-sided numpy gradient."""
    if t_sec is None or len(t_sec) < 2:
        return None
    if np.any(np.diff(t_sec) <= 0):
        return None
    return np.gradient(energy_j, t_sec) / 1e6


def draw_dashboard(fig, axes):
    """Updates all 6 Part V transient CHT, boiling-threshold, and energy-balance panels."""
    for ax in axes.flat:
        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

    # Clear leftover twinx axes created in previous frames
    for extra_ax in fig.axes[6:]:
        extra_ax.remove()

    # Load primary transient streams
    t_fmax, _, f_tmax, f_tlocs = read_field_minmax(
        get_all_data_files("fluid", "fluidTempExtrema", ["fieldMinMax.dat"]),
        target_field="T",
        return_locations=True,
    )
    t_smax, _, s_tmax = read_field_minmax(
        get_all_data_files("solid", "solidTempExtrema", ["fieldMinMax.dat"]),
        target_field="T",
    )
    if_avg_tbl = read_numeric_table(get_all_data_files("fluid", "interfaceTempFluid", ["surfaceFieldValue.dat"]))
    if_max_tbl = read_numeric_table(get_all_data_files("fluid", "interfaceTempFluidMax", ["surfaceFieldValue.dat"]))
    s_avg_tbl = read_numeric_table(get_all_data_files("solid", "solidTempAvg", ["volFieldValue.dat"]))
    f_avg_tbl = read_numeric_table(get_all_data_files("fluid", "fluidTempAvg", ["volFieldValue.dat"]))
    t_whf, q_int_w = read_wall_heat_flux(get_all_data_files("fluid", "wallHeatFlux", ["wallHeatFlux.dat"]))
    t_in_tbl = read_numeric_table(get_all_data_files("fluid", "inletTemperatureMix", ["surfaceFieldValue.dat"]))
    t_out_tbl = read_numeric_table(get_all_data_files("fluid", "outletTemperatureMix", ["surfaceFieldValue.dat"]))
    m_in_tbl = read_numeric_table(get_all_data_files("fluid", "massFlowInlet", ["surfaceFieldValue.dat"]))
    m_out_tbl = read_numeric_table(get_all_data_files("fluid", "massFlowOutlet", ["surfaceFieldValue.dat"]))
    p_in_tbl = read_numeric_table(get_all_data_files("fluid", "inletPressure", ["surfaceFieldValue.dat"]))
    p_out_tbl = read_numeric_table(get_all_data_files("fluid", "outletPressure", ["surfaceFieldValue.dat"]))
    t_co, _, co_max = read_field_minmax(
        get_all_data_files("fluid", "courantExtrema", ["fieldMinMax.dat"]),
        target_field="Co",
    )

    # Determine local saturation temperature from interface max pressure (fallback ~101.85 kPa -> 373.29 K)
    p_local_pa = 101850.0
    if if_max_tbl and "max(p)" in if_max_tbl and len(if_max_tbl["max(p)"]) > 0:
        p_local_pa = float(if_max_tbl["max(p)"][-1])
    elif p_out_tbl and "areaAverage(p)" in p_out_tbl and len(p_out_tbl["areaAverage(p)"]) > 0:
        p_local_pa = float(p_out_tbl["areaAverage(p)"][-1])
    t_sat_k = compute_tsat_k(p_local_pa)

    # [0, 0] Max Fluid & Interface Temperatures vs Time [ms] with Boiling Threshold
    ax_boil = axes[0, 0]
    ax_boil.axhline(t_sat_k, color="crimson", linestyle="--", linewidth=1.6,
                    label=f"T_sat(p) = {t_sat_k:.2f} K ({t_sat_k - 273.15:.2f} C)")

    plotted_boil = False
    boil_title_note = f"T_sat = {t_sat_k:.2f} K"
    if t_fmax is not None and len(t_fmax) > 0:
        t_ms = t_fmax * 1e3
        ax_boil.plot(t_ms, f_tmax, color="tab:red", linewidth=1.8,
                     label=f"Fluid T_max ({f_tmax[-1]:.2f} K)")
        plotted_boil = True
        idx_cross = np.where(f_tmax >= t_sat_k)[0]
        if len(idx_cross) > 0:
            ic = int(idx_cross[0])
            t_b = t_fmax[ic]
            # Linear interpolation when preceding point exists below threshold
            if ic > 0 and f_tmax[ic] > f_tmax[ic - 1]:
                frac = (t_sat_k - f_tmax[ic - 1]) / (f_tmax[ic] - f_tmax[ic - 1])
                t_b = t_fmax[ic - 1] + frac * (t_fmax[ic] - t_fmax[ic - 1])
            loc = f_tlocs[ic] if f_tlocs and ic < len(f_tlocs) else None
            ax_boil.axvline(t_b * 1e3, color="crimson", linestyle=":", linewidth=1.2)
            ax_boil.plot(t_b * 1e3, t_sat_k, "o", color="crimson", markersize=6)
            loc_str = f"\n@ ({loc[0]*1e3:.1f}, {loc[1]*1e3:.1f}, {loc[2]*1e3:.1f}) mm" if loc else ""
            ax_boil.annotate(f"t_boil = {t_b*1e3:.2f} ms{loc_str}",
                             xy=(t_b * 1e3, t_sat_k),
                             xytext=(min(t_b * 1e3 + 2.5, max(t_ms[-1] * 0.65, t_b * 1e3 + 2.0)), t_sat_k - 14.0),
                             fontsize=8, fontweight="bold", color="crimson",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.85),
                             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))
            boil_title_note = f"t_boil = {t_b*1e3:.2f} ms"

    if if_max_tbl and "max(T)" in if_max_tbl and len(if_max_tbl["max(T)"]) > 0:
        ax_boil.plot(if_max_tbl["Time"] * 1e3, if_max_tbl["max(T)"], color="tab:orange",
                     linestyle="-.", linewidth=1.4, label=f"Interface T_max ({if_max_tbl['max(T)'][-1]:.2f} K)")
        plotted_boil = True
    if if_avg_tbl and "areaAverage(T)" in if_avg_tbl and len(if_avg_tbl["areaAverage(T)"]) > 0:
        ax_boil.plot(if_avg_tbl["Time"] * 1e3, if_avg_tbl["areaAverage(T)"], color="tab:blue",
                     linewidth=1.5, label=f"Interface T_avg ({if_avg_tbl['areaAverage(T)'][-1]:.2f} K)")
        plotted_boil = True

    ax_boil.set_title(f"Fluid & Interface Temperatures ({boil_title_note})", fontsize=11, fontweight="bold")
    ax_boil.set_xlabel("Physical Time t [ms]")
    ax_boil.set_ylabel("Temperature [K]")
    if plotted_boil:
        ax_boil.legend(loc="upper left", fontsize=8)
    else:
        ax_boil.text(0.5, 0.5, "Waiting for fluidTempExtrema...", ha="center", va="center", color="gray",
                     transform=ax_boil.transAxes)

    # [0, 1] Cube Solid Temperatures vs Time [ms] + Adiabatic Slope Reference
    ax_sol = axes[0, 1]
    plotted_sol = False
    t_ref_sec = t_smax if t_smax is not None else (s_avg_tbl["Time"] if s_avg_tbl else None)
    if t_ref_sec is not None and len(t_ref_sec) > 0:
        t_adiab = T0_K + (Q_DOT / (RHO_S * CP_S)) * t_ref_sec
        ax_sol.plot(t_ref_sec * 1e3, t_adiab, color="black", linestyle=":", linewidth=1.3,
                    label=f"Adiabatic slope (6.70 K/ms)")
    if t_smax is not None and len(t_smax) > 0:
        ax_sol.plot(t_smax * 1e3, s_tmax, color="tab:red", linewidth=1.7,
                    label=f"Cube T_max ({s_tmax[-1]:.2f} K)")
        plotted_sol = True
    if s_avg_tbl and "volAverage(T)" in s_avg_tbl and len(s_avg_tbl["volAverage(T)"]) > 0:
        ax_sol.plot(s_avg_tbl["Time"] * 1e3, s_avg_tbl["volAverage(T)"], color="tab:orange", linewidth=1.6,
                    label=f"Cube T_volAvg ({s_avg_tbl['volAverage(T)'][-1]:.2f} K)")
        plotted_sol = True
    if if_avg_tbl and "areaAverage(T)" in if_avg_tbl and len(if_avg_tbl["areaAverage(T)"]) > 0:
        ax_sol.plot(if_avg_tbl["Time"] * 1e3, if_avg_tbl["areaAverage(T)"], color="tab:blue", linewidth=1.5,
                    label=f"Interface T_avg ({if_avg_tbl['areaAverage(T)'][-1]:.2f} K)")
        plotted_sol = True

    ax_sol.set_title("Solid Cube Thermal Response", fontsize=11, fontweight="bold")
    ax_sol.set_xlabel("Physical Time t [ms]")
    ax_sol.set_ylabel("Temperature [K]")
    if plotted_sol:
        ax_sol.legend(loc="upper left", fontsize=8)
    else:
        ax_sol.text(0.5, 0.5, "Waiting for solid temperature data...", ha="center", va="center", color="gray",
                    transform=ax_sol.transAxes)

    # Shared stored thermal energies E_solid [J] and E_fluid [J]
    t_es, e_s_j, dEs_dt_mw = None, None, None
    if s_avg_tbl and "volAverage(T)" in s_avg_tbl and len(s_avg_tbl["volAverage(T)"]) > 0:
        t_es = s_avg_tbl["Time"]
        e_s_j = RHO_S * CP_S * V_S * (s_avg_tbl["volAverage(T)"] - T0_K)
        dEs_dt_mw = _time_derivative_mw(t_es, e_s_j)

    t_ef, e_f_j, dEf_dt_mw = None, None, None
    if f_avg_tbl and "volAverage(T)" in f_avg_tbl and len(f_avg_tbl["volAverage(T)"]) > 0:
        t_ef = f_avg_tbl["Time"]
        e_f_j = RHO_F * CP_F * V_F * (f_avg_tbl["volAverage(T)"] - T0_K)
        dEf_dt_mw = _time_derivative_mw(t_ef, e_f_j)

    m_dot_val = M_DOT_NOMINAL
    if m_in_tbl and "sum(phi)" in m_in_tbl and len(m_in_tbl["sum(phi)"]) > 0:
        m_dot_val = float(np.abs(m_in_tbl["sum(phi)"][-1]))

    # [0, 2] Transient Energy Balance [MW]
    ax_eb = axes[0, 2]
    ax_eb.axhline(Q_GEN_MW, color="black", linestyle="-", linewidth=1.4,
                  label=f"Q_generated ({Q_GEN_MW:.3f} MW)")

    if t_whf is not None and len(t_whf) > 0:
        q_int_mw = q_int_w / 1e6
        ax_eb.plot(t_whf * 1e3, q_int_mw, color="tab:red", linewidth=1.6,
                   label=f"Q_interface ({q_int_mw[-1]:.3f} MW)")

    q_rem_mw, t_qrem = None, None
    if (t_in_tbl and t_out_tbl and "weightedAverage(T)" in t_in_tbl
            and "weightedAverage(T)" in t_out_tbl):
        n_q = min(len(t_in_tbl["Time"]), len(t_out_tbl["Time"]))
        t_qrem = t_out_tbl["Time"][:n_q]
        dT_mix = t_out_tbl["weightedAverage(T)"][:n_q] - t_in_tbl["weightedAverage(T)"][:n_q]
        q_rem_mw = (m_dot_val * CP_F * dT_mix) / 1e6
        ax_eb.plot(t_qrem * 1e3, q_rem_mw, color="tab:blue", linewidth=1.5,
                   label=f"Q_removed ({q_rem_mw[-1]:.3f} MW)")

    eps_num_str = "--"
    if dEs_dt_mw is not None and dEf_dt_mw is not None:
        n_e = min(len(t_es), len(t_ef))
        t_e = t_es[:n_e]
        dE_tot_mw = dEs_dt_mw[:n_e] + dEf_dt_mw[:n_e]
        ax_eb.plot(t_e * 1e3, dE_tot_mw, color="tab:green", linestyle="--", linewidth=1.5,
                   label=f"dE_s/dt + dE_f/dt ({dE_tot_mw[-1]:.3f} MW)")
        if q_rem_mw is not None and t_qrem is not None:
            q_rem_interp = np.interp(t_e, t_qrem, q_rem_mw)
            q_acc_mw = dE_tot_mw + q_rem_interp
            ax_eb.plot(t_e * 1e3, q_acc_mw, color="tab:purple", linestyle=":", linewidth=1.6,
                       label=f"Q_rem + dE/dt ({q_acc_mw[-1]:.3f} MW)")
            eps_num = abs(Q_GEN_MW - q_acc_mw[-1]) / Q_GEN_MW * 100.0
            eps_num_str = f"{eps_num:.2f}%"

    ax_eb.set_title(f"Transient Energy Balance (eps_num = {eps_num_str})", fontsize=11, fontweight="bold")
    ax_eb.set_xlabel("Physical Time t [ms]")
    ax_eb.set_ylabel("Thermal Power [MW]")
    ax_eb.legend(loc="best", fontsize=7.8)

    # [1, 0] Stored Thermal Energy E_solid(t), E_fluid(t), & Cumulative Generated Energy [kJ]
    ax_en = axes[1, 0]
    plotted_en = False
    t_cum_s = t_es if t_es is not None else t_ef
    if t_cum_s is not None and len(t_cum_s) > 0:
        e_cum_kj = (Q_GEN_W * t_cum_s) / 1e3
        ax_en.plot(t_cum_s * 1e3, e_cum_kj, color="black", linestyle="--", linewidth=1.4,
                   label=f"Integral Q_gen dt ({e_cum_kj[-1]:.2f} kJ)")
        plotted_en = True
    if e_s_j is not None:
        e_s_kj = e_s_j / 1e3
        ax_en.plot(t_es * 1e3, e_s_kj, color="tab:orange", linewidth=1.7,
                   label=f"E_solid ({e_s_kj[-1]:.2f} kJ)")
        plotted_en = True
    if e_f_j is not None:
        e_f_kj = e_f_j / 1e3
        ax_en.plot(t_ef * 1e3, e_f_kj, color="tab:blue", linewidth=1.7,
                   label=f"E_fluid ({e_f_kj[-1]:.2f} kJ)")
        plotted_en = True
    if e_s_j is not None and e_f_j is not None:
        n_e = min(len(t_es), len(t_ef))
        e_tot_kj = (e_s_j[:n_e] + e_f_j[:n_e]) / 1e3
        ax_en.plot(t_es[:n_e] * 1e3, e_tot_kj, color="tab:green", linestyle="-.", linewidth=1.5,
                   label=f"E_solid + E_fluid ({e_tot_kj[-1]:.2f} kJ)")

    ax_en.set_title("Cumulative & Stored Thermal Energy", fontsize=11, fontweight="bold")
    ax_en.set_xlabel("Physical Time t [ms]")
    ax_en.set_ylabel("Energy [kJ]")
    if plotted_en:
        ax_en.legend(loc="upper left", fontsize=8)
    else:
        ax_en.text(0.5, 0.5, "Waiting for volFieldValue data...", ha="center", va="center", color="gray",
                   transform=ax_en.transAxes)

    # [1, 1] Outlet Mixed-Mean Temperature Rise & Local Absolute Pressure
    ax_p = axes[1, 1]
    plotted_dt = False
    lines_11, labels_11 = [], []
    if (t_in_tbl and t_out_tbl and "weightedAverage(T)" in t_in_tbl
            and "weightedAverage(T)" in t_out_tbl):
        n_q = min(len(t_in_tbl["Time"]), len(t_out_tbl["Time"]))
        t_q = t_out_tbl["Time"][:n_q]
        dT_mix = t_out_tbl["weightedAverage(T)"][:n_q] - t_in_tbl["weightedAverage(T)"][:n_q]
        l1, = ax_p.plot(t_q * 1e3, dT_mix, color="tab:orange", linewidth=1.7,
                        label=f"Delta T_mix ({dT_mix[-1]:.3f} K)")
        lines_11.append(l1)
        labels_11.append(l1.get_label())
        plotted_dt = True

    ax_p_r = ax_p.twinx()
    ax_p_r.grid(False)
    plotted_pr = False
    if if_max_tbl and "max(p)" in if_max_tbl and len(if_max_tbl["max(p)"]) > 0:
        l2, = ax_p_r.plot(if_max_tbl["Time"] * 1e3, if_max_tbl["max(p)"] / 1e3, color="tab:blue",
                          linestyle="-", linewidth=1.4,
                          label=f"p_if,max ({if_max_tbl['max(p)'][-1]/1e3:.2f} kPa)")
        lines_11.append(l2)
        labels_11.append(l2.get_label())
        plotted_pr = True
    if p_in_tbl and "areaAverage(p)" in p_in_tbl and len(p_in_tbl["areaAverage(p)"]) > 0:
        l3, = ax_p_r.plot(p_in_tbl["Time"] * 1e3, p_in_tbl["areaAverage(p)"] / 1e3, color="tab:cyan",
                          linestyle="--", linewidth=1.2,
                          label=f"p_inlet ({p_in_tbl['areaAverage(p)'][-1]/1e3:.2f} kPa)")
        lines_11.append(l3)
        labels_11.append(l3.get_label())
        plotted_pr = True

    ax_p.set_title("Outlet Delta T_mix & Local Pressure", fontsize=11, fontweight="bold")
    ax_p.set_xlabel("Physical Time t [ms]")
    ax_p.set_ylabel("T_out,mix - T_in,mix [K]", color="tab:orange")
    ax_p_r.set_ylabel("Absolute Pressure p [kPa]", color="tab:blue")
    if plotted_dt or plotted_pr:
        ax_p.legend(lines_11, labels_11, loc="best", fontsize=8)
    else:
        ax_p.text(0.5, 0.5, "Waiting for outlet/pressure data...", ha="center", va="center", color="gray",
                  transform=ax_p.transAxes)

    # [1, 2] Courant Number, Solver Residuals, & Continuity Imbalance
    ax_num = axes[1, 2]
    res_f = read_solver_residuals(get_all_data_files("fluid", "residualsFluid", ["solverInfo.dat"]))
    res_s = read_solver_residuals(get_all_data_files("solid", "residualsSolid", ["solverInfo.dat"]))

    lines_12, labels_12 = [], []
    if res_f and "Time" in res_f:
        t_ms_r = res_f["Time"] * 1e3
        for key, lbl, col, ls in [
            ("Ux", "Ux", "tab:blue", "-"),
            ("Uz", "Uz", "tab:green", "-."),
            ("p_rgh", "p_rgh", "tab:red", "-"),
            ("h", "h (fluid)", "tab:orange", "-"),
        ]:
            if key in res_f and len(res_f[key]) > 0:
                l, = ax_num.semilogy(t_ms_r, np.clip(res_f[key], 1e-12, None),
                                     label=lbl, color=col, linestyle=ls, linewidth=1.2)
                lines_12.append(l)
                labels_12.append(lbl)
    if res_s and "h" in res_s and len(res_s["h"]) > 0:
        l, = ax_num.semilogy(res_s["Time"] * 1e3, np.clip(res_s["h"], 1e-12, None),
                             label="h (solid)", color="black", linestyle="--", linewidth=1.3)
        lines_12.append(l)
        labels_12.append("h (solid)")

    eps_m_note = "--"
    if m_in_tbl and m_out_tbl and "sum(phi)" in m_in_tbl and "sum(phi)" in m_out_tbl:
        n_m = min(len(m_in_tbl["Time"]), len(m_out_tbl["Time"]))
        phi_in = m_in_tbl["sum(phi)"][:n_m]
        phi_out = m_out_tbl["sum(phi)"][:n_m]
        err_pct = np.abs(phi_in + phi_out) / (np.abs(phi_in) + 1e-12) * 100.0
        l, = ax_num.semilogy(m_in_tbl["Time"][:n_m] * 1e3, np.clip(err_pct, 1e-10, None),
                             color="tab:purple", linestyle=":", linewidth=1.5,
                             label=f"eps_m ({err_pct[-1]:.1e}%)")
        lines_12.append(l)
        labels_12.append(l.get_label())
        eps_m_note = f"{err_pct[-1]:.1e}%"

    ax_co = ax_num.twinx()
    ax_co.grid(False)
    co_note = "--"
    if t_co is not None and len(t_co) > 0:
        l, = ax_co.plot(t_co * 1e3, co_max, color="tab:brown", linewidth=1.6,
                        label=f"Co_max ({co_max[-1]:.2f})")
        lines_12.append(l)
        labels_12.append(l.get_label())
        ax_co.set_ylim(0, max(6.5, float(np.nanmax(co_max)) * 1.15))
        co_note = f"{co_max[-1]:.2f}"

    ax_num.set_title(f"Co_max={co_note} | Residuals & eps_m={eps_m_note}", fontsize=10.5, fontweight="bold")
    ax_num.set_xlabel("Physical Time t [ms]")
    ax_num.set_ylabel("Initial Residual / eps_m [%]")
    ax_co.set_ylabel("Max Courant Number Co_max", color="tab:brown")
    if lines_12:
        ax_num.legend(lines_12, labels_12, loc="best", ncol=2, fontsize=7.5)
    else:
        ax_num.text(0.5, 0.5, "Waiting for Co / solverInfo...", ha="center", va="center", color="gray",
                    transform=ax_num.transAxes)

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Part V live transient CHT monitor.")
    parser.add_argument("--interval", type=float, default=1.5, help="Poll interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no GUI loop).")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save snapshot image.")
    args = parser.parse_args()

    if args.once:
        plt.switch_backend("Agg")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    if not args.once and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("OpenFOAM Part V: Transient CHT Monitor (chtMultiRegionFoam)")

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
