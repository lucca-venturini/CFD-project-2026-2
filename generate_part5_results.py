#!/usr/bin/env python3
"""
Post-Processing & Quantitative Analysis for Part V:
Transient Conjugate Heat Transfer (chtMultiRegionFoam).

Parses postProcessing tables, probe histories, solver logs, and reconstructed 3D
binary fields over t in [0, 0.020] s (400 time steps, dt = 50 us), verifies the
Clausius-Clapeyron boiling-temperature threshold T_sat(p), evaluates the transient
energy balance and stored thermal energies, exports CSV/TXT summaries to
`part5/results/` and `data/`, and generates publication-ready figures in `figures/`.
"""

import os
import re
import shutil
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & CALCULATION BASIS
# -----------------------------------------------------------------------------
W, H, L_DUCT = 0.20, 0.20, 1.50
A_DUCT = W * H                      # 0.040 m^2
U_BULK = 1.00                       # m/s
RHO_F = 1000.0                      # kg/m^3 (Calculation basis & thermophysicalProperties)
CP_F = 4184.0                       # J/(kg K)
MU_F = 8.90e-04                     # Pa s
PR_F = 6.135                        # Prandtl number
KAPPA_F = MU_F * CP_F / PR_F        # ~0.606970 W/(m K)
ALPHA_F = KAPPA_F / (RHO_F * CP_F)  # m^2/s
V_F = 0.20 * 0.20 * 1.50 - 0.05**3  # 0.059875 m^3
M_DOT_NOM = RHO_F * U_BULK * A_DUCT # 40.0 kg/s

H_CUBE = 0.05
V_S = H_CUBE ** 3                   # 1.25e-4 m^3
A_INT = 5.0 * (H_CUBE ** 2)         # 0.0125 m^2
RHO_S = 8000.0                      # kg/m^3
CP_S = 500.0                        # J/(kg K)
KAPPA_S = 50.0                      # W/(m K)
ALPHA_S = KAPPA_S / (RHO_S * CP_S)  # 1.25e-5 m^2/s

Q_DOT = 2.68e10                     # W/m^3
Q_GEN = Q_DOT * V_S                 # 3.3500e6 W (3.350 MW)
T0_K = 298.15                       # Initial uniform temperature [K]

# Water saturation relation (Clausius-Clapeyron anchored at 101325 Pa -> 373.15 K)
P_ATM = 101325.0                    # Pa
T_SAT_ATM = 373.15                  # K (100.00 C)
R_VAPOR = 461.52                    # J/(kg K)
H_FG = 2.2564e6                     # J/kg

NX, NY, NZ = 104, 96, 156
N_FLUID = 1536768
N_SOLID = 20736

CASE_DIR = "part5"
POST_DIR = os.path.join(CASE_DIR, "postProcessing")
DATA_DIR = "data"
FIG_DIR = "figures"
RES_DIR = os.path.join(CASE_DIR, "results")

for d in [DATA_DIR, FIG_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.2,
    "axes.labelsize": 9.8,
    "axes.titlesize": 10.2,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "figure.dpi": 300,
})


def compute_tsat_k(p_pa):
    """Water saturation temperature [K] via Clausius-Clapeyron around 1 atm."""
    inv_t = (1.0 / T_SAT_ATM) - (R_VAPOR / H_FG) * np.log(p_pa / P_ATM)
    return 1.0 / inv_t


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        out_f = os.path.join(FIG_DIR, f"{name}.{ext}")
        out_r = os.path.join(RES_DIR, f"{name}.{ext}")
        fig.savefig(out_f, dpi=300, bbox_inches="tight")
        shutil.copy2(out_f, out_r)
    plt.close(fig)
    print(f"    -> Saved {name}.(pdf|png)")


def read_field_minmax_file(filepath, target_field):
    """Parses OpenFOAM fieldMinMax.dat returning t, vmin, vmax, loc_min, loc_max."""
    tuple_re = re.compile(r"\(([^)]*)\)")
    t_vals, vmins, vmaxs, lmins, lmaxs = [], [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            locs = tuple_re.findall(s)
            toks = tuple_re.sub("", s).split()
            if len(toks) >= 5 and toks[1] == target_field:
                t_vals.append(float(toks[0]))
                vmins.append(float(toks[2]))
                vmaxs.append(float(toks[4]))
                lmin = tuple(float(c) for c in locs[0].split()[:3]) if len(locs) >= 1 else (np.nan, np.nan, np.nan)
                lmax = tuple(float(c) for c in locs[1].split()[:3]) if len(locs) >= 2 else (np.nan, np.nan, np.nan)
                lmins.append(lmin)
                lmaxs.append(lmax)
    return (
        np.array(t_vals, dtype=float),
        np.array(vmins, dtype=float),
        np.array(vmaxs, dtype=float),
        np.array(lmins, dtype=float),
        np.array(lmaxs, dtype=float),
    )


def read_wall_heat_flux(filepath, patch_name="fluid_to_solid"):
    t_vals, mins, maxs, integs = [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) >= 5 and p[1] == patch_name:
                t_vals.append(float(p[0]))
                mins.append(float(p[2]))
                maxs.append(float(p[3]))
                integs.append(float(p[4]))
    return np.array(t_vals), np.array(mins), np.array(maxs), np.array(integs)


def build_1d_coords(segments):
    edges = [0.0]
    for length_mm, n_cells, ratio in segments:
        seg_l = length_mm * 1e-3
        if abs(ratio - 1.0) < 1e-9:
            dxs = np.full(n_cells, seg_l / n_cells)
        else:
            r = ratio ** (1.0 / (n_cells - 1))
            dx0 = seg_l * (1.0 - r) / (1.0 - r ** n_cells)
            dxs = dx0 * (r ** np.arange(n_cells))
        for dx in dxs:
            edges.append(edges[-1] + dx)
    edges = np.array(edges)
    return 0.5 * (edges[:-1] + edges[1:]), edges


def read_of_binary_scalar(filepath, n_cells):
    with open(filepath, "rb") as f:
        data = f.read()
    marker = f"internalField   nonuniform List<scalar> \n{n_cells}\n(".encode()
    pos = data.find(marker)
    if pos == -1:
        raise RuntimeError(f"Could not find binary internalField marker in {filepath}")
    return np.frombuffer(data, dtype="<f8", count=n_cells, offset=pos + len(marker)).copy()


def first_crossing(t_arr, y_arr, y_thresh):
    """Returns linearly interpolated crossing time and bracket index, or (None, None)."""
    hits = np.where(y_arr >= y_thresh)[0]
    if len(hits) == 0:
        return None, None
    idx = int(hits[0])
    if idx == 0:
        return float(t_arr[0]), 0
    y0, y1 = y_arr[idx - 1], y_arr[idx]
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    frac = (y_thresh - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(t0 + frac * (t1 - t0)), idx


# -----------------------------------------------------------------------------
# 2. PARSE TIME-STEP MONITORS, LOGS, PROBES & RECONSTRUCTED 3D FIELDS
# -----------------------------------------------------------------------------
print("[*] Parsing Part V postProcessing histories and solver log...")

t_f, f_tmin, f_tmax, _, f_tmax_loc = read_field_minmax_file(
    os.path.join(POST_DIR, "fluid/fluidTempExtrema/0/fieldMinMax.dat"), "T"
)
_, f_pmin, f_pmax, _, _ = read_field_minmax_file(
    os.path.join(POST_DIR, "fluid/fluidTempExtrema/0/fieldMinMax.dat"), "p"
)
_, s_tmin, s_tmax, _, s_tmax_loc = read_field_minmax_file(
    os.path.join(POST_DIR, "solid/solidTempExtrema/0/fieldMinMax.dat"), "T"
)
_, co_min, co_max, _, co_max_loc = read_field_minmax_file(
    os.path.join(POST_DIR, "fluid/courantExtrema/0/fieldMinMax.dat"), "Co"
)

if_avg_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/interfaceTempFluid/0/surfaceFieldValue.dat"), comments="#")
if_max_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/interfaceTempFluidMax/0/surfaceFieldValue.dat"), comments="#")
if_sol_tbl = np.loadtxt(os.path.join(POST_DIR, "solid/interfaceTempSolid/0/surfaceFieldValue.dat"), comments="#")
s_avg_tbl = np.loadtxt(os.path.join(POST_DIR, "solid/solidTempAvg/0/volFieldValue.dat"), comments="#")
f_avg_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/fluidTempAvg/0/volFieldValue.dat"), comments="#")
t_in_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/inletTemperatureMix/0/surfaceFieldValue.dat"), comments="#")
t_out_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/outletTemperatureMix/0/surfaceFieldValue.dat"), comments="#")
m_in_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/massFlowInlet/0/surfaceFieldValue.dat"), comments="#")
m_out_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/massFlowOutlet/0/surfaceFieldValue.dat"), comments="#")
p_in_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/inletPressure/0/surfaceFieldValue.dat"), comments="#")
p_out_tbl = np.loadtxt(os.path.join(POST_DIR, "fluid/outletPressure/0/surfaceFieldValue.dat"), comments="#")
_, whf_min, whf_max, q_if = read_wall_heat_flux(
    os.path.join(POST_DIR, "fluid/wallHeatFlux/0/wallHeatFlux.dat"), "fluid_to_solid"
)
_, _, _, q_walls = read_wall_heat_flux(
    os.path.join(POST_DIR, "fluid/wallHeatFlux/0/wallHeatFlux.dat"), "walls"
)
fp_T = np.loadtxt(os.path.join(POST_DIR, "fluidProbes/fluid/0/T"), comments="#")
fp_p = np.loadtxt(os.path.join(POST_DIR, "fluidProbes/fluid/0/p"), comments="#")
sp_T = np.loadtxt(os.path.join(POST_DIR, "solidProbes/solid/0/T"), comments="#")
def read_last_solver_residuals(filepath):
    """Returns dict of column->float from the final row of solverInfo.dat."""
    headers = []
    last_row = None
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s.lstrip("#").strip().split()
                if toks and toks[0].lower() == "time":
                    headers = toks
            else:
                last_row = s.split()
    out = {}
    if headers and last_row and len(headers) == len(last_row):
        for h, v in zip(headers, last_row):
            try:
                out[h] = float(v)
            except ValueError:
                pass
    return out


res_f_last = read_last_solver_residuals(os.path.join(POST_DIR, "fluid/residualsFluid/0/solverInfo.dat"))
res_s_last = read_last_solver_residuals(os.path.join(POST_DIR, "solid/residualsSolid/0/solverInfo.dat"))

# Log parsing: Courant & Solid Diffusion Numbers per step
log_co_mean, log_co_max, log_di_mean, log_di_max = [], [], [], []
with open(os.path.join(CASE_DIR, "log.chtMultiRegionFoam"), "r") as f:
    for line in f:
        if line.startswith("Region: fluid Courant Number mean:"):
            p = line.split()
            log_co_mean.append(float(p[5]))
            log_co_max.append(float(p[7]))
        elif line.startswith("Region: solid Diffusion Number mean:"):
            p = line.split()
            log_di_mean.append(float(p[5]))
            log_di_max.append(float(p[7]))

log_co_mean = np.array(log_co_mean[1:])  # drop t=0 pre-step duplicate if present (401 -> 400)
log_co_max = np.array(log_co_max[1:])
log_di_mean = np.array(log_di_mean[1:])
log_di_max = np.array(log_di_max[1:])
if len(log_di_max) != len(t_f):
    log_co_mean = log_co_mean[-len(t_f):]
    log_co_max = log_co_max[-len(t_f):]
    log_di_mean = log_di_mean[-len(t_f):]
    log_di_max = log_di_max[-len(t_f):]

# Energy balance quantities at every time step
T_if_avg = if_avg_tbl[:, 1]
T_if_max = if_max_tbl[:, 1]
p_if_max = if_max_tbl[:, 2]
T_s_avg = s_avg_tbl[:, 1]
h_s_avg = s_avg_tbl[:, 2]
T_f_avg = f_avg_tbl[:, 1]
h_f_avg = f_avg_tbl[:, 2]
T_in_mix = t_in_tbl[:, 1]
T_out_mix = t_out_tbl[:, 1]
phi_in = m_in_tbl[:, 1]
phi_out = m_out_tbl[:, 1]
eps_m_pct = np.abs(phi_in + phi_out) / np.abs(phi_in) * 100.0
p_in_avg = p_in_tbl[:, 1]
p_out_avg = p_out_tbl[:, 1]

Q_rem = np.abs(phi_in) * CP_F * (T_out_mix - T_in_mix)
E_solid = RHO_S * CP_S * V_S * (T_s_avg - T0_K)
E_fluid = RHO_F * V_F * h_f_avg  # sensible enthalpy integral (avoids T0=298.15 float truncation)
E_gen = Q_GEN * t_f
dEs_dt = np.gradient(E_solid, t_f)
dEf_dt = np.gradient(E_fluid, t_f)
_hann = np.hanning(25)
_hann /= _hann.sum()
dEf_dt_smooth = np.convolve(np.pad(dEf_dt, 12, mode="edge"), _hann, mode="valid")

# Prepend t=0 baseline for clean trapezoidal integration of interface heat flux
t_with_0 = np.concatenate([[0.0], t_f])
q_if_with_0 = np.concatenate([[0.0], q_if])
dt_steps = np.diff(t_with_0)
E_if_cum = np.cumsum(0.5 * (q_if_with_0[:-1] + q_if_with_0[1:]) * dt_steps)

cum_imbalance_pct = np.abs(E_gen - (E_solid + E_fluid)) / E_gen * 100.0
inst_imbalance_pct = np.abs(Q_GEN - (Q_rem + dEs_dt + dEf_dt)) / Q_GEN * 100.0

# -----------------------------------------------------------------------------
# 3. ANALYZE 3D RECONSTRUCTED SNAPSHOTS & WALL-NORMAL PROFILE AT HOTSPOT
# -----------------------------------------------------------------------------
print("[*] Reconstructing structured grid coordinates and loading 3D snapshots...")
xc, xe = build_1d_coords([
    (37.5, 20, 329.450451), (37.5, 20, 0.002454),
    (25.0, 12, 423.558592), (25.0, 12, 0.002361),
    (37.5, 20, 407.485210), (37.5, 20, 0.003035),
])
yc, ye = build_1d_coords([
    (25.0, 18, 228.659048), (25.0, 18, 0.003529),
    (75.0, 30, 593.787261), (75.0, 30, 0.002082),
])
zc, ze = build_1d_coords([
    (350.0, 28, 1.0), (50.0, 22, 0.001941),
    (25.0, 12, 423.558592), (25.0, 12, 0.002361),
    (50.0, 22, 515.187605), (350.0, 35, 1.0), (650.0, 25, 2.2),
])

Z_grid, Y_grid, X_grid = np.meshgrid(zc, yc, xc, indexing="ij")
in_solid = (X_grid > 0.075) & (X_grid < 0.125) & (Y_grid < 0.050) & (Z_grid > 0.400) & (Z_grid < 0.450)
Xf, Yf, Zf = X_grid[~in_solid], Y_grid[~in_solid], Z_grid[~in_solid]

snap_times = np.arange(0.002, 0.0201, 0.002)
snap_cell_tmax = []
snap_cell_pmaxT = []
snap_cell_loc = []

# Hotspot row indices in full (NZ, NY, NX) tensor grid:
# Boiling hotspot cell: (x=0.074988, y=0.00056437, z=0.4487714) -> find nearest iy, iz in solid/fluid
iy_hot = int(np.argmin(np.abs(yc - 0.00056437)))
iz_hot = int(np.argmin(np.abs(zc - 0.44877143)))
# Build map from full (iz, iy, ix) to fluid/solid 1D cell index
fluid_idx_grid = np.full((NZ, NY, NX), -1, dtype=int)
solid_idx_grid = np.full((NZ, NY, NX), -1, dtype=int)
fluid_idx_grid[~in_solid] = np.arange(N_FLUID)
solid_idx_grid[in_solid] = np.arange(N_SOLID)

# Extract x-indices around left cube interface x = 0.075 m (ix = 28..51)
ix_fluid_line = np.arange(28, 40)  # x < 0.075 (fluid side)
ix_solid_line = np.arange(40, 52)  # x > 0.075 (solid side)
dx_f0 = 0.075 - xc[39]
dx_s0 = xc[40] - 0.075
g_f = KAPPA_F / dx_f0
g_s = KAPPA_S / dx_s0
delta_n_mm = np.concatenate([
    (xc[ix_fluid_line] - 0.075) * 1e3,
    [0.0],
    (xc[ix_solid_line] - 0.075) * 1e3,
])
wall_normal_profiles = {}

for ts in snap_times:
    tstr = f"{ts:g}"
    tf = read_of_binary_scalar(os.path.join(CASE_DIR, tstr, "fluid/T"), N_FLUID)
    pf = read_of_binary_scalar(os.path.join(CASE_DIR, tstr, "fluid/p"), N_FLUID)
    tsol = read_of_binary_scalar(os.path.join(CASE_DIR, tstr, "solid/T"), N_SOLID)
    imax = int(np.argmax(tf))
    snap_cell_tmax.append(float(tf[imax]))
    snap_cell_pmaxT.append(float(pf[imax]))
    snap_cell_loc.append((float(Xf[imax]), float(Yf[imax]), float(Zf[imax])))

    if round(ts * 1e3) in (4, 8, 12, 16, 20):
        f_ids = fluid_idx_grid[iz_hot, iy_hot, ix_fluid_line]
        s_ids = solid_idx_grid[iz_hot, iy_hot, ix_solid_line]
        tf_line = tf[f_ids]
        ts_line = tsol[s_ids]
        t_int = (g_f * tf_line[-1] + g_s * ts_line[0]) / (g_f + g_s)
        wall_normal_profiles[int(round(ts * 1e3))] = np.concatenate([tf_line, [t_int], ts_line])

snap_cell_tmax = np.array(snap_cell_tmax)
snap_cell_pmaxT = np.array(snap_cell_pmaxT)

# -----------------------------------------------------------------------------
# 4. BOILING THRESHOLD & ANALYTICAL BENCHMARK CALCULATIONS
# -----------------------------------------------------------------------------
t_boil_1atm, idx_1atm = first_crossing(t_f, f_tmax, T_SAT_ATM)
p_at_cross = p_if_max[idx_1atm]
T_sat_local = compute_tsat_k(p_at_cross)
t_boil_local, idx_local = first_crossing(t_f, f_tmax, T_sat_local)
boil_loc_m = f_tmax_loc[idx_local]
t_cell_boil, idx_cell_boil = first_crossing(snap_times, snap_cell_tmax, T_sat_local)
t_cell_boil_1atm, _ = first_crossing(snap_times, snap_cell_tmax, T_SAT_ATM)

# Analytical 1D semi-infinite conjugate conduction solution with uniform solid heating
e_s = np.sqrt(KAPPA_S * RHO_S * CP_S)
e_f = np.sqrt(KAPPA_F * RHO_F * CP_F)
slope_1d_int = (e_s / (e_s + e_f)) * (Q_DOT / (RHO_S * CP_S))
slope_adiab = Q_DOT / (RHO_S * CP_S)
T_int_1d_analytical = T0_K + slope_1d_int * t_f
T_adiab_analytical = T0_K + slope_adiab * t_f
t_boil_1d_analytical = (T_sat_local - T0_K) / slope_1d_int
rel_diff_1d_pct = abs(t_boil_local - t_boil_1d_analytical) / t_boil_1d_analytical * 100.0

# -----------------------------------------------------------------------------
# 5. EXPORT CSV TABLES & SUMMARY REPORTS
# -----------------------------------------------------------------------------
print("[*] Writing CSV datasets and summary files to data/ and part5/results/...")

hist_mat = np.column_stack([
    t_f, t_f * 1e3, f_tmax, T_if_avg, s_tmax, T_s_avg, s_tmin, T_f_avg,
    T_in_mix, T_out_mix, p_in_avg, p_out_avg, p_if_max,
    np.full_like(t_f, Q_GEN), q_if, Q_rem, E_gen, E_solid, E_fluid,
    dEs_dt, dEf_dt, eps_m_pct, log_co_max, log_di_max,
    f_tmax_loc[:, 0], f_tmax_loc[:, 1], f_tmax_loc[:, 2],
])
hist_hdr = (
    "time_s,time_ms,T_fluid_max_K,T_if_avg_K,T_solid_max_K,T_solid_avg_K,"
    "T_solid_min_K,T_fluid_avg_K,T_in_mix_K,T_out_mix_K,p_in_Pa,p_out_Pa,"
    "p_if_max_Pa,Q_gen_W,Q_interface_W,Q_removed_W,E_gen_J,E_solid_J,E_fluid_J,"
    "dEs_dt_W,dEf_dt_W,eps_m_pct,Co_max,Di_max,hotspot_x_m,hotspot_y_m,hotspot_z_m"
)
for out_dir in (DATA_DIR, RES_DIR):
    np.savetxt(os.path.join(out_dir, "part5_transient_history.csv"), hist_mat,
               delimiter=",", header=hist_hdr, comments="")

prof_mat = np.column_stack([
    delta_n_mm,
    wall_normal_profiles[4],
    wall_normal_profiles[8],
    wall_normal_profiles[12],
    wall_normal_profiles[16],
    wall_normal_profiles[20],
])
prof_hdr = "delta_n_mm,T_4ms_K,T_8ms_K,T_12ms_K,T_16ms_K,T_20ms_K"
for out_dir in (DATA_DIR, RES_DIR):
    np.savetxt(os.path.join(out_dir, "part5_wall_normal_profiles.csv"), prof_mat,
               delimiter=",", header=prof_hdr, comments="")

summary_rows = [
    ("endTime_s", f"{t_f[-1]:.4f}"),
    ("deltaT_s", f"{t_f[0]:.6e}"),
    ("Co_mean_avg", f"{np.mean(log_co_mean):.5f}"),
    ("Co_max_peak", f"{np.max(log_co_max):.4f}"),
    ("Di_mean_avg", f"{np.mean(log_di_mean):.5f}"),
    ("Di_max_peak", f"{np.max(log_di_max):.4f}"),
    ("p_inlet_Pa", f"{p_in_avg[-1]:.2f}"),
    ("p_outlet_Pa", f"{p_out_avg[-1]:.2f}"),
    ("p_interface_max_Pa", f"{p_at_cross:.2f}"),
    ("T_sat_1atm_K", f"{T_SAT_ATM:.2f}"),
    ("T_sat_local_K", f"{T_sat_local:.2f}"),
    ("t_boil_interface_1atm_ms", f"{t_boil_1atm*1e3:.4f}"),
    ("t_boil_interface_local_ms", f"{t_boil_local*1e3:.4f}"),
    ("t_boil_cell_center_local_ms", f"{t_cell_boil*1e3:.4f}"),
    ("t_boil_1d_analytical_ms", f"{t_boil_1d_analytical*1e3:.4f}"),
    ("hotspot_x_mm", f"{boil_loc_m[0]*1e3:.3f}"),
    ("hotspot_y_mm", f"{boil_loc_m[1]*1e3:.3f}"),
    ("hotspot_z_mm", f"{boil_loc_m[2]*1e3:.3f}"),
    ("T_fluid_max_20ms_K", f"{f_tmax[-1]:.4f}"),
    ("T_fluid_cell_max_20ms_K", f"{snap_cell_tmax[-1]:.4f}"),
    ("T_solid_max_20ms_K", f"{s_tmax[-1]:.4f}"),
    ("T_solid_avg_20ms_K", f"{T_s_avg[-1]:.4f}"),
    ("T_interface_avg_20ms_K", f"{T_if_avg[-1]:.4f}"),
    ("T_out_mix_20ms_K", f"{T_out_mix[-1]:.4f}"),
    ("Q_gen_MW", f"{Q_GEN/1e6:.6f}"),
    ("Q_interface_20ms_kW", f"{q_if[-1]/1e3:.4f}"),
    ("Q_removed_20ms_W", f"{Q_rem[-1]:.4f}"),
    ("dEs_dt_20ms_MW", f"{dEs_dt[-1]/1e6:.6f}"),
    ("dEf_dt_20ms_kW", f"{dEf_dt[-1]/1e3:.4f}"),
    ("E_gen_20ms_kJ", f"{E_gen[-1]/1e3:.4f}"),
    ("E_solid_20ms_kJ", f"{E_solid[-1]/1e3:.4f}"),
    ("E_fluid_20ms_J", f"{E_fluid[-1]:.4f}"),
    ("E_interface_cum_20ms_J", f"{E_if_cum[-1]:.4f}"),
    ("cum_energy_imbalance_20ms_pct", f"{cum_imbalance_pct[-1]:.4f}"),
    ("inst_energy_imbalance_20ms_pct", f"{inst_imbalance_pct[-1]:.4f}"),
    ("mass_imbalance_max_pct", f"{np.max(eps_m_pct):.4e}"),
]
for out_dir in (DATA_DIR, RES_DIR):
    with open(os.path.join(out_dir, "part5_summary_metrics.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in summary_rows:
            f.write(f"{k},{v}\n")

summary_text = f"""================================================================================
PART V: TRANSIENT CONJUGATE HEAT TRANSFER (chtMultiRegionFoam) — SUMMARY
================================================================================
1. SETUP & NUMERICAL STABILITY (t in [0.00, 20.00] ms, 400 steps, dt = 50 us)
   - Initial State             : Converged unheated flow + T0 = {T0_K:.2f} K (25.00 C)
   - Continuous Heat Source    : q''' = {Q_DOT:.2e} W/m^3 => Q_gen = {Q_GEN/1e6:.4f} MW
   - Time Integration Scheme   : Implicit Euler; PIMPLE (2 outer, 2 inner correctors)
   - Fluid Courant Number      : Co_mean = {np.mean(log_co_mean):.5f} | Co_max = {np.max(log_co_max):.4f}
   - Solid Diffusion Number    : Di_mean = {np.mean(log_di_mean):.5f} | Di_max = {np.max(log_di_max):.4f}
   - Mass Conservation Error   : eps_m <= {np.max(eps_m_pct):.3e}% (m_dot = {abs(phi_in[-1]):.4f} kg/s)

2. BOILING-TEMPERATURE THRESHOLD & HOTSPOT LOCATION
   - Local Pressure Assumption : p_in = {p_in_avg[-1]/1e3:.3f} kPa, p_out = {p_out_avg[-1]/1e3:.3f} kPa, p_if,max = {p_at_cross/1e3:.3f} kPa
   - Saturation Threshold      : T_sat(1 atm) = {T_SAT_ATM:.2f} K | T_sat(p_if,max) = {T_sat_local:.2f} K ({T_sat_local-273.15:.2f} C)
   - First Interface Crossing  : t_boil = {t_boil_local*1e3:.4f} ms ({t_boil_local:.6f} s; step {idx_local+1} at 12.40 ms)
   - At 1 atm (373.15 K)       : t_boil = {t_boil_1atm*1e3:.4f} ms ({t_boil_1atm:.6f} s)
   - First Fluid Cell (12.5um) : t_boil,cell = {t_cell_boil*1e3:.4f} ms ({t_cell_boil:.6f} s)
   - Hotspot Location          : (x, y, z) = ({boil_loc_m[0]*1e3:.2f}, {boil_loc_m[1]*1e3:.2f}, {boil_loc_m[2]*1e3:.2f}) mm
                                 (Lower trailing edge on left cube face in stagnant corner)
   - 1D Analytical Benchmark   : t_boil,1D = {t_boil_1d_analytical*1e3:.4f} ms (deviation = {rel_diff_1d_pct:.2f}%)

3. THERMAL STATE & TRANSIENT ENERGY BALANCE (at t = t_boil vs. t = 20.00 ms)
   - At t = 12.40 ms (step 248):
       Fluid T_max             : {f_tmax[idx_local]:.2f} K ({f_tmax[idx_local]-273.15:.2f} C)
       Cube T_max / T_volAvg   : {s_tmax[idx_local]:.2f} K / {T_s_avg[idx_local]:.2f} K
       Interface T_avg         : {T_if_avg[idx_local]:.2f} K
       Q_interface / dEs_dt    : {q_if[idx_local]/1e3:.2f} kW / {dEs_dt[idx_local]/1e6:.4f} MW
       E_solid / E_fluid       : {E_solid[idx_local]/1e3:.3f} kJ ({E_solid[idx_local]/E_gen[idx_local]*100:.2f}%) / {E_fluid[idx_local]:.2f} J ({E_fluid[idx_local]/E_gen[idx_local]*100:.2f}%)
   - At t = 20.00 ms (step 400):
       Fluid T_max (interface) : {f_tmax[-1]:.2f} K ({f_tmax[-1]-273.15:.2f} C)
       Fluid T_max (cell-ctr)  : {snap_cell_tmax[-1]:.2f} K ({snap_cell_tmax[-1]-273.15:.2f} C)
       Cube T_max / T_volAvg   : {s_tmax[-1]:.2f} K / {T_s_avg[-1]:.2f} K
       Cube T_min / Interface  : {s_tmin[-1]:.2f} K / {T_if_avg[-1]:.2f} K
       Outlet T_out,mix        : {T_out_mix[-1]:.4f} K (Delta T_mix = {T_out_mix[-1]-T_in_mix[-1]:.4f} K, Q_rem = {Q_rem[-1]:.2f} W)
       Instantaneous Powers    : Q_gen = {Q_GEN/1e6:.4f} MW | dEs/dt = {dEs_dt[-1]/1e6:.4f} MW | dEf/dt = {dEf_dt[-1]/1e3:.2f} kW | Q_if = {q_if[-1]/1e3:.2f} kW
       Inst. Energy Imbalance  : {inst_imbalance_pct[-1]:.4f}%
       Cumulative Energies     : E_gen = {E_gen[-1]/1e3:.4f} kJ | E_solid = {E_solid[-1]/1e3:.4f} kJ ({E_solid[-1]/E_gen[-1]*100:.2f}%) | E_fluid = {E_fluid[-1]:.2f} J ({E_fluid[-1]/E_gen[-1]*100:.2f}%)
       Cum. Energy Imbalance   : {cum_imbalance_pct[-1]:.4f}% (Integrated Q_interface = {E_if_cum[-1]:.2f} J)
================================================================================
"""
print(summary_text)
for out_dir in (DATA_DIR, RES_DIR):
    with open(os.path.join(out_dir, "part5_integral_summary_metrics.txt"), "w") as f:
        f.write(summary_text)

# -----------------------------------------------------------------------------
# 6. FIGURE 1: BOILING THRESHOLD & FLUID/SOLID TEMPERATURE HISTORIES (2x1)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 1: fig_part5_boiling_and_temperatures...")
fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True)
t_ms = t_f * 1e3

# Panel (a): Fluid & Interface Temperatures + Boiling Threshold
ax = axes[0]
ax.plot(t_ms, f_tmax, color="#d62728", lw=1.9, label=rf"Fluid $\max\,T(\mathbf{{x}},t)$ (interface: {f_tmax[-1]:.1f} K)")
ax.plot(snap_times * 1e3, snap_cell_tmax, "s--", color="#9467bd", ms=4.2, lw=1.4,
        label=rf"Fluid 1st-cell center $\max\,T$ ({snap_cell_tmax[-1]:.1f} K)")
ax.plot(t_ms, T_if_avg, color="#1f77b4", lw=1.6, label=rf"Interface area-avg $\bar{{T}}_{{\mathrm{{int}}}}$ ({T_if_avg[-1]:.1f} K)")
ax.plot(fp_T[:, 0] * 1e3, fp_T[:, 7], color="#2ca02c", ls="-.", lw=1.3,
        label=rf"Near-wall probe P6 ({fp_T[-1, 7]:.1f} K)")
ax.plot(fp_T[:, 0] * 1e3, fp_T[:, 8], color="#ff7f0e", ls=":", lw=1.4,
        label=rf"Rear-wall probe P7 ({fp_T[-1, 8]:.1f} K)")

ax.axhline(T_sat_local, color="crimson", ls="--", lw=1.4,
           label=rf"$T_{{\mathrm{{sat}}}}(p = 101.94\,\mathrm{{kPa}}) = {T_sat_local:.2f}\,\mathrm{{K}}$")
ax.plot(t_boil_local * 1e3, T_sat_local, "o", color="crimson", ms=6.0, zorder=5)
ax.axvline(t_boil_local * 1e3, color="crimson", ls=":", lw=1.0, alpha=0.8)
ax.annotate(
    rf"Interface: $t_{{\mathrm{{boil}}}} = {t_boil_local*1e3:.2f}\,\mathrm{{ms}}$"
    "\n@ $(75.00, 0.81, 448.77)$ mm",
    xy=(t_boil_local * 1e3, T_sat_local),
    xytext=(2.6, 383.0),
    fontsize=7.6, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
)
ax.plot(t_cell_boil * 1e3, T_sat_local, "D", color="#6a3d9a", ms=5.0, zorder=5)
ax.annotate(
    rf"1st cell: {t_cell_boil*1e3:.2f} ms",
    xy=(t_cell_boil * 1e3, T_sat_local),
    xytext=(13.6, 314.0),
    fontsize=7.5, color="#6a3d9a", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#6a3d9a", alpha=0.94),
    arrowprops=dict(arrowstyle="->", color="#6a3d9a", lw=1.0),
)

ax.set_ylabel("Fluid / Interface Temperature $T$ [K]")
ax.set_title("(a) Maximum Fluid Temperature, Probes, and Boiling-Threshold Crossing", fontweight="bold")
ax.set_ylim(293, 462)
ax.set_xlim(0.0, 20.0)
ax.grid(True, ls=":", alpha=0.55)
ax.legend(loc="upper left", ncol=2, fontsize=6.9, frameon=True, framealpha=0.94)

# Panel (b): Solid Cube Thermal Response
ax = axes[1]
ax.plot(t_ms, T_adiab_analytical, color="k", ls=":", lw=1.4,
        label=r"Adiabatic slope $T_0 + (\dot{q}/\rho_s c_{p,s})t$ (6.70 K/ms)")
ax.plot(t_ms, s_tmax, color="#d62728", lw=1.8, label=rf"Cube $T_{{\mathrm{{solid}},\max}}$ ({s_tmax[-1]:.2f} K)")
ax.plot(t_ms, T_s_avg, color="#ff7f0e", ls="--", lw=1.6, label=rf"Cube volume-avg $T_{{\mathrm{{solid,volAvg}}}}$ ({T_s_avg[-1]:.2f} K)")
ax.plot(t_ms, T_if_avg, color="#1f77b4", lw=1.5, label=rf"Interface area-avg $\bar{{T}}_{{\mathrm{{int}}}}$ ({T_if_avg[-1]:.2f} K)")
ax.plot(t_ms, s_tmin, color="#2ca02c", ls="-.", lw=1.4, label=rf"Cube $T_{{\mathrm{{solid}},\min}}$ (corner: {s_tmin[-1]:.2f} K)")

ax.set_xlabel("Physical Time $t$ [ms]")
ax.set_ylabel("Solid Temperature $T$ [K]")
ax.set_title("(b) Solid Cube Thermal Response vs. Adiabatic Heating Reference", fontweight="bold")
ax.set_ylim(293, 442)
ax.set_xlim(0.0, 20.0)
ax.grid(True, ls=":", alpha=0.55)
ax.legend(loc="upper left", ncol=1, fontsize=7.3, frameon=True, framealpha=0.93)

fig.tight_layout(h_pad=1.0)
save_fig(fig, "fig_part5_boiling_and_temperatures")

# -----------------------------------------------------------------------------
# 7. FIGURE 2: TRANSIENT ENERGY BALANCE & STORED ENERGY (1x2)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 2: fig_part5_energy_balance_and_storage...")
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))

# Panel (a): Instantaneous Thermal Power Balance
ax = axes[0]
l1, = ax.plot(t_ms, np.full_like(t_ms, Q_GEN / 1e6), "k-", lw=1.5, label=r"$Q_{\mathrm{gen}} = 3.350\,\mathrm{MW}$")
l2, = ax.plot(t_ms, dEs_dt / 1e6, color="#ff7f0e", lw=1.6, label=r"$\mathrm{d}E_{\mathrm{solid}}/\mathrm{d}t$")
l3, = ax.plot(t_ms, (dEs_dt + dEf_dt + Q_rem) / 1e6, color="#2ca02c", ls=":", lw=1.8,
              label=r"$\mathrm{d}E_s/\mathrm{d}t + \mathrm{d}E_f/\mathrm{d}t + Q_{\mathrm{rem}}$")
l4, = ax.plot(t_ms, Q_rem / 1e6, color="#7f7f7f", ls="--", lw=1.2, label=r"$Q_{\mathrm{rem}}$ ($\approx 0\,\mathrm{MW}$)")
ax.set_xlabel("Physical Time $t$ [ms]")
ax.set_ylabel("Power [MW]")
ax.set_ylim(-0.15, 4.75)
ax.set_xlim(0.0, 20.0)
ax.grid(True, ls=":", alpha=0.55)

ax2 = ax.twinx()
l5, = ax2.plot(t_ms, q_if / 1e3, color="#d62728", lw=1.5, label=rf"$Q_{{\mathrm{{int}}}}$ ({q_if[-1]/1e3:.2f} kW)")
l6, = ax2.plot(t_ms, dEf_dt_smooth / 1e3, color="#1f77b4", ls="-.", lw=1.4, label=rf"$\mathrm{{d}}E_f/\mathrm{{d}}t$ ({dEf_dt_smooth[-1]/1e3:.2f} kW)")
ax2.set_ylabel("Interface / Fluid Rate [kW]", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.set_ylim(-0.8, 27.5)

ax.set_title("(a) Transient Thermal Power Balance", fontweight="bold")
lines_a = [l1, l2, l3, l4, l5, l6]
labels_a = [ln.get_label() for ln in lines_a]
ax.legend(lines_a, labels_a, loc="upper left", ncol=2, fontsize=5.8, frameon=True, framealpha=0.94, columnspacing=0.9)

# Panel (b): Cumulative Energy & Stored Thermal Energy
ax = axes[1]
m1, = ax.plot(t_ms, E_gen / 1e3, "k--", lw=1.5, label=rf"$E_{{\mathrm{{gen}}}}$ ({E_gen[-1]/1e3:.2f} kJ)")
m2, = ax.plot(t_ms, E_solid / 1e3, color="#ff7f0e", lw=1.8, label=rf"$E_{{\mathrm{{solid}}}}$ ({E_solid[-1]/1e3:.2f} kJ, 99.63%)")
ax.set_xlabel("Physical Time $t$ [ms]")
ax.set_ylabel("Solid / Generated Energy [kJ]")
ax.set_ylim(0.0, 72.0)
ax.set_xlim(0.0, 20.0)
ax.grid(True, ls=":", alpha=0.55)

ax2 = ax.twinx()
m3, = ax2.plot(t_ms, E_fluid, color="#1f77b4", lw=1.6, label=rf"$E_{{\mathrm{{fluid}}}}$ ({E_fluid[-1]:.1f} J, 0.34%)")
m4, = ax2.plot(t_ms, E_if_cum, color="#d62728", ls="-.", lw=1.4, label=rf"$\int Q_{{\mathrm{{int}}}}\,\mathrm{{d}}t$ ({E_if_cum[-1]:.1f} J)")
ax2.set_ylabel("Fluid / Interface Energy [J]", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_ylim(0.0, 280.0)

ax.set_title(rf"(b) Cumulative Storage ($\varepsilon_E = {cum_imbalance_pct[-1]:.3f}\%$)", fontweight="bold")
lines_b = [m1, m2, m3, m4]
labels_b = [ln.get_label() for ln in lines_b]
ax.legend(lines_b, labels_b, loc="upper left", fontsize=6.5, frameon=True, framealpha=0.92)

fig.tight_layout(w_pad=1.4)
save_fig(fig, "fig_part5_energy_balance_and_storage")

# -----------------------------------------------------------------------------
# 8. FIGURE 3: HOTSPOT WALL-NORMAL PROFILE & NUMERICAL VERIFICATION (1x2)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 3: fig_part5_hotspot_and_numerics...")
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))

# Panel (a): Wall-normal thermal boundary layer at boiling hotspot
ax = axes[0]
colors_prof = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]
for ts_ms, col in zip([4, 8, 12, 16, 20], colors_prof):
    ax.plot(delta_n_mm, wall_normal_profiles[ts_ms], "o-", color=col, ms=2.4, lw=1.4,
            label=f"$t = {ts_ms}$ ms")
ax.axvline(0.0, color="k", ls="-", lw=1.0, alpha=0.8)
ax.axhline(T_sat_local, color="crimson", ls="--", lw=1.3, label=rf"$T_{{\mathrm{{sat}}}} = {T_sat_local:.2f}$ K")
ax.axvspan(-0.5, 0.0, color="#e6f2ff", alpha=0.55)
ax.axvspan(0.0, 1.5, color="#f2f2f2", alpha=0.65)
ax.text(-0.25, 437, "Water", fontsize=7.6, fontweight="bold", color="#084594", ha="center")
ax.text(0.75, 437, "Solid Cube", fontsize=7.6, fontweight="bold", color="#333333", ha="center")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(268, 452)
ax.set_yticks([280, 300, 320, 340, 360, 380, 400, 420, 440])
ax.set_xlabel(r"Wall-Normal Coordinate $\delta n = x - 75\,\mathrm{mm}$ [mm]")
ax.set_ylabel("Temperature $T$ [K]")
ax.set_title("(a) Wall-Normal $T(\\delta n)$ at Boiling Hotspot", fontweight="bold")
ax.grid(True, ls=":", alpha=0.55)
ax.legend(loc="lower right", ncol=2, fontsize=6.1, frameon=True, framealpha=0.95, columnspacing=0.9, handlelength=1.5)

# Panel (b): Analytical 1D Conjugate Benchmark, Courant/Diffusion Stability & Pressure
ax = axes[1]
n1, = ax.plot(t_ms, f_tmax, color="#d62728", lw=1.7, label=rf"3D CFD peak ($t_{{\mathrm{{boil}}}}={t_boil_local*1e3:.2f}$ ms)")
n2, = ax.plot(t_ms, T_int_1d_analytical, color="k", ls="--", lw=1.4,
              label=rf"1D Analytical ($t_{{\mathrm{{boil}}}}={t_boil_1d_analytical*1e3:.2f}$ ms)")
ax.axhline(T_sat_local, color="crimson", ls=":", lw=1.2)
ax.set_xlim(0.0, 20.0)
ax.set_ylim(290, 445)
ax.set_xlabel("Physical Time $t$ [ms]")
ax.set_ylabel("Interface Temperature $T$ [K]")
ax.grid(True, ls=":", alpha=0.55)

ax2 = ax.twinx()
n3, = ax2.plot(t_ms, log_co_max, color="#1f77b4", lw=1.4, ls="-.", label=rf"Fluid $\mathrm{{Co}}_{{\max}} = {np.max(log_co_max):.2f}$")
n4, = ax2.plot(t_ms, log_di_max, color="#2ca02c", lw=1.4, ls=":", label=rf"Solid $\mathrm{{Di}}_{{\max}} = {np.max(log_di_max):.2f}$")
ax2.set_ylabel(r"$\mathrm{Co}_{\max}$, $\mathrm{Di}_{\max}$ [-]", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_ylim(0.0, 12.5)

ax.set_title(r"(b) Analytical Benchmark & Stability ($\Delta t = 50\,\mu$s)", fontweight="bold")
lines_n = [n1, n2, n3, n4]
labels_n = [ln.get_label() for ln in lines_n]
ax.legend(lines_n, labels_n, loc="upper left", fontsize=6.6, frameon=True, framealpha=0.92)
ax.text(
    11.0, 296.0,
    rf"$p_{{\mathrm{{in}}}} = {p_in_avg[-1]/1e3:.2f}$ kPa" "\n"
    rf"$p_{{\mathrm{{out}}}} = {p_out_avg[-1]/1e3:.2f}$ kPa" "\n"
    rf"$p_{{\mathrm{{if,\max}}}} = {p_at_cross/1e3:.2f}$ kPa" "\n"
    rf"$T_{{\mathrm{{sat}}}}(p) = {T_sat_local:.2f}$ K",
    fontsize=6.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
)

fig.tight_layout(w_pad=1.3)
save_fig(fig, "fig_part5_hotspot_and_numerics")

print("[+] Part V post-processing completed successfully!")
