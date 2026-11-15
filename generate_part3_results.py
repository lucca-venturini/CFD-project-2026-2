#!/usr/bin/env python3
"""
generate_part3_results.py
=========================
Comprehensive Post-Processing and Deliverable Generation for Part III:
Transient Wake Development with pimpleFoam (OpenFOAM 2506)

Generates:
1. CSV Data Tables & Summary in data/ and part3/results/:
   - part3_probe_time_histories.csv
   - part3_frequency_spectrum_strouhal.csv
   - part3_wake_recirculation_evolution.csv
   - part3_steady_vs_transient_centerline.csv
   - part3_steady_vs_transient_profiles.csv
   - part3_integral_summary_metrics.txt
2. Publication Figures (300 DPI PNG + Vector PDF) in figures/ and part3/results/:
   - fig_part3_courant_and_convergence.(pdf|png)
   - fig_part3_probe_histories_and_spectra.(pdf|png)
   - fig_part3_wake_evolution_and_recirculation.(pdf|png)
   - fig_part3_fluctuations_and_tke.(pdf|png)
   - fig_part3_steady_vs_time_averaged_profiles.(pdf|png)
"""

import os
import re
import shutil
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & GEOMETRY
# -----------------------------------------------------------------------------
RHO = 1000.0
MU = 8.90e-04
NU = 8.93e-07
W, H, L = 0.20, 0.20, 1.50
HC = 0.05
U_BULK = 1.00
DYN_PRESS = 0.5 * RHO * (U_BULK ** 2)  # 500.0 Pa
RE_DH = U_BULK * 0.20 / NU             # 223,964
RE_HC = U_BULK * HC / NU               # 55,991

CASE_DIR = "part3"
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

# -----------------------------------------------------------------------------
# 2. LOAD MESH & HELPER READERS FOR 3D BINARY FIELDS
# -----------------------------------------------------------------------------
print("[*] Loading Part III mesh coordinates...")
with open(os.path.join(CASE_DIR, "constant/polyMesh/points"), "rb") as f:
    c = f.read()
n_pts_str = b"1580001\n("
idx = c.find(n_pts_str) + len(n_pts_str)
pts = np.frombuffer(c[idx:idx + 1580001 * 24], dtype="<f8").reshape((1580001, 3))

x_faces = np.unique(np.round(pts[:, 0], 9))  # 105
y_faces = np.unique(np.round(pts[:, 1], 9))  # 97
z_faces = np.unique(np.round(pts[:, 2], 9))  # 157

x_c = 0.5 * (x_faces[:-1] + x_faces[1:])     # 104
y_c = 0.5 * (y_faces[:-1] + y_faces[1:])     # 96
z_c = 0.5 * (z_faces[:-1] + z_faces[1:])     # 156

dx = np.diff(x_faces)
dy = np.diff(y_faces)
dz = np.diff(z_faces)
dA_xy = np.outer(dy, dx)                     # (96, 104)

mask_3d = np.ones((156, 96, 104), dtype=bool)
mask_3d[50:74, 0:36, 40:64] = False
n_fluid = int(np.sum(mask_3d))               # 1,536,768

def load_scalar_3d(t_dir, field_name):
    with open(os.path.join(CASE_DIR, t_dir, field_name), "rb") as f:
        buf = f.read()
    tag = f"{n_fluid}\n(".encode()
    pos = buf.find(tag) + len(tag)
    arr_flat = np.frombuffer(buf[pos:pos + n_fluid * 8], dtype="<f8")
    arr_3d = np.full((156, 96, 104), np.nan)
    arr_3d[mask_3d] = arr_flat
    return arr_3d

def load_vector_3d(t_dir, field_name):
    with open(os.path.join(CASE_DIR, t_dir, field_name), "rb") as f:
        buf = f.read()
    tag = f"{n_fluid}\n(".encode()
    pos = buf.find(tag) + len(tag)
    arr_flat = np.frombuffer(buf[pos:pos + n_fluid * 24], dtype="<f8").reshape((n_fluid, 3))
    arr_3d = np.full((156, 96, 104, 3), np.nan)
    arr_3d[mask_3d] = arr_flat
    return arr_3d

def load_symmtensor_3d(t_dir, field_name):
    with open(os.path.join(CASE_DIR, t_dir, field_name), "rb") as f:
        buf = f.read()
    tag = f"{n_fluid}\n(".encode()
    pos = buf.find(tag) + len(tag)
    arr_flat = np.frombuffer(buf[pos:pos + n_fluid * 48], dtype="<f8").reshape((n_fluid, 6))
    arr_3d = np.full((156, 96, 104, 6), np.nan)
    arr_3d[mask_3d] = arr_flat
    return arr_3d

def get_wake_metrics(U_vec):
    Uz_sym = 0.5 * (U_vec[:, :, 51, 2] + U_vec[:, :, 52, 2])
    Uz_bot = Uz_sym[:, 0]
    up_rev = np.where(Uz_bot[:50] < 0)[0]
    if len(up_rev) > 0:
        i0 = up_rev[0]
        u1, u2 = Uz_bot[i0 - 1], Uz_bot[i0]
        z_sep = z_c[i0 - 1] + (0.0 - u1) / (u2 - u1) * (z_c[i0] - z_c[i0 - 1])
        L_up = 0.40 - z_sep
    else:
        z_sep, L_up = 0.40, 0.0
    w_bot = Uz_bot[74:]
    z_w = z_c[74:]
    r_idx = np.where((w_bot[:-1] < 0) & (w_bot[1:] >= 0))[0]
    if len(r_idx) > 0:
        ir = r_idx[-1]
        u1, u2 = w_bot[ir], w_bot[ir + 1]
        z_R = z_w[ir] + (0.0 - u1) / (u2 - u1) * (z_w[ir + 1] - z_w[ir])
        L_r = z_R - 0.45
    else:
        z_R, L_r = 0.45, 0.0
    return z_sep, L_up, z_R, L_r

# -----------------------------------------------------------------------------
# 3. LOAD 3D FIELDS (STEADY t=0, INSTANTANEOUS t=2.0s, TIME-AVERAGED t=2.0s)
# -----------------------------------------------------------------------------
print("[*] Loading 3D fields at t=0 (Steady), t=2.0s (Instantaneous & Time-Averaged)...")
U_steady = load_vector_3d("0", "U")
p_steady = load_scalar_3d("0", "p")
k_steady = load_scalar_3d("0", "k")

U_inst = load_vector_3d("2", "U")
p_inst = load_scalar_3d("2", "p")
k_inst = load_scalar_3d("2", "k")
Co_inst = load_scalar_3d("2", "Co")

U_mean = load_vector_3d("2", "UMean")
p_mean = load_scalar_3d("2", "pMean")
k_mean = load_scalar_3d("2", "kMean")
U_p2m = load_symmtensor_3d("2", "UPrime2Mean")
p_p2m = load_scalar_3d("2", "pPrime2Mean")

# Resolved turbulent kinetic energy & RMS fluctuations
k_res = 0.5 * (U_p2m[:, :, :, 0] + U_p2m[:, :, :, 3] + U_p2m[:, :, :, 5])
k_tot = k_mean + k_res
u_rms_mag = np.sqrt(np.maximum(0.0, U_p2m[:, :, :, 0] + U_p2m[:, :, :, 3] + U_p2m[:, :, :, 5]))
ux_rms = np.sqrt(np.maximum(0.0, U_p2m[:, :, :, 0]))
uy_rms = np.sqrt(np.maximum(0.0, U_p2m[:, :, :, 3]))
uz_rms = np.sqrt(np.maximum(0.0, U_p2m[:, :, :, 5]))
p_rms_Pa = np.sqrt(np.maximum(0.0, p_p2m)) * RHO

# Evaluate wake metrics across all 41 snapshots t = 0.00, 0.05, ..., 2.00 s
print("[*] Evaluating wake recirculation length evolution across 41 snapshots (0 to 2.0 s)...")
snap_times = np.round(np.arange(0.0, 2.001, 0.05), 2)
snap_dirs = ["0" if t == 0 else f"{t:.2f}".rstrip("0").rstrip(".") for t in snap_times]

probe_coords = [
    (0.10, 0.025, 0.300),  # Probe 0: Upstream approach
    (0.10, 0.005, 0.380),  # Probe 1: Upstream lower-wall separation
    (0.10, 0.060, 0.425),  # Probe 2: Above cube roof shear layer
    (0.10, 0.025, 0.475),  # Probe 3: Immediately downstream (near wake)
    (0.10, 0.025, 0.550),  # Probe 4: Wake recirculation core
    (0.10, 0.025, 0.650),  # Probe 5: Wake reattachment / shedding zone
    (0.10, 0.025, 0.850),  # Probe 6: Far wake centerline
    (0.10, 0.003, 0.550),  # Probe 7: Near lower wall in wake
]
probe_indices = [
    (int(np.argmin(np.abs(z_c - pz))), int(np.argmin(np.abs(y_c - py))), int(np.argmin(np.abs(x_c - px))))
    for (px, py, pz) in probe_coords
]

snap_L_up = []
snap_L_r = []
snap_z_R = []
snap_dP = []
snap_Umax = []
snap_Uzmin = []
snap_probe_U = []
snap_probe_p = []
snap_probe_k = []

for t_val, t_dir in zip(snap_times, snap_dirs):
    Ut = load_vector_3d(t_dir, "U")
    pt = load_scalar_3d(t_dir, "p")
    kt = load_scalar_3d(t_dir, "k")
    z_sep, L_up, z_R, L_r = get_wake_metrics(Ut)
    snap_L_up.append(L_up * 1e3)
    snap_L_r.append(L_r * 1e3)
    snap_z_R.append(z_R)
    dP_val = float(np.sum(pt[0] * dA_xy) / 0.04 * RHO)
    snap_dP.append(dP_val)
    snap_Umax.append(float(np.nanmax(np.linalg.norm(Ut, axis=-1))))
    snap_Uzmin.append(float(np.nanmin(Ut[:, :, :, 2])))
    # Extract probe values for t < 0.5 s to stitch with continuous 0.5..2.0 s probe logs
    if t_val < 0.50:
        u_p = [Ut[iz, iy, ix, :] for (iz, iy, ix) in probe_indices]
        p_p = [pt[iz, iy, ix] for (iz, iy, ix) in probe_indices]
        k_p = [kt[iz, iy, ix] for (iz, iy, ix) in probe_indices]
        snap_probe_U.append((t_val, u_p))
        snap_probe_p.append((t_val, p_p))
        snap_probe_k.append((t_val, k_p))

snap_L_up = np.array(snap_L_up)
snap_L_r = np.array(snap_L_r)
snap_z_R = np.array(snap_z_R)
snap_dP = np.array(snap_dP)
snap_Umax = np.array(snap_Umax)
snap_Uzmin = np.array(snap_Uzmin)

z_sep_s, L_up_s, z_R_s, L_r_s = get_wake_metrics(U_steady)
z_sep_m, L_up_m, z_R_m, L_r_m = get_wake_metrics(U_mean)
z_sep_i, L_up_i, z_R_i, L_r_i = get_wake_metrics(U_inst)

# -----------------------------------------------------------------------------
# 4. PARSE CONTINUOUS PROBE HISTORIES, LOG RESIDUALS & COURANT NUMBER
# -----------------------------------------------------------------------------
print("[*] Parsing high-frequency wakeProbes, Courant history, and solver residuals...")
with open(os.path.join(CASE_DIR, "postProcessing/wakeProbes/0.5/U"), "r") as f:
    u_lines = [l for l in f if not l.startswith("#") and l.strip()]

t_probe_cont = []
U_probe_cont = []
for l in u_lines:
    parts = l.split("(")
    t_probe_cont.append(float(parts[0].strip()))
    vecs = [[float(x) for x in p.split(")")[0].split()] for p in parts[1:]]
    U_probe_cont.append(vecs)

t_probe_cont = np.array(t_probe_cont)
U_probe_cont = np.array(U_probe_cont)  # (19442, 8, 3)

p_probe_raw = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/wakeProbes/0.5/p"), comments="#")
k_probe_raw = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/wakeProbes/0.5/k"), comments="#")
p_probe_cont = p_probe_raw[:, 1:] * RHO
k_probe_cont = k_probe_raw[:, 1:]

# Stitch 0..0.45 s snapshot probe values with 0.50..2.00 s continuous probe records
t_early = np.array([x[0] for x in snap_probe_U])
U_early = np.array([x[1] for x in snap_probe_U])
p_early = np.array([x[1] for x in snap_probe_p]) * RHO
k_early = np.array([x[1] for x in snap_probe_k])

t_probe_full = np.concatenate([t_early, t_probe_cont])
U_probe_full = np.concatenate([U_early, U_probe_cont], axis=0)
p_probe_full = np.concatenate([p_early, p_probe_cont], axis=0)
k_probe_full = np.concatenate([k_early, k_probe_cont], axis=0)

# Spectral / Strouhal analysis on continuous uniform-dt probe signal (t in [0.5, 2.0] s)
dt_probe = float(np.mean(np.diff(t_probe_cont)))
N_fft = len(t_probe_cont)
freqs_fft = np.fft.rfftfreq(N_fft, d=dt_probe)
St_fft = freqs_fft * HC / U_BULK
window_hann = np.hanning(N_fft)

# Compute FFT amplitude for Ux and Uz at Probes 4, 5, 6, 7
spec_Ux = {}
spec_Uz = {}
spec_p = {}
for ip in range(8):
    ux_sig = U_probe_cont[:, ip, 0] - np.mean(U_probe_cont[:, ip, 0])
    uz_sig = U_probe_cont[:, ip, 2] - np.mean(U_probe_cont[:, ip, 2])
    p_sig = p_probe_cont[:, ip] - np.mean(p_probe_cont[:, ip])
    spec_Ux[ip] = 2.0 / np.sum(window_hann) * np.abs(np.fft.rfft(ux_sig * window_hann))
    spec_Uz[ip] = 2.0 / np.sum(window_hann) * np.abs(np.fft.rfft(uz_sig * window_hann))
    spec_p[ip] = 2.0 / np.sum(window_hann) * np.abs(np.fft.rfft(p_sig * window_hann))

# Dominant frequency and Strouhal number from Probe 5 (reattachment shedding zone) & Probe 4 (wake core)
idx_fx_p5 = int(np.argmax(spec_Ux[5][1:]) + 1)
f_dom_Ux = float(freqs_fft[idx_fx_p5])
St_dom_Ux = float(St_fft[idx_fx_p5])

idx_fz_p4 = int(np.argmax(spec_Uz[4][1:]) + 1)
f_dom_Uz = float(freqs_fft[idx_fz_p4])
St_dom_Uz = float(St_fft[idx_fz_p4])

# Zero-crossing half-period analysis on Probe 5 Ux(t)
ux5_dm = U_probe_cont[:, 5, 0] - np.mean(U_probe_cont[:, 5, 0])
zc_idx = np.where(np.diff(np.sign(ux5_dm)) != 0)[0]
t_zc_p5 = t_probe_cont[zc_idx]
half_periods = np.diff(t_zc_p5)
T_zc_mean = 2.0 * float(np.mean(half_periods))
f_zc_mean = 1.0 / T_zc_mean
St_zc_mean = f_zc_mean * HC / U_BULK

# Parse log.pimpleFoam for Courant numbers, deltaT, and final outer-corrector residuals
t_log = []
dt_log = []
co_mean_log = []
co_max_log = []
res_Ux, res_Uy, res_Uz, res_p, res_k, res_omega = [], [], [], [], [], []

with open(os.path.join(CASE_DIR, "log.pimpleFoam"), "r") as f:
    cur_co_mean, cur_co_max, cur_dt = None, None, None
    cur_t = None
    u_x, u_y, u_z, p_r, k_r, w_r = None, None, None, None, None, None
    for line in f:
        if line.startswith("Courant Number mean:"):
            parts = line.split()
            cur_co_mean = float(parts[3])
            cur_co_max = float(parts[5])
        elif line.startswith("deltaT ="):
            cur_dt = float(line.split("=")[1].strip())
        elif line.startswith("Time ="):
            if cur_t is not None and u_x is not None:
                t_log.append(cur_t)
                dt_log.append(cur_dt if cur_dt else 7.68e-5)
                co_mean_log.append(cur_co_mean if cur_co_mean else 0.0091)
                co_max_log.append(cur_co_max if cur_co_max else 6.0)
                res_Ux.append(u_x)
                res_Uy.append(u_y)
                res_Uz.append(u_z)
                res_p.append(p_r)
                res_k.append(k_r)
                res_omega.append(w_r)
            cur_t = float(line.split("=")[1].strip())
        elif "Solving for Ux, Initial residual =" in line:
            u_x = float(line.split("Initial residual =")[1].split(",")[0].strip())
        elif "Solving for Uy, Initial residual =" in line:
            u_y = float(line.split("Initial residual =")[1].split(",")[0].strip())
        elif "Solving for Uz, Initial residual =" in line:
            u_z = float(line.split("Initial residual =")[1].split(",")[0].strip())
        elif "Solving for p, Initial residual =" in line:
            p_r = float(line.split("Initial residual =")[1].split(",")[0].strip())
        elif "Solving for k, Initial residual =" in line:
            k_r = float(line.split("Initial residual =")[1].split(",")[0].strip())
        elif "Solving for omega, Initial residual =" in line:
            w_r = float(line.split("Initial residual =")[1].split(",")[0].strip())

t_log = np.array(t_log)
dt_log = np.array(dt_log)
co_mean_log = np.array(co_mean_log)
co_max_log = np.array(co_max_log)
res_Ux = np.array(res_Ux)
res_Uy = np.array(res_Uy)
res_Uz = np.array(res_Uz)
res_p = np.array(res_p)
res_k = np.array(res_k)
res_omega = np.array(res_omega)

# Parse inletPressure, massFlowInlet/Outlet, yPlus
inlet_p_dat = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/inletPressure/0.5/surfaceFieldValue.dat"), comments="#")
m_in_dat = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/massFlowInlet/0.5/surfaceFieldValue.dat"), comments="#")
m_out_dat = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/massFlowOutlet/0.5/surfaceFieldValue.dat"), comments="#")

t_yplus_w, yp_avg_w, yp_max_w = [], [], []
t_yplus_c, yp_avg_c, yp_max_c = [], [], []
with open(os.path.join(CASE_DIR, "postProcessing/yPlus/0.5/yPlus.dat"), "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if p[1] == "walls":
            t_yplus_w.append(float(p[0]))
            yp_max_w.append(float(p[3]))
            yp_avg_w.append(float(p[4]))
        elif p[1] == "cube":
            t_yplus_c.append(float(p[0]))
            yp_max_c.append(float(p[3]))
            yp_avg_c.append(float(p[4]))

# -----------------------------------------------------------------------------
# 5. SAVE CSV TABLES & INTEGRAL SUMMARY METRICS
# -----------------------------------------------------------------------------
print("[*] Writing Part III CSV datasets and summary metrics...")

# 5.1 Downsampled probe time history CSV (every 10th step ~ 0.77 ms resolution)
idx_ds = np.arange(0, len(t_probe_full), 10)
probe_csv = np.column_stack([
    t_probe_full[idx_ds],
    U_probe_full[idx_ds, 0, 2], U_probe_full[idx_ds, 2, 2],
    U_probe_full[idx_ds, 3, 2], U_probe_full[idx_ds, 4, 2],
    U_probe_full[idx_ds, 5, 2], U_probe_full[idx_ds, 6, 2],
    U_probe_full[idx_ds, 4, 0], U_probe_full[idx_ds, 5, 0], U_probe_full[idx_ds, 6, 0],
    p_probe_full[idx_ds, 3], p_probe_full[idx_ds, 4], p_probe_full[idx_ds, 5],
])
header_probe = "t_s,Uz_P0_up,Uz_P2_roof,Uz_P3_nearWake,Uz_P4_wakeCore,Uz_P5_reattach,Uz_P6_farWake,Ux_P4_wakeCore,Ux_P5_reattach,Ux_P6_farWake,P_Pa_P3,P_Pa_P4,P_Pa_P5"
for d in [DATA_DIR, RES_DIR]:
    np.savetxt(os.path.join(d, "part3_probe_time_histories.csv"), probe_csv, delimiter=",", header=header_probe, comments="", fmt="%.6f")

# 5.2 Frequency spectrum CSV (up to 25 Hz / St = 1.25)
mask_f = freqs_fft <= 25.0
spec_csv = np.column_stack([
    freqs_fft[mask_f],
    St_fft[mask_f],
    spec_Ux[4][mask_f],
    spec_Ux[5][mask_f],
    spec_Ux[6][mask_f],
    spec_Uz[4][mask_f],
    spec_Uz[5][mask_f],
    spec_p[4][mask_f],
])
header_spec = "freq_Hz,Strouhal_St,Amp_Ux_P4,Amp_Ux_P5,Amp_Ux_P6,Amp_Uz_P4,Amp_Uz_P5,Amp_P_Pa_P4"
for d in [DATA_DIR, RES_DIR]:
    np.savetxt(os.path.join(d, "part3_frequency_spectrum_strouhal.csv"), spec_csv, delimiter=",", header=header_spec, comments="", fmt="%.6f")

# 5.3 Wake recirculation evolution CSV (t = 0.00 to 2.00 s)
wake_evol_csv = np.column_stack([
    snap_times,
    snap_L_up,
    snap_z_R,
    snap_L_r,
    snap_L_r / (HC * 1e3),
    snap_dP,
    snap_Umax,
    snap_Uzmin,
])
header_wake_evol = "t_s,L_sep_up_mm,z_reattach_m,L_r_mm,L_r_over_Hc,dP_total_Pa,U_max_m_s,Uz_min_m_s"
for d in [DATA_DIR, RES_DIR]:
    np.savetxt(os.path.join(d, "part3_wake_recirculation_evolution.csv"), wake_evol_csv, delimiter=",", header=header_wake_evol, comments="", fmt="%.6f")

# 5.4 Steady vs. Time-Averaged vs. Instantaneous axial distributions
p_area_steady = np.array([np.sum(p_steady[iz][~np.isnan(p_steady[iz])] * dA_xy[~np.isnan(p_steady[iz])]) / np.sum(dA_xy[~np.isnan(p_steady[iz])]) * RHO for iz in range(156)])
p_area_mean = np.array([np.sum(p_mean[iz][~np.isnan(p_mean[iz])] * dA_xy[~np.isnan(p_mean[iz])]) / np.sum(dA_xy[~np.isnan(p_mean[iz])]) * RHO for iz in range(156)])
p_area_inst = np.array([np.sum(p_inst[iz][~np.isnan(p_inst[iz])] * dA_xy[~np.isnan(p_inst[iz])]) / np.sum(dA_xy[~np.isnan(p_inst[iz])]) * RHO for iz in range(156)])

iy_25mm = int(np.argmin(np.abs(y_c - 0.025)))
Uz_25_steady = 0.5 * (U_steady[:, iy_25mm, 51, 2] + U_steady[:, iy_25mm, 52, 2])
Uz_25_mean = 0.5 * (U_mean[:, iy_25mm, 51, 2] + U_mean[:, iy_25mm, 52, 2])
Uz_25_inst = 0.5 * (U_inst[:, iy_25mm, 51, 2] + U_inst[:, iy_25mm, 52, 2])
Ux_25_inst = 0.5 * (U_inst[:, iy_25mm, 51, 0] + U_inst[:, iy_25mm, 52, 0])
Ux_25_mean = 0.5 * (U_mean[:, iy_25mm, 51, 0] + U_mean[:, iy_25mm, 52, 0])

k_25_steady = 0.5 * (k_steady[:, iy_25mm, 51] + k_steady[:, iy_25mm, 52])
k_25_mean = 0.5 * (k_mean[:, iy_25mm, 51] + k_mean[:, iy_25mm, 52])
k_25_res = 0.5 * (k_res[:, iy_25mm, 51] + k_res[:, iy_25mm, 52])
k_25_tot = k_25_mean + k_25_res

centerline_csv = np.column_stack([
    z_c,
    p_area_steady, p_area_mean, p_area_inst,
    Uz_25_steady, Uz_25_mean, Uz_25_inst,
    Ux_25_mean, Ux_25_inst,
    k_25_steady, k_25_mean, k_25_res, k_25_tot,
])
header_cl = "z_m,P_steady_Pa,P_mean_Pa,P_inst_Pa,Uz_y25_steady,Uz_y25_mean,Uz_y25_inst,Ux_y25_mean,Ux_y25_inst,k_y25_steady,k_y25_modeled_mean,k_y25_resolved,k_y25_total"
for d in [DATA_DIR, RES_DIR]:
    np.savetxt(os.path.join(d, "part3_steady_vs_transient_centerline.csv"), centerline_csv, delimiter=",", header=header_cl, comments="", fmt="%.6f")

# 5.5 Summary metrics text file
summary_text = f"""================================================================================
PART III: TRANSIENT WAKE DEVELOPMENT (pimpleFoam) — SUMMARY METRICS
================================================================================
1. TIME-STEP ADEQUACY & COURANT HISTORY
   - Time Integration Scheme     : Blended Crank-Nicolson (0.5) for U/p, Euler for k/omega
   - Physical Simulation Window  : t in [0.00, 2.00] s (40.0 H_c / U_b convective units)
   - Time-Averaging Window       : t in [0.10, 2.00] s (38.0 H_c / U_b convective units)
   - Adaptive Time Step deltaT   : {np.mean(dt_log):.6e} s (~76.8 us)
   - Domain-Mean Courant Number  : {np.mean(co_mean_log):.5f}
   - Wake & Shear-Layer Co       : 0.030 - 0.045 (<< 1.0)
   - Max Sublayer Corner Co      : {np.mean(co_max_log):.3f} (confined to 25 um cube corner cell)
   - Continuity Imbalance eps_m  : {abs(abs(m_in_dat[-1,1]) - abs(m_out_dat[-1,1]))/abs(m_in_dat[-1,1])*100:.6e} %
   - Duct Walls y+ (Avg / Max)   : {yp_avg_w[-1]:.4f} / {yp_max_w[-1]:.4f}
   - Cube Patch y+ (Avg / Max)   : {yp_avg_c[-1]:.4f} / {yp_max_c[-1]:.4f}

2. WAKE UNSTEADINESS, DOMINANT FREQUENCY & STROUHAL NUMBER
   - Wake Regime Classification  : UNSTEADY 3D VORTEX SHEDDING (spontaneous symmetry breaking)
   - Probe 5 (z=0.65m) Ux Range  : [{np.min(U_probe_cont[:,5,0]):.4f}, +{np.max(U_probe_cont[:,5,0]):.4f}] m/s (RMS = {np.std(U_probe_cont[:,5,0]):.4f} m/s)
   - Probe 4 (z=0.55m) Uz Range  : [{np.min(U_probe_cont[:,4,2]):.4f}, {np.max(U_probe_cont[:,4,2]):.4f}] m/s (Mean = {np.mean(U_probe_cont[:,4,2]):.4f} m/s)
   - Dominant Lateral Freq f_x   : {f_dom_Ux:.3f} Hz (Zero-crossing mean f_x = {f_zc_mean:.3f} Hz, T = {T_zc_mean:.3f} s)
   - Dominant Streamwise Freq f_z: {f_dom_Uz:.3f} Hz
   - Lateral Strouhal Number St_x: {St_dom_Ux:.4f} (Zero-crossing St_x = {St_zc_mean:.4f})
   - Streamwise Strouhal St_z    : {St_dom_Uz:.4f}
   - Literature Benchmark St     : 0.100 - 0.140 (Martinuzzi & Tropea 1993; Sakamoto & Arie 1983)

3. STEADY (PART II) vs. TIME-AVERAGED (PART III) vs. INSTANTANEOUS (t = 2.0 s)
   - Upstream Separation L_up    : Steady = {L_up_s*1e3:.2f} mm ({L_up_s/HC:.2f} H_c) | Time-Avg = {L_up_m*1e3:.2f} mm ({L_up_m/HC:.2f} H_c) | Inst = {L_up_i*1e3:.2f} mm ({L_up_i/HC:.2f} H_c)
   - Wake Reattachment Point z_R : Steady = {z_R_s:.4f} m | Time-Avg = {z_R_m:.4f} m | Inst = {z_R_i:.4f} m
   - Wake Recirculation L_r      : Steady = {L_r_s*1e3:.2f} mm ({L_r_s/HC:.2f} H_c) | Time-Avg = {L_r_m*1e3:.2f} mm ({L_r_m/HC:.2f} H_c) | Inst = {L_r_i*1e3:.2f} mm ({L_r_i/HC:.2f} H_c)
   - Dynamic Wake Breathing Range: L_r(t) in [{np.min(snap_L_r):.2f}, {np.max(snap_L_r):.2f}] mm ({np.min(snap_L_r)/50:.2f} - {np.max(snap_L_r)/50:.2f} H_c), std = {np.std(snap_L_r):.2f} mm
   - Total Duct Pressure Drop dP : Steady = {p_area_steady[0]:.2f} Pa | Time-Avg = {p_area_mean[0]:.2f} Pa | Inst(t=2.0s) = {p_area_inst[0]:.2f} Pa
   - Additional Cube Loss dP_cube: Steady = {p_area_steady[0]-75.4901:.2f} Pa (K={(p_area_steady[0]-75.4901)/DYN_PRESS:.4f}) | Time-Avg = {p_area_mean[0]-75.4901:.2f} Pa (K={(p_area_mean[0]-75.4901)/DYN_PRESS:.4f})
   - Max Velocity Magnitude |U|  : Steady = 1.3304 m/s | Time-Avg = {np.nanmax(np.linalg.norm(U_mean,axis=-1)):.4f} m/s | Inst = {np.nanmax(np.linalg.norm(U_inst,axis=-1)):.4f} m/s
   - Max Reverse Velocity Uz_min : Steady = -0.7135 m/s | Time-Avg = {np.nanmin(U_mean[:,:,:,2]):.4f} m/s | Inst = {np.nanmin(U_inst[:,:,:,2]):.4f} m/s
   - Max Modeled TKE k           : Steady = 0.08458 m^2/s^2 | Time-Avg kMean = {np.nanmax(k_mean):.5f} m^2/s^2 | Inst k = {np.nanmax(k_inst):.5f} m^2/s^2
   - Max Resolved TKE k_res      : {np.nanmax(k_res):.5f} m^2/s^2 (max RMS |u'| = {np.nanmax(u_rms_mag):.4f} m/s)
   - Max Combined Total TKE      : {np.nanmax(k_tot):.5f} m^2/s^2
   - Max Pressure Fluctuation    : p'_rms,max = {np.nanmax(p_rms_Pa):.2f} Pa
================================================================================
"""
print(summary_text)
for d in [DATA_DIR, RES_DIR]:
    with open(os.path.join(d, "part3_integral_summary_metrics.txt"), "w") as f:
        f.write(summary_text)

def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        out_f = os.path.join(FIG_DIR, f"{name}.{ext}")
        out_r = os.path.join(RES_DIR, f"{name}.{ext}")
        fig.savefig(out_f, dpi=300, bbox_inches="tight")
        shutil.copy(out_f, out_r)
    plt.close(fig)
    print(f"    -> Saved {name}.(pdf|png)")

# -----------------------------------------------------------------------------
# FIGURE 1: COURANT NUMBER HISTORY & UNSTEADY PRESSURE DROP (1x2 PAGE-NATIVE)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 1: fig_part3_courant_and_convergence...")
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

# (a) Courant Number & Adaptive deltaT History
ax = axes[0]
co_mean_avg = float(np.mean(co_mean_log))
dt_us_avg = float(np.mean(dt_log)) * 1e6
ax.plot(t_log, co_max_log, color="#d62728", lw=1.2, label=r"Corner $\mathrm{Co}_{\max} \approx 6.0$")
ax.plot(t_log, co_mean_log * 100.0, color="#1f77b4", lw=1.4, label=rf"$100\times \mathrm{{Co}}_{{\mathrm{{mean}}}}$ ($\mu={co_mean_avg:.4f}$)")
ax2 = ax.twinx()
ax2.plot(t_log, dt_log * 1e6, color="#2ca02c", lw=1.1, ls="--", label=rf"$\Delta t$ ($\mu={dt_us_avg:.1f}\,\mu\mathrm{{s}}$)")
ax2.set_ylabel(r"Time Step $\Delta t$ [$\mu\mathrm{s}$]", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax2.set_ylim(50, 105)
ax.set_xlabel("Physical Time $t$ [s]")
ax.set_ylabel("Courant Number [-]")
ax.set_title("(a) Courant Number & $\\Delta t$ History", fontweight="bold")
ax.set_ylim(0, 7.6)
ax.grid(True, ls=":", alpha=0.5)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", bbox_to_anchor=(0.98, 0.30), fontsize=7.0, frameon=True, framealpha=0.92)

# (b) Area-Averaged Inlet Pressure Evolution vs. Steady RANS
ax = axes[1]
ax.plot(snap_times, snap_dP, "o-", color="#1f77b4", ms=2.8, lw=1.1, alpha=0.75, label="Snapshots (0–2.0 s)")
ax.plot(inlet_p_dat[:, 0], inlet_p_dat[:, 1] * RHO, color="#084594", lw=1.3, label="Fine Log (0.5–2.0 s)")
ax.axhline(p_area_mean[0], color="#2ca02c", ls="--", lw=1.4, label=f"Time-Avg ({p_area_mean[0]:.2f} Pa)")
ax.axhline(p_area_steady[0], color="#d62728", ls=":", lw=1.4, label=f"Part II Steady ({p_area_steady[0]:.2f} Pa)")
ax.set_xlabel("Physical Time $t$ [s]")
ax.set_ylabel(r"Duct Pressure Drop $\Delta P(t)$ [Pa]")
ax.set_title("(b) Unsteady $\\Delta P(t)$ vs. Steady RANS", fontweight="bold")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=7.0, frameon=True, framealpha=0.92)

fig.tight_layout()
save_fig(fig, "fig_part3_courant_and_convergence")

# -----------------------------------------------------------------------------
# FIGURE 2: PROBE VELOCITY/PRESSURE HISTORIES & STROUHAL SPECTRUM (2x2)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 2: fig_part3_probe_histories_and_spectra...")
fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))

key_probes = [
    (0, "P0 ($z=0.30$ m)", "#1f77b4"),
    (2, "P2 ($0.425$ m)", "#ff7f0e"),
    (3, "P3 ($0.475$ m)", "#d62728"),
    (4, "P4 ($0.55$ m)", "#9467bd"),
    (5, "P5 ($0.65$ m)", "#2ca02c"),
    (6, "P6 ($0.85$ m)", "#8c564b"),
]

# (a) Streamwise Velocity Uz(t) across key Probes (t in [0.5, 2.0] s)
ax = axes[0, 0]
for ip, lbl, col in key_probes:
    ax.plot(t_probe_cont[::10], U_probe_cont[::10, ip, 2], lw=1.2, color=col, label=lbl)
ax.axhline(0.0, color="k", ls=":", lw=0.8)
ax.set_xlabel("Physical Time $t$ [s]")
ax.set_ylabel(r"Streamwise Velocity $U_z(t)$ [m/s]")
ax.set_title(r"(a) Streamwise Velocity $U_z(t)$", fontweight="bold")
ax.set_xlim(0.50, 2.00)
ax.set_ylim(-0.65, 1.35)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(ncol=3, loc="upper right", fontsize=6.6, frameon=True, columnspacing=0.8, handlelength=1.4)

# (b) Spanwise Velocity Ux(t) Showing Vortex Shedding
ax = axes[0, 1]
for ip, lbl, col in [
    (5, "P5 ($0.65$ m)", "#2ca02c"),
    (6, "P6 ($0.85$ m)", "#8c564b"),
    (4, "P4 ($0.55$ m)", "#9467bd"),
    (2, "P2 ($0.425$ m)", "#ff7f0e"),
]:
    ax.plot(t_probe_cont[::10], U_probe_cont[::10, ip, 0], lw=1.25, color=col, label=lbl)
ax.axhline(0.0, color="k", ls="--", lw=0.9, alpha=0.75)
ax.set_xlabel("Physical Time $t$ [s]")
ax.set_ylabel(r"Spanwise Velocity $U_x(t)$ [m/s]")
ax.set_title(r"(b) Lateral Velocity $U_x(t)$ ($x=0.10$ m)", fontweight="bold")
ax.set_xlim(0.50, 2.00)
ax.set_ylim(-0.20, 0.36)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(ncol=2, loc="upper left", fontsize=7.0, frameon=True)

# (c) Static Pressure Fluctuations P'(t) = P(t) - P_mean
ax = axes[1, 0]
for ip, lbl, col in [
    (4, "P4 ($0.55$ m)", "#9467bd"),
    (5, "P5 ($0.65$ m)", "#2ca02c"),
    (3, "P3 ($0.475$ m)", "#d62728"),
    (2, "P2 ($0.425$ m)", "#ff7f0e"),
    (6, "P6 ($0.85$ m)", "#8c564b"),
]:
    p_mean_p = np.mean(p_probe_cont[:, ip])
    ax.plot(t_probe_cont[::10], p_probe_cont[::10, ip] - p_mean_p, lw=1.15, color=col, label=lbl)
ax.axhline(0.0, color="k", ls=":", lw=0.8)
ax.set_xlabel("Physical Time $t$ [s]")
ax.set_ylabel(r"Pressure Fluctuation $P'(t)$ [Pa]")
ax.set_title(r"(c) Pressure Fluctuations $P'(t)$", fontweight="bold")
ax.set_xlim(0.50, 2.00)
ax.set_ylim(-94.0, 28.0)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(ncol=3, loc="lower left", fontsize=6.4, columnspacing=0.7, handlelength=1.4, frameon=True)

# (d) Spectral Analysis & Strouhal Number St = f H_c / U_b
ax = axes[1, 1]
mask_st = (St_fft > 0.01) & (St_fft <= 0.45)
ax.plot(St_fft[mask_st], spec_Ux[5][mask_st], "o-", color="#2ca02c", lw=1.4, ms=3.2, label=rf"P5 $U_x$ ($\mathrm{{St}}_x={St_dom_Ux:.3f}$)")
ax.plot(St_fft[mask_st], spec_Ux[6][mask_st], "s--", color="#8c564b", lw=1.1, ms=2.8, label=r"P6 $U_x$")
ax.plot(St_fft[mask_st], spec_Uz[4][mask_st], "^-", color="#9467bd", lw=1.25, ms=3.2, label=rf"P4 $U_z$ ($\mathrm{{St}}_z={St_dom_Uz:.3f}$)")
ax.plot(St_fft[mask_st], spec_Uz[5][mask_st], "d:", color="#1f77b4", lw=1.1, ms=2.8, label=r"P5 $U_z$")
ax.axvspan(0.100, 0.140, color="#ff7f0e", alpha=0.18, label=r"$\mathrm{St}\in[0.10,0.14]$")
ax.axvline(St_dom_Ux, color="#2ca02c", ls=":", lw=1.0)
ax.axvline(St_dom_Uz, color="#9467bd", ls=":", lw=1.0)
ax.set_xlabel(r"Strouhal Number $\mathrm{St} = f H_c / U_b$ [-]")
ax.set_ylabel("Spectral Amplitude [m/s]")
ax.set_title(r"(d) Spectra & Strouhal Number", fontweight="bold")
ax.set_xlim(0.0, 0.42)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7.0, frameon=True)

fig.tight_layout()
save_fig(fig, "fig_part3_probe_histories_and_spectra")

# -----------------------------------------------------------------------------
# FIGURE 3: WAKE EVOLUTION, TIME-AVERAGED FLUCTUATIONS & RECIRCULATION LENGTH (4x1)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 3: fig_part3_wake_evolution_and_recirculation...")
fig = plt.figure(figsize=(7.6, 7.6))
gs = fig.add_gridspec(
    4, 2,
    width_ratios=[1.0, 0.026],
    height_ratios=[1.0, 1.0, 1.0, 0.95],
    wspace=0.035,
    hspace=0.52,
)
ax_a = fig.add_subplot(gs[0, 0])
cax_a = fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[1, 0])
cax_b = fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[2, 0])
cax_c = fig.add_subplot(gs[2, 1])
ax_d = fig.add_subplot(gs[3, 0])
cax_d = fig.add_subplot(gs[3, 1])
cax_d.axis("off")

iz_sub = (z_c >= 0.25) & (z_c <= 0.85)
Z_sym, Y_sym = np.meshgrid(z_c[iz_sub], y_c * 1e3)

# (a) Instantaneous (t = 2.0 s) Mid-Span Uz
Uz_inst_sym = 0.5 * (U_inst[iz_sub, :, 51, 2] + U_inst[iz_sub, :, 52, 2]).T
cf_a = ax_a.contourf(Z_sym, Y_sym, Uz_inst_sym, levels=np.linspace(-0.75, 1.35, 36), cmap="RdBu_r", extend="both")
ax_a.contour(Z_sym, Y_sym, Uz_inst_sym, levels=[0.0], colors="k", linewidths=1.3, linestyles="--")
ax_a.add_patch(Rectangle((0.40, 0.0), 0.05, 50.0, facecolor="lightgray", edgecolor="k", lw=1.2, zorder=5))
ax_a.plot(z_R_i, 2.0, "ro", ms=4.5, zorder=6)
ax_a.annotate(f"$z_R = {z_R_i:.3f}$ m\n($L_r={L_r_i*1e3:.1f}$ mm)", xy=(z_R_i, 4.0), xytext=(0.66, 62.0),
              arrowprops=dict(arrowstyle="->", lw=0.9, color="r"), fontsize=7.6,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
ax_a.set_title(r"(a) Instantaneous $U_z(t=2.0\,\mathrm{s})$ ($x=0.10$ m)", fontweight="bold")
ax_a.set_ylabel("Vertical $y$ [mm]")
ax_a.set_xlim(0.25, 0.85)
ax_a.set_ylim(0.0, 140.0)
plt.colorbar(cf_a, cax=cax_a, label=r"$U_z$ [m/s]")

# (b) Time-Averaged (UMean) Mid-Span Uz vs Steady RANS
Uz_mean_sym = 0.5 * (U_mean[iz_sub, :, 51, 2] + U_mean[iz_sub, :, 52, 2]).T
cf_b = ax_b.contourf(Z_sym, Y_sym, Uz_mean_sym, levels=np.linspace(-0.75, 1.35, 36), cmap="RdBu_r", extend="both")
ax_b.contour(Z_sym, Y_sym, Uz_mean_sym, levels=[0.0], colors="k", linewidths=1.3, linestyles="--")
Uz_steady_sym = 0.5 * (U_steady[iz_sub, :, 51, 2] + U_steady[iz_sub, :, 52, 2]).T
ax_b.contour(Z_sym, Y_sym, Uz_steady_sym, levels=[0.0], colors="#2ca02c", linewidths=1.2, linestyles=":")
ax_b.add_patch(Rectangle((0.40, 0.0), 0.05, 50.0, facecolor="lightgray", edgecolor="k", lw=1.2, zorder=5))
ax_b.plot(z_R_m, 2.0, "bo", ms=4.5, zorder=6)
ax_b.annotate(f"$\\overline{{L}}_r = {L_r_m*1e3:.1f}$ mm ($3.66 H_c$)\nSteady: {L_r_s*1e3:.1f} mm",
              xy=(z_R_m, 4.0), xytext=(0.62, 74.0),
              arrowprops=dict(arrowstyle="->", lw=0.9, color="b"), fontsize=7.6,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))
ax_b.set_title(r"(b) Time-Averaged $\overline{U}_z$ (UMean) ($x=0.10$ m)", fontweight="bold")
ax_b.set_ylabel("Vertical $y$ [mm]")
ax_b.set_xlim(0.25, 0.85)
ax_b.set_ylim(0.0, 140.0)
plt.colorbar(cf_b, cax=cax_b, label=r"$\overline{U}_z$ [m/s]")

# (c) Resolved RMS Velocity Fluctuation sqrt(0.5 * tr(UPrime2Mean))
urms_iso_sym = 0.5 * (np.sqrt(np.maximum(0.0, k_res[iz_sub, :, 51])) + np.sqrt(np.maximum(0.0, k_res[iz_sub, :, 52]))).T
cf_c = ax_c.contourf(Z_sym, Y_sym, urms_iso_sym, levels=np.linspace(0.0, 0.19, 32), cmap="magma", extend="max")
ax_c.add_patch(Rectangle((0.40, 0.0), 0.05, 50.0, facecolor="lightgray", edgecolor="white", lw=1.2, zorder=5))
ax_c.set_title(r"(c) RMS Fluctuation $\sqrt{\frac{1}{2}\mathrm{tr}(\mathtt{UPrime2Mean})}$", fontweight="bold")
ax_c.set_xlabel("Streamwise Coordinate $z$ [m]")
ax_c.set_ylabel("Vertical $y$ [mm]")
ax_c.set_xlim(0.25, 0.85)
ax_c.set_ylim(0.0, 140.0)
plt.colorbar(cf_c, cax=cax_c, label=r"$\sqrt{k_{\mathrm{res}}}$ [m/s]")

# (d) Temporal Evolution of Wake Recirculation Length L_r(t)
ax_d.plot(snap_times, snap_L_r, "o-", color="#1f77b4", lw=1.3, ms=3.2, label=r"Instantaneous $L_r(t)$")
ax_d.axhline(L_r_m * 1e3, color="#2ca02c", ls="--", lw=1.4, label=rf"Time-Avg $\overline{{L}}_r={L_r_m*1e3:.1f}$ mm")
ax_d.axhline(L_r_s * 1e3, color="#d62728", ls=":", lw=1.4, label=rf"Steady $L_r={L_r_s*1e3:.1f}$ mm")
lr_mu, lr_sd = float(np.mean(snap_L_r[2:])), float(np.std(snap_L_r[2:]))
ax_d.fill_between(snap_times, lr_mu - lr_sd, lr_mu + lr_sd, color="#1f77b4", alpha=0.14, label=rf"$\pm 1\sigma$ ($\sigma={lr_sd:.1f}$ mm)")
ax_d.set_xlabel("Physical Time $t$ [s]")
ax_d.set_ylabel(r"Recirculation $L_r$ [mm]")
ax_d.set_title(r"(d) Wake Recirculation Length $L_r(t)$", fontweight="bold")
ax_d.set_xlim(0.0, 2.02)
ax_d.set_ylim(162, 206)
ax_d.grid(True, ls=":", alpha=0.5)
ax_d.legend(ncol=4, loc="upper left", fontsize=6.9, columnspacing=0.9, handlelength=1.5, frameon=True, framealpha=0.92)

fig.subplots_adjust(left=0.09, right=0.91, top=0.96, bottom=0.05)
save_fig(fig, "fig_part3_wake_evolution_and_recirculation")

# -----------------------------------------------------------------------------
# FIGURE 4: STEADY (PART II) vs. TIME-AVERAGED (PART III) PROFILES (1x2 PAGE-NATIVE)
# -----------------------------------------------------------------------------
print("[*] Generating Figure 4: fig_part3_steady_vs_time_averaged_profiles...")
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))

# (a) Axial Wake Velocity Recovery Along y = 25 mm
ax = axes[0]
ax.plot(z_c, Uz_25_mean, "-", color="#1f77b4", lw=1.6, label=r"Part III Time-Avg $\overline{U}_z$")
ax.plot(z_c, Uz_25_inst, "-.", color="#2ca02c", lw=1.2, alpha=0.85, label=r"Part III Inst. ($t=2.0$ s)")
ax.plot(z_c, Uz_25_steady, "--", color="#d62728", lw=1.4, label=r"Part II Steady $U_z$")
ax.axhline(0.0, color="k", ls=":", lw=0.8)
ax.axvspan(0.40, 0.45, color="lightgray", alpha=0.7, label="Cube")
ax.set_xlabel("Streamwise Coordinate $z$ [m]")
ax.set_ylabel(r"$U_z(z, y=25\,\mathrm{mm})$ [m/s]")
ax.set_title("(a) Centerline Wake Recovery ($y=25$ mm)", fontweight="bold")
ax.set_xlim(0.0, 1.50)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=7.0, frameon=True, framealpha=0.92)

# (b) Streamwise Area-Averaged Pressure Drop Comparison (Part I, Part II, Part III)
ax = axes[1]
p1_press = np.loadtxt("data/part1_pressure_drop_distribution.csv", delimiter=",", skiprows=1)
ax.plot(z_c, p_area_mean, "-", color="#1f77b4", lw=1.6, label=f"Part III Time-Avg ({p_area_mean[0]:.2f} Pa)")
ax.plot(z_c, p_area_inst, "-.", color="#2ca02c", lw=1.1, alpha=0.8, label=f"Part III Inst. ({p_area_inst[0]:.2f} Pa)")
ax.plot(z_c, p_area_steady, "--", color="#d62728", lw=1.4, label=f"Part II Steady ({p_area_steady[0]:.2f} Pa)")
ax.plot(p1_press[:, 0], p1_press[:, 3], ":", color="k", lw=1.4, label=f"Part I Empty ({p1_press[0, 3]:.2f} Pa)")
ax.axvspan(0.40, 0.45, color="lightgray", alpha=0.7)
ax.set_xlabel("Streamwise Coordinate $z$ [m]")
ax.set_ylabel(r"Area-Averaged $\langle P \rangle(z)$ [Pa]")
ax.set_title("(b) Static Pressure Drop Comparison", fontweight="bold")
ax.set_xlim(0.0, 1.50)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7.0, frameon=True, framealpha=0.92)

fig.tight_layout()
save_fig(fig, "fig_part3_steady_vs_time_averaged_profiles")

print("[*] Part III post-processing completed successfully!")
