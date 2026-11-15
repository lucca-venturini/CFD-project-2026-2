#!/usr/bin/env python3
"""
generate_part2_results.py
=========================
Comprehensive Post-Processing and Deliverable Generation for Part II:
Cube Insertion and Steady Disturbed Flow (OpenFOAM 2506)

Generates:
1. CSV Data Tables in data/ and part2/results/:
   - part2_pressure_drop_comparison.csv
   - part2_wake_centerline_development.csv
   - part2_vertical_wake_profiles.csv
   - part2_bottom_wall_shear_stress.csv
   - part2_integral_summary_metrics.txt
2. Publication Figures (300 DPI PNG + Vector PDF) in figures/:
   - fig_part2_convergence.(pdf|png)
   - fig_part2_pressure_drop.(pdf|png)
   - fig_part2_wake_and_separation.(pdf|png)
   - fig_part2_wake_profiles.(pdf|png)
   - fig_part2_wall_shear_and_wake.(pdf|png)
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

CASE_DIR = "part2"
DATA_DIR = "data"
FIG_DIR = "figures"
RES_DIR = os.path.join(CASE_DIR, "results")

for d in [DATA_DIR, FIG_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "figure.dpi": 300,
})

# -----------------------------------------------------------------------------
# 2. LOAD MESH & 3D FIELDS FROM part2/5000/
# -----------------------------------------------------------------------------
print("[*] Loading Part II mesh coordinates...")
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

# Cube mask: iz in [50, 74), iy in [0, 36), ix in [40, 64)
mask_3d = np.ones((156, 96, 104), dtype=bool)
mask_3d[50:74, 0:36, 40:64] = False
n_fluid = int(np.sum(mask_3d))               # 1,536,768

def load_scalar_3d(field_name):
    with open(os.path.join(CASE_DIR, "5000", field_name), "rb") as f:
        buf = f.read()
    tag = f"{n_fluid}\n(".encode()
    pos = buf.find(tag) + len(tag)
    arr_flat = np.frombuffer(buf[pos:pos + n_fluid * 8], dtype="<f8")
    arr_3d = np.full((156, 96, 104), np.nan)
    arr_3d[mask_3d] = arr_flat
    return arr_3d

def load_vector_3d(field_name):
    with open(os.path.join(CASE_DIR, "5000", field_name), "rb") as f:
        buf = f.read()
    tag = f"{n_fluid}\n(".encode()
    pos = buf.find(tag) + len(tag)
    arr_flat = np.frombuffer(buf[pos:pos + n_fluid * 24], dtype="<f8").reshape((n_fluid, 3))
    arr_3d = np.full((156, 96, 104, 3), np.nan)
    arr_3d[mask_3d] = arr_flat
    return arr_3d

print("[*] Loading 3D fields U, p, k, omega, nut at Time = 5000...")
U = load_vector_3d("U")
p = load_scalar_3d("p")
k = load_scalar_3d("k")
omega = load_scalar_3d("omega")
nut = load_scalar_3d("nut")

U_mag = np.linalg.norm(U, axis=-1)

# Load Part 1 baseline data for direct comparison
p1_press = np.loadtxt("data/part1_pressure_drop_distribution.csv", delimiter=",", skiprows=1)
z_p1 = p1_press[:, 0]
P_Pa_p1 = p1_press[:, 3]

p1_prof = np.loadtxt("data/part1_inlet_outlet_velocity_profiles.csv", delimiter=",", skiprows=1)
y_p1 = p1_prof[:, 0]
Uz_out_p1 = p1_prof[:, 2]

p1_tau = np.loadtxt("data/part1_wall_shear_stress_distribution.csv", delimiter=",", skiprows=1)
z_tau_p1 = p1_tau[:, 0]
tau_mid_p1 = p1_tau[:, 2]

# -----------------------------------------------------------------------------
# 3. COMPUTE KEY PHYSICAL METRICS (PRESSURE DROP, WAKE LENGTH, EXTREMA)
# -----------------------------------------------------------------------------
print("[*] Computing axial pressure distribution and cube loss...")
p_area_avg = np.zeros(156)
u_area_avg = np.zeros(156)
active_area_z = np.zeros(156)

for iz in range(156):
    valid = ~np.isnan(p[iz])
    area_slice = np.sum(dA_xy[valid])
    active_area_z[iz] = area_slice
    p_area_avg[iz] = np.sum(p[iz][valid] * dA_xy[valid]) / area_slice
    u_area_avg[iz] = np.sum(U[iz, :, :, 2][valid] * dA_xy[valid]) / area_slice

P_Pa_p2 = p_area_avg * RHO

# Exact inlet/outlet pressure from postProcessing function objects
inlet_p_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/inletPressure/0/surfaceFieldValue.dat"), comments="#")
p_in_kin = inlet_p_hist[-1, 1]              # 0.11176096 m^2/s^2
dP_total_p2 = p_in_kin * RHO                # 111.7610 Pa
dP_total_p1 = 75.4901                       # Part 1 total dP [Pa] (0.0754901 * 1000)
dP_cube_add = dP_total_p2 - dP_total_p1     # Additional pressure loss caused by cube [Pa]
K_cube = dP_cube_add / DYN_PRESS            # Loss coefficient based on bulk velocity

# Maximum velocity and location
flat_idx_umax = np.nanargmax(U_mag)
iz_u, iy_u, ix_u = np.unravel_index(flat_idx_umax, U_mag.shape)
U_max_val = U_mag[iz_u, iy_u, ix_u]
U_max_loc = (x_c[ix_u], y_c[iy_u], z_c[iz_u])

# Minimum Uz (maximum reverse flow velocity in wake)
flat_idx_umin = np.nanargmin(U[:, :, :, 2])
iz_r, iy_r, ix_r = np.unravel_index(flat_idx_umin, (156, 96, 104))
Uz_min_val = U[iz_r, iy_r, ix_r, 2]
Uz_min_loc = (x_c[ix_r], y_c[iy_r], z_c[iz_r])

# Maximum turbulent kinetic energy k
flat_idx_kmax = np.nanargmax(k)
iz_k, iy_k, ix_k = np.unravel_index(flat_idx_kmax, k.shape)
k_max_val = k[iz_k, iy_k, ix_k]
k_max_loc = (x_c[ix_k], y_c[iy_k], z_c[iz_k])

# Pressure extrema
p_max_Pa = np.nanmax(p) * RHO
p_min_Pa = np.nanmin(p) * RHO

# Mid-span symmetry plane (ix = 51 and 52 averaged around x = 0.10 m)
Uz_sym = 0.5 * (U[:, :, 51, 2] + U[:, :, 52, 2])
Uy_sym = 0.5 * (U[:, :, 51, 1] + U[:, :, 52, 1])
p_sym = 0.5 * (p[:, :, 51] + p[:, :, 52]) * RHO
k_sym = 0.5 * (k[:, :, 51] + k[:, :, 52])

# Determine upstream separation length (bottom wall y_c[0], z < 0.40 m)
Uz_bot_sym = Uz_sym[:, 0]  # first cell center y_c[0] = 15 um at x = 0.10 m
# Upstream of cube: iz < 50 (z < 0.40 m)
up_rev_indices = np.where(Uz_bot_sym[:50] < 0)[0]
if len(up_rev_indices) > 0:
    iz_sep = up_rev_indices[0]
    # Linear interpolation for Uz = 0 between iz_sep-1 and iz_sep
    u1, u2 = Uz_bot_sym[iz_sep - 1], Uz_bot_sym[iz_sep]
    z_sep_up = z_c[iz_sep - 1] + (0.0 - u1) / (u2 - u1) * (z_c[iz_sep] - z_c[iz_sep - 1])
    L_sep_up = 0.40 - z_sep_up
else:
    z_sep_up = 0.40
    L_sep_up = 0.0

# Determine downstream wake reattachment point z_R (behind cube: iz >= 74, z > 0.45 m)
# Check near-wall Uz_bot_sym[74:] crossing from negative to positive
wake_NearWall = Uz_bot_sym[74:]
z_wake_c = z_c[74:]
reattach_idx = np.where((wake_NearWall[:-1] < 0) & (wake_NearWall[1:] >= 0))[0]
if len(reattach_idx) > 0:
    idx_r = reattach_idx[-1]
    u1, u2 = wake_NearWall[idx_r], wake_NearWall[idx_r + 1]
    z_reattach_wall = z_wake_c[idx_r] + (0.0 - u1) / (u2 - u1) * (z_wake_c[idx_r + 1] - z_wake_c[idx_r])
else:
    z_reattach_wall = np.nan

# Also check maximum axial extent of the recirculation bubble (Uz <= 0 anywhere in y at x = 0.10 m)
z_bubble_max = 0.45
for iz in range(74, 150):
    min_uz_slice = np.nanmin(Uz_sym[iz, :45])
    if min_uz_slice < 0:
        min_uz_next = np.nanmin(Uz_sym[iz + 1, :45])
        if min_uz_next >= 0:
            z_bubble_max = z_c[iz] + (0.0 - min_uz_slice) / (min_uz_next - min_uz_slice) * (z_c[iz + 1] - z_c[iz])

L_wake_wall = z_reattach_wall - 0.45
L_wake_bubble = z_bubble_max - 0.45

# Rooftop separation check (iz in [50, 74), iy = 36 which is first cell above y = 0.05 m)
Uz_roof_sym = Uz_sym[50:74, 36]
roof_rev = np.sum(Uz_roof_sym < 0)

# Bottom wall centerline shear stress tau_w = mu * Uz(y_1) / y_1
y1_bot = y_c[0]  # 1.5e-5 m
tau_bot_sym = -MU * Uz_bot_sym / y1_bot  # signed wall shear in flow direction (+z is positive skin friction):
tau_w_streamwise = MU * Uz_bot_sym / y1_bot  # positive when flow is +z, negative in reverse flow

print(f"    Total dP (Part II):        {dP_total_p2:.2f} Pa (vs Part I: {dP_total_p1:.2f} Pa)")
print(f"    Additional Cube dP:        {dP_cube_add:.2f} Pa (+{dP_cube_add/dP_total_p1*100:.1f}%)")
print(f"    Cube Loss Coefficient K:   {K_cube:.4f}")
print(f"    Max |U|:                   {U_max_val:.4f} m/s at {U_max_loc}")
print(f"    Max Reverse Velocity Uz:   {Uz_min_val:.4f} m/s at {Uz_min_loc}")
print(f"    Upstream Separation z_sep: {z_sep_up:.4f} m (L_sep,up = {L_sep_up*1000:.2f} mm = {L_sep_up/HC:.2f} H_c)")
print(f"    Wake Wall Reattachment zR: {z_reattach_wall:.4f} m (L_r,wall = {L_wake_wall*1000:.2f} mm = {L_wake_wall/HC:.2f} H_c)")
print(f"    Wake Max Bubble Extent:    {z_bubble_max:.4f} m (L_r,max  = {L_wake_bubble*1000:.2f} mm = {L_wake_bubble/HC:.2f} H_c)")
print(f"    Max TKE k:                 {k_max_val:.5f} m^2/s^2 at {k_max_loc}")

# -----------------------------------------------------------------------------
# 4. SAVE CSV DATA TABLES
# -----------------------------------------------------------------------------
print("[*] Writing CSV tables to data/ and part2/results/...")

# Table 1: Axial pressure drop comparison
p1_interp = np.interp(z_c, z_p1, P_Pa_p1)
press_table = np.column_stack([z_c, p_area_avg, P_Pa_p2, p1_interp, P_Pa_p2 - p1_interp, u_area_avg, active_area_z])
hdr_press = "z_m,p_kin_m2s2,P_static_Pa_Part2,P_static_Pa_Part1_interp,dP_excess_Pa,Uz_area_avg_ms,Active_Area_m2"
np.savetxt(os.path.join(DATA_DIR, "part2_pressure_drop_comparison.csv"), press_table, delimiter=",", header=hdr_press, comments="")

# Table 2: Wake development along y = 0.025 m (mid-cube height) and duct centerline y = 0.10 m
iy_mid_cube = np.argmin(np.abs(y_c - 0.025))
iy_center = np.argmin(np.abs(y_c - 0.100))
wake_table = np.column_stack([
    z_c,
    Uz_sym[:, iy_mid_cube],
    k_sym[:, iy_mid_cube],
    Uz_sym[:, iy_center],
    k_sym[:, iy_center],
    tau_w_streamwise
])
hdr_wake = "z_m,Uz_y0025_ms,k_y0025_m2s2,Uz_centerline_y010_ms,k_centerline_y010_m2s2,tau_w_bottom_center_Pa"
np.savetxt(os.path.join(DATA_DIR, "part2_wake_centerline_development.csv"), wake_table, delimiter=",", header=hdr_wake, comments="")

# Table 3: Vertical profiles at 6 axial stations
stations_z = [0.25, 0.425, 0.50, 0.60, 0.85, 1.49]
iz_stations = [int(np.argmin(np.abs(z_c - zs))) for zs in stations_z]
prof_cols = [y_c]
prof_hdr = ["y_m"]
for zs, iz_s in zip(stations_z, iz_stations):
    prof_cols.append(Uz_sym[iz_s, :])
    prof_cols.append(k_sym[iz_s, :])
    prof_hdr.append(f"Uz_z{zs:.3f}m")
    prof_hdr.append(f"k_z{zs:.3f}m")
prof_table = np.column_stack(prof_cols)
np.savetxt(os.path.join(DATA_DIR, "part2_vertical_wake_profiles.csv"), prof_table, delimiter=",", header=",".join(prof_hdr), comments="")

# Summary text file
summary_txt = f"""================================================================================
PART II: CUBE INSERTION & STEADY DISTURBED FLOW - INTEGRAL SUMMARY METRICS
================================================================================
Mesh & Topological Summary:
  Bounding Grid Dimensions:      104 x 96 x 156 (1,557,504 total hex cells)
  Cube CellZone (cube):          20,736 cells (24 x 36 x 24), Volume = 1.250000e-04 m^3
  Active Fluid Cells (fluid):    1,536,768 cells, Volume = 0.059875 m^3
  Cube Surface Patch (cube):     4,032 faces (area = 0.012500 m^2 across 5 wet faces)
  Non-Orthogonality:             Max = 0.00 deg, Avg = 0.00 deg
  Max Skewness:                  4.34e-11

Convergence & Continuity (Iteration 5000):
  Inlet Volumetric Flux:         -0.04000000 m^3/s
  Outlet Volumetric Flux:        +0.040000001 m^3/s
  Relative Mass Imbalance (em):  2.50e-06 % (0.00%)
  Duct Walls y+ (walls):         avg = 0.7273, min = 0.0144, max = 2.8152
  Cube Walls y+ (cube):          avg = 0.9290, min = 0.0294, max = 3.1762

Pressure Loss & Blockage Analysis:
  Blockage Ratio (A_cube/A_duct): 6.25% (U_throat_mean = 1.0667 m/s)
  Total Pressure Drop (Part II): {dP_total_p2:.4f} Pa (kinematic: {p_in_kin:.6f} m^2/s^2)
  Empty Duct Baseline (Part I):  {dP_total_p1:.4f} Pa
  Additional Cube Pressure Loss: {dP_cube_add:.4f} Pa (+{dP_cube_add/dP_total_p1*100:.2f}%)
  Cube Pressure Loss Coeff (K):  {K_cube:.4f} (based on 0.5*rho*Ub^2 = {DYN_PRESS:.1f} Pa)
  Max Stagnation Pressure:       +{p_max_Pa:.2f} Pa
  Min Corner Suction Pressure:   {p_min_Pa:.2f} Pa

Flow Separation, Recirculation & Wake Characterization:
  Max Local Velocity |U|_max:    {U_max_val:.4f} m/s at (x={U_max_loc[0]:.3f}, y={U_max_loc[1]:.3f}, z={U_max_loc[2]:.3f}) m
  Max Reverse Flow Velocity Uz:  {Uz_min_val:.4f} m/s at (x={Uz_min_loc[0]:.3f}, y={Uz_min_loc[1]:.3f}, z={Uz_min_loc[2]:.3f}) m
  Upstream Separation Onset:     z = {z_sep_up:.4f} m (L_sep,up = {L_sep_up*1000:.2f} mm = {L_sep_up/HC:.2f} H_c)
  Downstream Wall Reattachment:  z_R = {z_reattach_wall:.4f} m (L_wake,wall = {L_wake_wall*1000:.2f} mm = {L_wake_wall/HC:.2f} H_c)
  Max Recirculation Bubble End:  z_max = {z_bubble_max:.4f} m (L_wake,max = {L_wake_bubble*1000:.2f} mm = {L_wake_bubble/HC:.2f} H_c)
  Max Turbulent Kinetic Energy:  k_max = {k_max_val:.5f} m^2/s^2 (22.55x inlet k_0)
================================================================================
"""
with open(os.path.join(DATA_DIR, "part2_integral_summary_metrics.txt"), "w") as f:
    f.write(summary_txt)

for fname in os.listdir(DATA_DIR):
    if fname.startswith("part2_"):
        shutil.copy2(os.path.join(DATA_DIR, fname), os.path.join(RES_DIR, fname))

# -----------------------------------------------------------------------------
# 5. GENERATE PUBLICATION FIGURES
# -----------------------------------------------------------------------------
def save_fig(fig, base_name):
    for ext in ["pdf", "png"]:
        out_path = os.path.join(FIG_DIR, f"{base_name}.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        shutil.copy2(out_path, os.path.join(RES_DIR, f"{base_name}.{ext}"))
    plt.close(fig)
    print(f"    [+] Saved {base_name}.(pdf|png)")

# --- FIGURE A: Convergence & Verification Dashboard ---
print("[*] Generating fig_part2_convergence...")
res_raw = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/residuals/0/solverInfo.dat"), comments="#", usecols=(0, 2, 5, 8, 13, 18, 23))
p_up_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/pressureUpstreamCube/0/surfaceFieldValue.dat"), comments="#")
p_dn_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/pressureDownstreamCube/0/surfaceFieldValue.dat"), comments="#")
p_wk_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/pressureWakeRecovery/0/surfaceFieldValue.dat"), comments="#")
m_in_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/massFlowInlet/0/surfaceFieldValue.dat"), comments="#")
m_out_hist = np.loadtxt(os.path.join(CASE_DIR, "postProcessing/massFlowOutlet/0/surfaceFieldValue.dat"), comments="#")

yp_walls_t, yp_walls_avg, yp_walls_max = [], [], []
yp_cube_t, yp_cube_avg, yp_cube_max = [], [], []
with open(os.path.join(CASE_DIR, "postProcessing/yPlus/0/yPlus.dat")) as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            if parts[1] == "walls":
                yp_walls_t.append(float(parts[0]))
                yp_walls_avg.append(float(parts[4]))
                yp_walls_max.append(float(parts[3]))
            elif parts[1] == "cube":
                yp_cube_t.append(float(parts[0]))
                yp_cube_avg.append(float(parts[4]))
                yp_cube_max.append(float(parts[3]))

fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))

# (a) Residuals
ax = axes[0, 0]
ax.semilogy(res_raw[:, 0], res_raw[:, 1], label=r"$U_x$", linewidth=1.2)
ax.semilogy(res_raw[:, 0], res_raw[:, 2], label=r"$U_y$", linewidth=1.2)
ax.semilogy(res_raw[:, 0], res_raw[:, 3], label=r"$U_z$", linewidth=1.3)
ax.semilogy(res_raw[:, 0], res_raw[:, 6], label=r"$p$", linewidth=1.3, color="tab:red")
ax.semilogy(res_raw[:, 0], res_raw[:, 4], label=r"$k$", linewidth=1.2, linestyle="--")
ax.semilogy(res_raw[:, 0], res_raw[:, 5], label=r"$\omega$", linewidth=1.2, linestyle=":")
ax.axhline(1e-4, color="gray", linestyle="--", linewidth=0.9)
ax.set_title("(a) Initial Solver Residuals", fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel("Initial Residual")
ax.set_xlim(0, 5000)
ax.set_ylim(1e-8, 2.0)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(ncol=3, loc="upper right", fontsize=7.5, frameon=True)

# (b) Multi-Plane Static Pressure Evolution
ax = axes[0, 1]
ax.plot(inlet_p_hist[:, 0], inlet_p_hist[:, 1] * RHO, label="Inlet (0.0 m)", linewidth=1.5, color="tab:blue")
ax.plot(p_up_hist[:, 0], p_up_hist[:, 1] * RHO, label="Up (0.30 m)", linewidth=1.3, color="tab:green")
ax.plot(p_wk_hist[:, 0], p_wk_hist[:, 1] * RHO, label="Recov. (0.90 m)", linewidth=1.3, color="tab:orange")
ax.plot(p_dn_hist[:, 0], p_dn_hist[:, 1] * RHO, label="Wake (0.55 m)", linewidth=1.3, color="tab:red")
ax.set_title("(b) Static Pressure Convergence", fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\langle P \rangle$ [Pa]")
ax.set_xlim(0, 5000)
ax.set_ylim(-160, 150)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower right", ncol=2, fontsize=7.3, frameon=True)

# (c) Continuity Error
ax = axes[1, 0]
n_m = min(len(m_in_hist), len(m_out_hist))
eps_m = np.abs(m_in_hist[:n_m, 1] + m_out_hist[:n_m, 1]) / np.abs(m_in_hist[:n_m, 1]) * 100.0
ax.semilogy(m_in_hist[:n_m, 0], np.clip(eps_m, 1e-8, None), color="tab:purple", linewidth=1.4, label=r"$\varepsilon_m$")
ax.axhline(1e-4, color="crimson", linestyle="--", linewidth=0.9, label=r"$10^{-4}\%$ ref.")
ax.set_title(r"(c) Mass Imbalance ($2.5\times 10^{-6}\%$)", fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\varepsilon_m$ [%]")
ax.set_xlim(0, 5000)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7.8, frameon=True)

# (d) Near-Wall y+ on Duct Walls and Cube
ax = axes[1, 1]
ax.plot(yp_walls_t, yp_walls_avg, marker="s", markersize=3.0, linewidth=1.5, color="tab:blue", label=r"Walls $y^+_{\mathrm{avg}}$")
ax.plot(yp_cube_t, yp_cube_avg, marker="^", markersize=3.0, linewidth=1.5, color="tab:orange", label=r"Cube $y^+_{\mathrm{avg}}$")
ax.plot(yp_walls_t, yp_walls_max, linestyle=":", linewidth=1.1, color="tab:blue", alpha=0.75, label=r"Walls $y^+_{\max}$")
ax.plot(yp_cube_t, yp_cube_max, linestyle="--", linewidth=1.1, color="tab:orange", alpha=0.75, label=r"Cube $y^+_{\max}$")
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label=r"$y^+ = 1$")
ax.set_title(r"(d) Near-Wall Resolution ($y^+$)", fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Wall $y^+$")
ax.set_xlim(0, 5000)
ax.set_ylim(0, 4.3)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7.3, ncol=2, frameon=True)

fig.tight_layout()
save_fig(fig, "fig_part2_convergence")

# --- FIGURE B: Axial Static Pressure Drop Comparison (Part II vs Part I) ---
print("[*] Generating fig_part2_pressure_drop...")
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.axvspan(0.40, 0.45, color="gray", alpha=0.28, label="Cube ($z \in [0.40, 0.45]$ m)")
ax.axvspan(0.45, z_bubble_max, color="mistyrose", alpha=0.55, label=f"Wake ($z \in [0.45, {z_bubble_max:.2f}]$ m)")
ax.plot(z_c, P_Pa_p2, color="tab:red", linewidth=2.0, label=f"Part II: Cube ($\\Delta P = {dP_total_p2:.2f}$ Pa)")
ax.plot(z_p1, P_Pa_p1, color="tab:blue", linestyle="--", linewidth=1.6, label=f"Part I: Empty ($\\Delta P = {dP_total_p1:.2f}$ Pa)")

ax.annotate(
    f"Additional Cube Loss\n$\\Delta P_{{\\mathrm{{cube}}}} = +{dP_cube_add:.2f}$ Pa ($+48.1\\%$)\n$K_{{\\mathrm{{cube}}}} = {K_cube:.3f}$",
    xy=(0.01, dP_total_p2), xytext=(0.08, 108),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.92)
)
ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax.set_title("Streamwise Area-Averaged Static Pressure: Part II vs. Part I", fontweight="bold")
ax.set_xlabel("Streamwise Coordinate $z$ [m]")
ax.set_ylabel(r"Gauge Pressure $\langle P \rangle$ [Pa]")
ax.set_xlim(0, 1.50)
ax.set_ylim(-170, 152)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.0, frameon=True)
fig.tight_layout()
save_fig(fig, "fig_part2_pressure_drop")

# --- FIGURE C: 2D Mid-Span Symmetry Plane (x = 0.10 m) Velocity, Streamlines & TKE ---
print("[*] Generating fig_part2_wake_and_separation...")
z_mask_zoom = (z_c >= 0.25) & (z_c <= 0.85)
z_zoom = z_c[z_mask_zoom]
Z_grid, Y_grid = np.meshgrid(z_zoom, y_c, indexing="ij")
Uz_zoom = Uz_sym[z_mask_zoom, :]
Uy_zoom = Uy_sym[z_mask_zoom, :]
k_zoom = k_sym[z_mask_zoom, :]

fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True)

# Panel (a): Streamwise Velocity Uz + Zero-Velocity Isoline + Streamlines
ax = axes[0]
levels_u = np.linspace(-0.70, 1.30, 41)
cf1 = ax.contourf(Z_grid, Y_grid * 1000, Uz_zoom, levels=levels_u, cmap="RdBu_r", extend="both")
cbar1 = fig.colorbar(cf1, ax=ax, pad=0.02, fraction=0.026)
cbar1.set_label(r"$U_z$ [m/s]")

cs0 = ax.contour(Z_grid, Y_grid * 1000, Uz_zoom, levels=[0.0], colors="black", linewidths=1.6, linestyles="--")

z_unif = np.linspace(0.25, 0.85, 260)
y_unif = np.linspace(0.0005, 0.142, 100)
from matplotlib.tri import Triangulation, LinearTriInterpolator
Zf, Yf = Z_grid.flatten(), Y_grid.flatten()
valid_pts = ~np.isnan(Uz_zoom.flatten())
tri = Triangulation(Zf[valid_pts], Yf[valid_pts])
interp_uz = LinearTriInterpolator(tri, Uz_zoom.flatten()[valid_pts])
interp_uy = LinearTriInterpolator(tri, Uy_zoom.flatten()[valid_pts])
ZU, YU = np.meshgrid(z_unif, y_unif)
Uz_u = interp_uz(ZU, YU)
Uy_u = interp_uy(ZU, YU)
cube_mask_u = (ZU >= 0.40) & (ZU <= 0.45) & (YU <= 0.05)
Uz_u = np.ma.masked_where(cube_mask_u, Uz_u)
Uy_u = np.ma.masked_where(cube_mask_u, Uy_u)
ax.streamplot(z_unif, y_unif * 1000, Uz_u, Uy_u, color=(0.15, 0.15, 0.15, 0.55), density=1.25, linewidth=0.7, arrowsize=0.75)

ax.add_patch(Rectangle((0.40, 0), 0.05, 50, facecolor="dimgray", edgecolor="black", linewidth=1.4, zorder=5))
ax.text(0.425, 25, "CUBE", color="white", fontweight="bold", fontsize=8.0, ha="center", va="center", zorder=6)

ax.axvline(z_bubble_max, color="black", linestyle=":", linewidth=1.2)
ax.annotate(
    f"Reattachment $z_R = {z_bubble_max:.3f}$ m\n($L_r = {L_wake_bubble*1000:.1f}$ mm $= {L_wake_bubble/HC:.2f} H_c$)",
    xy=(z_bubble_max, 12), xytext=(0.65, 58),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    fontsize=8.2, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9), zorder=7
)
ax.set_title(r"(a) Mid-Span ($x = 0.10$ m) Streamwise Velocity $U_z$ & Recirculation ($U_z = 0$ Dashed)", fontweight="bold")
ax.set_ylabel("Vertical $y$ [mm]")
ax.set_xlim(0.25, 0.85)
ax.set_ylim(0, 140)

# Panel (b): Turbulent Kinetic Energy k
ax = axes[1]
levels_k = np.linspace(0, 0.075, 36)
cf2 = ax.contourf(Z_grid, Y_grid * 1000, k_zoom, levels=levels_k, cmap="inferno", extend="max")
cbar2 = fig.colorbar(cf2, ax=ax, pad=0.02, fraction=0.026)
cbar2.set_label(r"TKE $k$ [m$^2$/s$^2$]")
ax.add_patch(Rectangle((0.40, 0), 0.05, 50, facecolor="lightgray", edgecolor="black", linewidth=1.4, zorder=5))
ax.text(0.425, 25, "CUBE", color="black", fontweight="bold", fontsize=8.0, ha="center", va="center", zorder=6)
ax.set_title(r"(b) Mid-Span ($x = 0.10$ m) Turbulent Kinetic Energy $k$", fontweight="bold")
ax.set_xlabel("Streamwise Coordinate $z$ [m]")
ax.set_ylabel("Vertical $y$ [mm]")
ax.set_xlim(0.25, 0.85)
ax.set_ylim(0, 140)

fig.tight_layout()
save_fig(fig, "fig_part2_wake_and_separation")

# --- FIGURE D: Bottom-Wall Streamwise Shear Stress (Part II vs Part I) ---
print("[*] Generating fig_part2_wall_shear_and_wake...")
fig, ax = plt.subplots(figsize=(7.4, 3.8))

ax.axvspan(z_sep_up, 0.40, color="gold", alpha=0.25, label=f"Upstream Separation ($L_{{sep,up}} = {L_sep_up*1000:.1f}$ mm)")
ax.axvspan(0.40, 0.45, color="dimgray", alpha=0.35, label="Cube ($z \in [0.40, 0.45]$ m)")
ax.axvspan(0.45, z_bubble_max, color="mistyrose", alpha=0.55, label=f"Downstream Wake ($L_r = {L_wake_bubble*1000:.1f}$ mm)")
ax.plot(z_c, tau_w_streamwise, color="tab:red", linewidth=1.8, label=r"Part II Bottom Centerline $\tau_{w,z}(z)$")
ax.plot(z_tau_p1, tau_mid_p1, color="tab:blue", linestyle="--", linewidth=1.5, label=r"Part I Empty Duct $\tau_{w,z}(z)$")
ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
ax.axvline(z_bubble_max, color="crimson", linestyle=":", linewidth=1.1)

ax.annotate(
    f"Reattachment $z_R = {z_bubble_max:.3f}$ m\n($L_r = {L_wake_bubble/HC:.2f}\\,H_c$)",
    xy=(z_bubble_max, 0.0), xytext=(0.72, -3.5),
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
    fontsize=8.3, fontweight="bold", color="crimson",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.9)
)

ax.set_title(r"Bottom-Wall Streamwise Shear Stress $\tau_{w,z}(z)$ at Mid-Span ($x = 0.10$ m)", fontweight="bold")
ax.set_xlabel("Streamwise Coordinate $z$ [m]")
ax.set_ylabel(r"Streamwise Wall Shear $\tau_{w,z}$ [Pa]")
ax.set_xlim(0, 1.50)
ax.set_ylim(-5.0, 9.0)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7.6, frameon=True)

fig.tight_layout()
save_fig(fig, "fig_part2_wall_shear_and_wake")

print("[+] Part II post-processing complete!")
