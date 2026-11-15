#!/usr/bin/env python3
"""
generate_part4_results.py
=========================
Comprehensive Post-Processing and Deliverable Generation for Part IV:
Steady Conjugate Heat Transfer (chtMultiRegionSimpleFoam, OpenFOAM 2506)

Reads reconstructed Time = 15000 multi-region binary fields (fluid & solid)
and merged postProcessing logs (0/, 1250/, 10000/) to produce:
1. CSV Data Tables in data/ and part4/results/:
   - part4_summary_metrics.csv
   - part4_convergence_and_energy_balance.csv
   - part4_interface_face_statistics.csv
   - part4_thermal_wake_profiles.csv
2. Publication Figures (300 DPI PNG + Vector PDF) in figures/:
   - fig_part4_convergence_and_energy_balance.(pdf|png)
   - fig_part4_temperature_fields_and_solid.(pdf|png)
   - fig_part4_interface_flux_and_thermal_wake.(pdf|png)
"""

import os
import re
import glob
import shutil
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# -----------------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS & ANALYTICAL REFERENCES (PDF Sections 6 & 9)
# -----------------------------------------------------------------------------
RHO_F = 1000.0        # Water density [kg/m^3] (part4/constant/fluid/thermophysicalProperties)
CP_F = 4184.0         # Water specific heat [J/(kg.K)]
MU_F = 8.90e-4        # Water dynamic viscosity [Pa.s]
PR_F = 6.135          # Water Prandtl number
KAPPA_F = MU_F * CP_F / PR_F  # ~0.607 W/(m.K)

RHO_S = 8000.0        # Solid density [kg/m^3] (part4/constant/solid/thermophysicalProperties)
CP_S = 500.0          # Solid specific heat [J/(kg.K)]
KAPPA_S = 50.0        # Solid thermal conductivity [W/(m.K)]

U_BULK = 1.00         # Inlet bulk velocity [m/s]
W_DUCT, H_DUCT, L_DUCT = 0.20, 0.20, 1.50
HC = 0.05             # Cube edge length [m]
A_DUCT = W_DUCT * H_DUCT              # 0.04 m^2
MDOT_IN = RHO_F * U_BULK * A_DUCT     # 40.0 kg/s
V_SOLID_EXACT = HC ** 3               # 1.25e-4 m^3
A_INTERFACE_EXACT = 5.0 * (HC ** 2)   # 0.0125 m^2

Q_VOL = 2.68e10                       # fvOptions source [W/m^3]
Q_FVOPTIONS = Q_VOL * V_SOLID_EXACT   # 3.3500e6 W
T_INLET = 298.15                      # Inlet water temperature [K]
T_BOIL = 373.15                       # Water saturation temperature at 1 atm [K]
DT_TARGET = 20.0                      # Target mixed-mean temperature rise [K]
Q_ANALYTICAL_20K = MDOT_IN * CP_F * DT_TARGET  # 3.3472e6 W
EPS_SOURCE = abs(Q_FVOPTIONS - Q_ANALYTICAL_20K) / Q_ANALYTICAL_20K
DT_EXACT_FV = Q_FVOPTIONS / (MDOT_IN * CP_F)   # 20.01673 K

CASE_DIR = "part4"
TIME_DIR = "15000"
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
# 2. BINARY OPENFOAM MESH & FIELD PARSERS (REGION-SAFE)
# -----------------------------------------------------------------------------
def parse_boundary_dict(boundary_path):
    with open(boundary_path, "r") as f:
        txt = f.read()
    patches = {}
    for m in re.finditer(r"(\S+)\s*\n\s*\{([^}]*)\}", txt):
        name = m.group(1)
        body = m.group(2)
        nf_m = re.search(r"nFaces\s+(\d+)\s*;", body)
        sf_m = re.search(r"startFace\s+(\d+)\s*;", body)
        if nf_m and sf_m:
            patches[name] = {
                "nFaces": int(nf_m.group(1)),
                "startFace": int(sf_m.group(1)),
            }
    return patches


def load_region_mesh(region):
    poly_dir = os.path.join(CASE_DIR, "constant", region, "polyMesh")
    with open(os.path.join(poly_dir, "points"), "rb") as f:
        b = f.read()
    m = re.search(rb"\n(\d+)\n\(", b)
    n_pts = int(m.group(1))
    pts = np.frombuffer(b[m.end():m.end() + n_pts * 24], dtype="<f8").reshape((n_pts, 3))

    with open(os.path.join(poly_dir, "faces"), "rb") as f:
        b = f.read()
    m1 = re.search(rb"\n(\d+)\n\(", b)
    n_off = int(m1.group(1))
    offsets = np.frombuffer(b[m1.end():m1.end() + n_off * 4], dtype="<i4")
    pos2 = m1.end() + n_off * 4
    m2 = re.search(rb"\n(\d+)\n\(", b[pos2:pos2 + 64])
    n_verts = int(m2.group(1))
    verts = np.frombuffer(b[pos2 + m2.end():pos2 + m2.end() + n_verts * 4], dtype="<i4")

    with open(os.path.join(poly_dir, "owner"), "rb") as f:
        b = f.read()
    m = re.search(rb"\n(\d+)\n\(", b)
    n_faces = int(m.group(1))
    owner = np.frombuffer(b[m.end():m.end() + n_faces * 4], dtype="<i4")

    with open(os.path.join(poly_dir, "neighbour"), "rb") as f:
        b = f.read()
    m = re.search(rb"\n(\d+)\n\(", b)
    n_int = int(m.group(1))
    neighbour = np.frombuffer(b[m.end():m.end() + n_int * 4], dtype="<i4")

    counts = np.diff(offsets)
    if not np.all(counts == 4):
        raise ValueError(f"Expected quad faces in {region}, got {np.unique(counts)}")

    v4 = verts.reshape((n_faces, 4))
    p4 = pts[v4]
    fc = p4.mean(axis=1)
    fa = 0.5 * np.cross(p4[:, 2] - p4[:, 0], p4[:, 3] - p4[:, 1])
    fmag = np.linalg.norm(fa, axis=1)

    n_cells = int(max(owner.max(), neighbour.max())) + 1
    cell_sum = np.zeros((n_cells, 3), dtype=np.float64)
    cell_cnt = np.zeros(n_cells, dtype=np.float64)
    np.add.at(cell_sum, owner, fc)
    np.add.at(cell_cnt, owner, 1.0)
    np.add.at(cell_sum, neighbour, fc[:n_int])
    np.add.at(cell_cnt, neighbour, 1.0)
    cc = cell_sum / cell_cnt[:, None]

    # Exact orthogonal grid axes from point coordinates
    x_faces = np.unique(np.round(pts[:, 0], 8))
    y_faces = np.unique(np.round(pts[:, 1], 8))
    z_faces = np.unique(np.round(pts[:, 2], 8))
    x_c = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_c = 0.5 * (y_faces[:-1] + y_faces[1:])
    z_c = 0.5 * (z_faces[:-1] + z_faces[1:])

    ix = np.argmin(np.abs(cc[:, 0:1] - x_c[None, :]), axis=1)
    iy = np.argmin(np.abs(cc[:, 1:2] - y_c[None, :]), axis=1)
    iz = np.argmin(np.abs(cc[:, 2:3] - z_c[None, :]), axis=1)

    # Verify max snapping error is sub-micron
    max_err = max(
        np.max(np.abs(cc[:, 0] - x_c[ix])),
        np.max(np.abs(cc[:, 1] - y_c[iy])),
        np.max(np.abs(cc[:, 2] - z_c[iz])),
    )
    if max_err > 1e-6:
        raise RuntimeError(f"Grid snapping error in {region}: {max_err}")

    patches = parse_boundary_dict(os.path.join(poly_dir, "boundary"))
    return {
        "n_cells": n_cells,
        "cc": cc,
        "fc": fc,
        "fmag": fmag,
        "x_faces": x_faces,
        "y_faces": y_faces,
        "z_faces": z_faces,
        "x_c": x_c,
        "y_c": y_c,
        "z_c": z_c,
        "ix": ix,
        "iy": iy,
        "iz": iz,
        "patches": patches,
    }


def load_internal_scalar(region, field_name, n_cells):
    path = os.path.join(CASE_DIR, TIME_DIR, region, field_name)
    with open(path, "rb") as f:
        b = f.read()
    m = re.search(rf"\n{n_cells}\n\(".encode(), b)
    if not m:
        raise ValueError(f"Could not find internal scalar list in {path}")
    return np.frombuffer(b[m.end():m.end() + n_cells * 8], dtype="<f8").copy()


def load_internal_vector(region, field_name, n_cells):
    path = os.path.join(CASE_DIR, TIME_DIR, region, field_name)
    with open(path, "rb") as f:
        b = f.read()
    m = re.search(rf"\n{n_cells}\n\(".encode(), b)
    if not m:
        raise ValueError(f"Could not find internal vector list in {path}")
    return np.frombuffer(b[m.end():m.end() + n_cells * 24], dtype="<f8").reshape((n_cells, 3)).copy()


def load_patch_scalar(region, field_name, patch_name, n_faces, key="value"):
    path = os.path.join(CASE_DIR, TIME_DIR, region, field_name)
    with open(path, "rb") as f:
        b = f.read()
    p_start = b.find(patch_name.encode())
    if p_start < 0:
        raise ValueError(f"Patch {patch_name} not found in {path}")
    p_end = b.find(b"\n    }", p_start)
    block = b[p_start:p_end]
    m = re.search(rf"{key}\s+nonuniform List<scalar>\s*\n{n_faces}\n\(".encode(), block)
    if not m:
        raise ValueError(f"Key '{key}' list ({n_faces}) not found in {patch_name} of {path}")
    return np.frombuffer(block[m.end():m.end() + n_faces * 8], dtype="<f8").copy()


def to_grid(mesh, flat_arr, fill=np.nan):
    nx, ny, nz = len(mesh["x_c"]), len(mesh["y_c"]), len(mesh["z_c"])
    grid = np.full((nz, ny, nx), fill, dtype=np.float64)
    grid[mesh["iz"], mesh["iy"], mesh["ix"]] = flat_arr
    return grid


# -----------------------------------------------------------------------------
# 3. MULTI-SEGMENT POSTPROCESSING READER
# -----------------------------------------------------------------------------
def load_merged_series(rel_path, is_solver_info=False):
    """Merge segmented runs (0/, 1250/, 10000/) chronologically; later start overwrites."""
    base = os.path.join(CASE_DIR, "postProcessing", rel_path)
    segs = []
    for d in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(d):
            try:
                t_val = float(os.path.basename(d))
                segs.append((t_val, d))
            except ValueError:
                pass
    segs.sort(key=lambda x: x[0])

    merged = {}
    for _, seg_dir in segs:
        files = glob.glob(os.path.join(seg_dir, "*.dat")) + glob.glob(os.path.join(seg_dir, "T")) + \
                glob.glob(os.path.join(seg_dir, "p")) + glob.glob(os.path.join(seg_dir, "U"))
        if not files:
            continue
        fpath = sorted(files)[0]
        with open(fpath, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if is_solver_info:
                    toks = s.split()
                    try:
                        step = float(toks[0])
                        merged[step] = toks
                    except (ValueError, IndexError):
                        pass
                else:
                    merged[s.split()[0]] = s
    # Sort numerically by time/iteration
    items = []
    for k, v in merged.items():
        items.append((float(k), v))
    items.sort(key=lambda x: x[0])
    return items


def parse_scalar_dat(rel_path):
    items = load_merged_series(rel_path, is_solver_info=False)
    steps, vals = [], []
    for t, line in items:
        toks = line.split()
        steps.append(t)
        vals.append([float(x) for x in toks[1:]])
    return np.array(steps), np.array(vals)


def parse_wall_heat_flux_dat():
    base = os.path.join(CASE_DIR, "postProcessing", "fluid", "wallHeatFlux")
    segs = sorted(
        [(float(os.path.basename(d)), d) for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)],
        key=lambda x: x[0],
    )
    merged = {}
    for _, seg_dir in segs:
        fpath = os.path.join(seg_dir, "wallHeatFlux.dat")
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                if len(toks) >= 5 and toks[1] == "fluid_to_solid":
                    merged[float(toks[0])] = (float(toks[2]), float(toks[3]), float(toks[4]))
    steps = sorted(merged.keys())
    arr = np.array([merged[k] for k in steps])
    return np.array(steps), arr


# -----------------------------------------------------------------------------
# 4. LOAD PART IV MESHES & FIELDS AT ITERATION 15000
# -----------------------------------------------------------------------------
print("[*] Loading Part IV fluid & solid meshes and Time = 15000 binary fields...")
mesh_f = load_region_mesh("fluid")
mesh_s = load_region_mesh("solid")
print(f"    Fluid cells: {mesh_f['n_cells']:,}, Solid cells: {mesh_s['n_cells']:,}")

T_f_flat = load_internal_scalar("fluid", "T", mesh_f["n_cells"])
p_f_flat = load_internal_scalar("fluid", "p", mesh_f["n_cells"])
U_f_flat = load_internal_vector("fluid", "U", mesh_f["n_cells"])
T_s_flat = load_internal_scalar("solid", "T", mesh_s["n_cells"])

T_f_3d = to_grid(mesh_f, T_f_flat)
p_f_3d = to_grid(mesh_f, p_f_flat)
U_mag_3d = to_grid(mesh_f, np.linalg.norm(U_f_flat, axis=1))
Uz_3d = to_grid(mesh_f, U_f_flat[:, 2])
T_s_3d = to_grid(mesh_s, T_s_flat)

# Mid-plane slices at x = 0.10 m (average of two symmetric columns straddling x = 0.10 m)
ix_f_pair = np.argsort(np.abs(mesh_f["x_c"] - 0.10))[:2]
ix_s_pair = np.argsort(np.abs(mesh_s["x_c"] - 0.10))[:2]
print(f"    Fluid x-mid pair: {mesh_f['x_c'][ix_f_pair]}")
print(f"    Solid x-mid pair: {mesh_s['x_c'][ix_s_pair]}")

T_f_mid = np.where(np.all(np.isnan(T_f_3d[:, :, ix_f_pair]), axis=2), np.nan, np.nanmean(np.nan_to_num(T_f_3d[:, :, ix_f_pair], nan=0.0), axis=2))
p_f_mid = np.where(np.all(np.isnan(p_f_3d[:, :, ix_f_pair]), axis=2), np.nan, np.nanmean(np.nan_to_num(p_f_3d[:, :, ix_f_pair], nan=0.0), axis=2))
Uz_mid = np.where(np.all(np.isnan(Uz_3d[:, :, ix_f_pair]), axis=2), np.nan, np.nanmean(np.nan_to_num(Uz_3d[:, :, ix_f_pair], nan=0.0), axis=2))
T_s_mid = np.mean(T_s_3d[:, :, ix_s_pair], axis=2)      # (24, 36)

# Combine solid + fluid into seamless mid-plane grid (156, 96)
iz_cube_start = np.argmin(np.abs(mesh_f["z_c"] - mesh_s["z_c"][0]))
iz_cube_end = iz_cube_start + len(mesh_s["z_c"])
iy_cube_end = len(mesh_s["y_c"])
T_comb_mid = T_f_mid.copy()
T_comb_mid[iz_cube_start:iz_cube_end, :iy_cube_end] = T_s_mid

# Coupled patch data (fluid_to_solid)
f2s_info = mesh_f["patches"]["fluid_to_solid"]
sf, nf = f2s_info["startFace"], f2s_info["nFaces"]
fc_int = mesh_f["fc"][sf:sf + nf]
area_int = mesh_f["fmag"][sf:sf + nf]

whf_int = load_patch_scalar("fluid", "wallHeatFlux", "fluid_to_solid", nf, key="value")
T_int_f = load_patch_scalar("fluid", "T", "fluid_to_solid", nf, key="value")
yplus_int = load_patch_scalar("fluid", "yPlus", "fluid_to_solid", nf, key="value")

walls_info = mesh_f["patches"]["walls"]
yplus_walls = load_patch_scalar("fluid", "yPlus", "walls", walls_info["nFaces"], key="value")
area_walls = mesh_f["fmag"][walls_info["startFace"]:walls_info["startFace"] + walls_info["nFaces"]]


# -----------------------------------------------------------------------------
# 5. FACE-BY-FACE INTERFACE HEAT FLUX & TEMPERATURE BREAKDOWN
# -----------------------------------------------------------------------------
print("[*] Computing face-by-face coupled interface statistics...")
face_masks = [
    ("Front (z = 0.40 m)",  np.abs(fc_int[:, 2] - 0.40) < 1e-6),
    ("Top (y = 0.05 m)",    np.abs(fc_int[:, 1] - 0.05) < 1e-6),
    ("Left (x = 0.075 m)",  np.abs(fc_int[:, 0] - 0.075) < 1e-6),
    ("Right (x = 0.125 m)", np.abs(fc_int[:, 0] - 0.125) < 1e-6),
    ("Back (z = 0.45 m)",   np.abs(fc_int[:, 2] - 0.45) < 1e-6),
]

Q_total_int = float(np.sum(whf_int * area_int))
A_total_int = float(np.sum(area_int))
T_int_avg_all = float(np.sum(T_int_f * area_int) / A_total_int)

face_rows = []
for fname, mask in face_masks:
    Ai = float(np.sum(area_int[mask]))
    Qi = float(np.sum(whf_int[mask] * area_int[mask]))
    q_avg = Qi / Ai
    T_avg = float(np.sum(T_int_f[mask] * area_int[mask]) / Ai)
    h_loc = q_avg / (T_avg - T_INLET)
    face_rows.append({
        "face": fname,
        "n_faces": int(np.sum(mask)),
        "area_m2": Ai,
        "Q_W": Qi,
        "Q_MW": Qi / 1e6,
        "share_pct": Qi / Q_total_int * 100.0,
        "q_avg_MWm2": q_avg / 1e6,
        "q_min_MWm2": float(whf_int[mask].min() / 1e6),
        "q_max_MWm2": float(whf_int[mask].max() / 1e6),
        "T_avg_K": T_avg,
        "T_min_K": float(T_int_f[mask].min()),
        "T_max_K": float(T_int_f[mask].max()),
        "h_eff_Wm2K": h_loc,
    })

face_rows.append({
    "face": "Total / Mean",
    "n_faces": nf,
    "area_m2": A_total_int,
    "Q_W": Q_total_int,
    "Q_MW": Q_total_int / 1e6,
    "share_pct": 100.0,
    "q_avg_MWm2": (Q_total_int / A_total_int) / 1e6,
    "q_min_MWm2": float(whf_int.min() / 1e6),
    "q_max_MWm2": float(whf_int.max() / 1e6),
    "T_avg_K": T_int_avg_all,
    "T_min_K": float(T_int_f.min()),
    "T_max_K": float(T_int_f.max()),
    "h_eff_Wm2K": (Q_total_int / A_total_int) / (T_int_avg_all - T_INLET),
})

csv_face_path = os.path.join(DATA_DIR, "part4_interface_face_statistics.csv")
with open(csv_face_path, "w") as f:
    f.write("face,n_faces,area_m2,Q_MW,share_pct,q_avg_MWm2,q_min_MWm2,q_max_MWm2,T_avg_K,T_min_K,T_max_K,h_eff_Wm2K\n")
    for r in face_rows:
        f.write(
            f"{r['face']},{r['n_faces']},{r['area_m2']:.6f},{r['Q_MW']:.6f},{r['share_pct']:.4f},"
            f"{r['q_avg_MWm2']:.4f},{r['q_min_MWm2']:.4f},{r['q_max_MWm2']:.4f},"
            f"{r['T_avg_K']:.2f},{r['T_min_K']:.2f},{r['T_max_K']:.2f},{r['h_eff_Wm2K']:.2f}\n"
        )
shutil.copy2(csv_face_path, os.path.join(RES_DIR, "part4_interface_face_statistics.csv"))


# -----------------------------------------------------------------------------
# 6. PARSE CONVERGENCE HISTORY ACROSS ALL RESTARTS (1..15000)
# -----------------------------------------------------------------------------
print("[*] Merging postProcessing logs across segments (0/, 1250/, 10000/)...")
res_f_items = load_merged_series("fluid/residualsFluid", is_solver_info=True)
steps_res_f = np.array([t for t, _ in res_f_items])
# Column indices verified from header:
# 0:Time 2:Ux_init 5:Uy_init 8:Uz_init 13:h_init 18:k_init 23:omega_init 28:p_rgh_init
Ux_init = np.array([float(tok[2]) for _, tok in res_f_items])
Uz_init = np.array([float(tok[8]) for _, tok in res_f_items])
h_f_init = np.array([float(tok[13]) for _, tok in res_f_items])
k_init = np.array([float(tok[18]) for _, tok in res_f_items])
omega_init = np.array([float(tok[23]) for _, tok in res_f_items])
prgh_init = np.array([float(tok[28]) for _, tok in res_f_items])

res_s_items = load_merged_series("solid/residualsSolid", is_solver_info=True)
steps_res_s = np.array([t for t, _ in res_s_items])
h_s_init = np.array([float(tok[2]) for _, tok in res_s_items])

steps_tin, val_tin = parse_scalar_dat("fluid/inletTemperatureMix")
steps_tout, val_tout = parse_scalar_dat("fluid/outletTemperatureMix")
steps_min, val_min = parse_scalar_dat("fluid/massFlowInlet")
steps_mout, val_mout = parse_scalar_dat("fluid/massFlowOutlet")
steps_pin, val_pin = parse_scalar_dat("fluid/inletPressure")
steps_pout, val_pout = parse_scalar_dat("fluid/outletPressure")
steps_pu, val_pu = parse_scalar_dat("fluid/pressureUpstreamCube")
steps_pd, val_pd = parse_scalar_dat("fluid/pressureDownstreamCube")
steps_intf, val_intf = parse_scalar_dat("fluid/interfaceTempFluid")
steps_savg, val_savg = parse_scalar_dat("solid/solidTempAvg")
steps_whf, val_whf = parse_wall_heat_flux_dat()

# FieldMinMax parsing for solidTempExtrema and fluidTempExtrema
def parse_minmax_T(rel_path):
    base = os.path.join(CASE_DIR, "postProcessing", rel_path)
    segs = sorted(
        [(float(os.path.basename(d)), d) for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)],
        key=lambda x: x[0],
    )
    merged = {}
    for _, seg_dir in segs:
        fpath = os.path.join(seg_dir, "fieldMinMax.dat")
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                if len(toks) < 3 or toks[1] != "T":
                    continue
                t = float(toks[0])
                after_paren = s.split(")")[1]
                nums1 = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+", s.split("(")[0])
                nums2 = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+", after_paren)
                if nums1 and nums2:
                    merged[t] = (float(nums1[0]), float(nums2[0]))
    steps = sorted(merged.keys())
    return (
        np.array(steps),
        np.array([merged[k][0] for k in steps]),
        np.array([merged[k][1] for k in steps]),
    )


def parse_probes_scalar(rel_dir, fname="T"):
    base = os.path.join(CASE_DIR, "postProcessing", rel_dir)
    segs = sorted(
        [(float(os.path.basename(d)), d) for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)],
        key=lambda x: x[0],
    )
    merged = {}
    for _, seg_dir in segs:
        fpath = os.path.join(seg_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = s.split()
                merged[float(toks[0])] = [float(x) for x in toks[1:]]
    steps = sorted(merged.keys())
    return np.array(steps), np.array([merged[k] for k in steps])


def parse_probes_vector(rel_dir, fname="U"):
    base = os.path.join(CASE_DIR, "postProcessing", rel_dir)
    segs = sorted(
        [(float(os.path.basename(d)), d) for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)],
        key=lambda x: x[0],
    )
    merged = {}
    for _, seg_dir in segs:
        fpath = os.path.join(seg_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                t = float(s.split()[0])
                vecs = re.findall(r"\(([^)]+)\)", s)
                merged[t] = [[float(x) for x in v.split()] for v in vecs]
    steps = sorted(merged.keys())
    return np.array(steps), np.array([merged[k] for k in steps])


steps_s_ext, T_s_min_hist, T_s_max_hist = parse_minmax_T("solid/solidTempExtrema")
steps_f_ext, T_f_min_hist, T_f_max_hist = parse_minmax_T("fluid/fluidTempExtrema")
steps_fprobes, T_fprobes = parse_probes_scalar("fluidProbes/fluid", "T")
steps_fprobes_p, p_fprobes = parse_probes_scalar("fluidProbes/fluid", "p")
steps_fprobes_u, U_fprobes = parse_probes_vector("fluidProbes/fluid", "U")
steps_sprobes, T_sprobes = parse_probes_scalar("solidProbes/solid", "T")

# Build aligned convergence CSV on tout steps (every 10 iterations)
mdot_in_final = float(val_min[-1, 0])
mdot_out_final = float(val_mout[-1, 0])
eps_mass = abs(abs(mdot_in_final) - abs(mdot_out_final)) / abs(mdot_in_final)

T_in_mix_hist = np.interp(steps_tout, steps_tin, val_tin[:, 0])
h_in_mix_hist = np.interp(steps_tout, steps_tin, val_tin[:, 1])
T_out_mix_hist = val_tout[:, 0]
h_out_mix_hist = val_tout[:, 1]
dT_mix_hist = T_out_mix_hist - T_in_mix_hist
mdot_out_hist = np.interp(steps_tout, steps_mout, val_mout[:, 0])
mdot_in_hist = np.interp(steps_tout, steps_min, val_min[:, 0])
Q_rem_MW_hist = (mdot_out_hist * (h_out_mix_hist - h_in_mix_hist)) / 1e6
T_int_avg_hist = np.interp(steps_tout, steps_intf, val_intf[:, 0])
T_s_avg_hist = np.interp(steps_tout, steps_savg, val_savg[:, 0])
T_s_max_interp = np.interp(steps_tout, steps_s_ext, T_s_max_hist)
Q_int_MW_hist = np.interp(steps_whf, steps_whf, val_whf[:, 2]) / 1e6

csv_conv_path = os.path.join(DATA_DIR, "part4_convergence_and_energy_balance.csv")
with open(csv_conv_path, "w") as f:
    f.write("step,T_out_mix_K,dT_mix_K,Q_removed_MW,T_int_avg_K,T_solid_avg_K,T_solid_max_K,mdot_in_kgs,mdot_out_kgs\n")
    for i in range(len(steps_tout)):
        f.write(
            f"{int(steps_tout[i])},{T_out_mix_hist[i]:.6f},{dT_mix_hist[i]:.6f},"
            f"{Q_rem_MW_hist[i]:.6f},{T_int_avg_hist[i]:.2f},{T_s_avg_hist[i]:.2f},"
            f"{T_s_max_interp[i]:.2f},{mdot_in_hist[i]:.6f},{mdot_out_hist[i]:.6f}\n"
        )
shutil.copy2(csv_conv_path, os.path.join(RES_DIR, "part4_convergence_and_energy_balance.csv"))


# -----------------------------------------------------------------------------
# 7. SUMMARY METRICS & THERMAL WAKE PROFILES AT ITERATION 15000
# -----------------------------------------------------------------------------
Q_removed_final = float(mdot_out_final * (h_out_mix_hist[-1] - h_in_mix_hist[-1]))
eps_energy_fv = abs(Q_FVOPTIONS - Q_removed_final) / Q_FVOPTIONS
eps_energy_ref = abs(Q_ANALYTICAL_20K - Q_removed_final) / Q_ANALYTICAL_20K
eps_int_fv = abs(Q_total_int - Q_FVOPTIONS) / Q_FVOPTIONS

P_in_final = float(val_pin[-1, 0])
P_out_final = float(val_pout[-1, 0])
P_up_cube = float(val_pu[-1, 0])
P_dn_cube = float(val_pd[-1, 0])
dP_duct = P_in_final - P_out_final
dP_cube = P_up_cube - P_dn_cube

yplus_walls_arith = float(np.mean(yplus_walls))
yplus_walls_avg = float(np.sum(yplus_walls * area_walls) / np.sum(area_walls))
yplus_int_arith = float(np.mean(yplus_int))
yplus_int_avg = float(np.sum(yplus_int * area_int) / np.sum(area_int))

# Exact solid volume check
dx_s = np.diff(mesh_s["x_faces"])
dy_s = np.diff(mesh_s["y_faces"])
dz_s = np.diff(mesh_s["z_faces"])
V_s_mesh = float(np.sum(dx_s) * np.sum(dy_s) * np.sum(dz_s))

summary_rows = [
    ("V_solid_mesh [m^3]", f"{V_s_mesh:.8e}"),
    ("A_interface_mesh [m^2]", f"{A_total_int:.8e}"),
    ("q_volumetric_fvOptions [W/m^3]", f"{Q_VOL:.6e}"),
    ("Q_fvOptions [MW]", f"{Q_FVOPTIONS/1e6:.6f}"),
    ("Q_analytical_20K_target [MW]", f"{Q_ANALYTICAL_20K/1e6:.6f}"),
    ("eps_source_discrepancy [%]", f"{EPS_SOURCE*100:.4f}"),
    ("dT_analytical_from_Q_fvOptions [K]", f"{DT_EXACT_FV:.4f}"),
    ("mdot_in [kg/s]", f"{mdot_in_final:.7f}"),
    ("mdot_out [kg/s]", f"{mdot_out_final:.7f}"),
    ("eps_mass_imbalance [%]", f"{eps_mass*100:.6e}"),
    ("T_in_mix [K]", f"{T_in_mix_hist[-1]:.4f}"),
    ("T_out_mix [K]", f"{T_out_mix_hist[-1]:.4f}"),
    ("dT_mix [K]", f"{dT_mix_hist[-1]:.4f}"),
    ("Q_interface [MW]", f"{Q_total_int/1e6:.6f}"),
    ("Q_removed_coolant [MW]", f"{Q_removed_final/1e6:.6f}"),
    ("eps_Q_interface_vs_fvOptions [%]", f"{eps_int_fv*100:.4f}"),
    ("eps_Q_removed_vs_fvOptions [%]", f"{eps_energy_fv*100:.4f}"),
    ("eps_Q_removed_vs_20K_ref [%]", f"{eps_energy_ref*100:.4f}"),
    ("T_interface_avg [K]", f"{T_int_avg_all:.2f}"),
    ("T_solid_volAvg [K]", f"{T_s_avg_hist[-1]:.2f}"),
    ("T_solid_max [K]", f"{T_s_max_hist[-1]:.2f}"),
    ("T_solid_min [K]", f"{T_s_min_hist[-1]:.2f}"),
    ("T_fluid_max [K]", f"{T_f_max_hist[-1]:.2f}"),
    ("T_fluid_min [K]", f"{T_f_min_hist[-1]:.2f}"),
    ("dP_duct_total [Pa]", f"{dP_duct:.4f}"),
    ("dP_across_cube [Pa]", f"{dP_cube:.4f}"),
    ("yPlus_walls_arith_avg", f"{yplus_walls_arith:.4f}"),
    ("yPlus_walls_area_avg", f"{yplus_walls_avg:.4f}"),
    ("yPlus_walls_max", f"{yplus_walls.max():.4f}"),
    ("yPlus_cube_arith_avg", f"{yplus_int_arith:.4f}"),
    ("yPlus_cube_area_avg", f"{yplus_int_avg:.4f}"),
    ("yPlus_cube_max", f"{yplus_int.max():.4f}"),
]

csv_sum_path = os.path.join(DATA_DIR, "part4_summary_metrics.csv")
with open(csv_sum_path, "w") as f:
    f.write("metric,value\n")
    for k, v in summary_rows:
        f.write(f"{k},{v}\n")
shutil.copy2(csv_sum_path, os.path.join(RES_DIR, "part4_summary_metrics.csv"))

# Vertical thermal wake profiles at x = 0.10 m
wake_z_targets = [
    ("z0425_center", 0.425),
    ("z0475_0.5Hc",  0.475),
    ("z0550_2.0Hc",  0.550),
    ("z0650_4.0Hc",  0.650),
    ("z1000_11Hc",   1.000),
    ("z1400_19Hc",   1.400),
]
wake_cols = {}
z_actuals = {}
for label, zt in wake_z_targets:
    iz = int(np.argmin(np.abs(mesh_f["z_c"] - zt)))
    z_actuals[label] = float(mesh_f["z_c"][iz])
    wake_cols[label] = T_comb_mid[iz, :]

csv_wake_path = os.path.join(DATA_DIR, "part4_thermal_wake_profiles.csv")
with open(csv_wake_path, "w") as f:
    hdr = "y_m," + ",".join([f"{lbl}_T_K,{lbl}_dT_K" for lbl, _ in wake_z_targets]) + "\n"
    f.write(hdr)
    for iy in range(len(mesh_f["y_c"])):
        vals_str = [f"{mesh_f['y_c'][iy]:.6f}"]
        for lbl, _ in wake_z_targets:
            tv = wake_cols[lbl][iy]
            vals_str.append(f"{tv:.4f}")
            vals_str.append(f"{tv - T_INLET:.4f}")
        f.write(",".join(vals_str) + "\n")
shutil.copy2(csv_wake_path, os.path.join(RES_DIR, "part4_thermal_wake_profiles.csv"))


# -----------------------------------------------------------------------------
# 8. FIGURE 1: ENERGY BALANCE & SOLID/INTERFACE TEMPERATURE CONVERGENCE (1x2)
# -----------------------------------------------------------------------------
print("[*] Generating Fig 1: Energy Balance & Solid Temperature History...")
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))

# Panel (a): Global steady energy balance + mixed-mean dT
ax = axes[0]
l1, = ax.plot(steps_tout, Q_rem_MW_hist, color="#1f77b4", lw=1.6, label=r"$Q_{\mathrm{removed}} = \dot{m}\,\Delta h_{\mathrm{mix}}$")
l2, = ax.plot(steps_whf, val_whf[:, 2] / 1e6, color="#ff7f0e", lw=1.4, ls="-", label=r"$Q_{\mathrm{interface}}$")
l3 = ax.axhline(Q_FVOPTIONS / 1e6, color="#d62728", ls="--", lw=1.2, label=r"$Q_{\mathrm{fvOptions}} = 3.350\,\mathrm{MW}$")
ax.set_xlabel("SIMPLE Iteration")
ax.set_ylabel("Thermal Power [MW]", color="#1f77b4")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax.set_xlim(0, 15000)
ax.set_ylim(0.0, 3.85)
ax.grid(True, ls=":", lw=0.5, alpha=0.5)
ax.set_title("(a) Energy Balance & Mixed-Mean $\\Delta T_{\\mathrm{mix}}$", fontweight="bold")

ax_r = ax.twinx()
l4, = ax_r.plot(steps_tout, dT_mix_hist, color="#2ca02c", ls="-.", lw=1.3, label=r"$\Delta T_{\mathrm{mix}}$ [K]")
ax_r.axhline(DT_TARGET, color="#2ca02c", ls=":", lw=0.8, alpha=0.7)
ax_r.set_ylabel(r"Mixed-Mean Rise $\Delta T_{\mathrm{mix}}$ [K]", color="#2ca02c")
ax_r.tick_params(axis="y", labelcolor="#2ca02c")
ax_r.set_ylim(0.0, 23.0)
lines_b = [l1, l2, l3, l4]
labels_b = [l.get_label() for l in lines_b]
ax.legend(lines_b, labels_b, loc="lower right", frameon=True, framealpha=0.92, fontsize=6.9)
ax.text(
    4200, 2.30,
    f"Iter 15,000:\n"
    f"$Q_{{rem}} = {Q_removed_final/1e6:.4f}$ MW\n"
    f"$\\Delta T_{{mix}} = {dT_mix_hist[-1]:.2f}$ K\n"
    f"$\\varepsilon_Q = {eps_energy_fv*100:.2f}\\%$",
    fontsize=7.3,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.92),
)

# Panel (b): Solid & interface temperature evolution
ax = axes[1]
ax.plot(steps_s_ext, T_s_max_hist / 1e3, color="#d62728", lw=1.6, label=rf"$T_{{\mathrm{{solid}},\max}}$ ({T_s_max_hist[-1]/1e3:.1f} kK)")
ax.plot(steps_savg, val_savg[:, 0] / 1e3, color="#ff7f0e", lw=1.5, label=rf"$T_{{\mathrm{{solid,avg}}}}$ ({val_savg[-1, 0]/1e3:.1f} kK)")
ax.plot(steps_intf, val_intf[:, 0] / 1e3, color="#1f77b4", lw=1.5, label=rf"$T_{{\mathrm{{int,avg}}}}$ ({val_intf[-1, 0]/1e3:.1f} kK)")
ax.plot(steps_f_ext, T_f_max_hist / 1e3, color="#2ca02c", ls="--", lw=1.3, label=rf"$T_{{\mathrm{{fluid}},\max}}$ ({T_f_max_hist[-1]/1e3:.1f} kK)")
ax.set_xlabel("SIMPLE Iteration")
ax.set_ylabel("Temperature [$10^3$ K]")
ax.set_title("(b) Solid & Interface Thermal Convergence", fontweight="bold")
ax.set_xlim(0, 15000)
ax.set_ylim(0, 205)
ax.grid(True, ls=":", lw=0.5, alpha=0.5)
ax.legend(loc="center right", frameon=True, framealpha=0.92, fontsize=7.2)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_part4_convergence_and_energy_balance.{ext}"), bbox_inches="tight")
    shutil.copy2(os.path.join(FIG_DIR, f"fig_part4_convergence_and_energy_balance.{ext}"), os.path.join(RES_DIR, f"fig_part4_convergence_and_energy_balance.{ext}"))
plt.close(fig)


# -----------------------------------------------------------------------------
# 9. FIGURE 2: MID-PLANE CHT PLUME & SOLID INTERNAL TEMPERATURE (2x1)
# -----------------------------------------------------------------------------
print("[*] Generating Fig 2: Multi-Region Temperature Field & Solid Conduction...")
fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), gridspec_kw={"height_ratios": [1.0, 1.15]})

# Panel (a): Log10(T - T_in) across fluid + solid at x = 0.10 m
ax = axes[0]
z_mask = (mesh_f["z_c"] >= 0.30) & (mesh_f["z_c"] <= 0.90)
z_sub = mesh_f["z_c"][z_mask]
y_sub = mesh_f["y_c"]
T_sub = T_comb_mid[z_mask, :].T
dT_log = np.log10(np.maximum(T_sub - T_INLET, 0.5))

Z_g, Y_g = np.meshgrid(z_sub, y_sub)
cf = ax.contourf(Z_g, Y_g, dT_log, levels=np.linspace(-0.3, 5.3, 45), cmap="inferno", extend="both")
cs_boil = ax.contour(Z_g, Y_g, T_sub, levels=[T_BOIL], colors=["#00e5ff"], linewidths=1.3, linestyles="-")
ax.add_patch(Rectangle((0.40, 0.0), 0.05, 0.05, fill=False, ec="white", lw=1.2, ls="--"))
ax.set_xlabel("Axial Coordinate $z$ [m]")
ax.set_ylabel("Vertical Coordinate $y$ [m]")
ax.set_title(r"(a) Mid-Plane ($x=0.10$ m) $\log_{10}(T - T_{\mathrm{in}})$", fontweight="bold")
ax.set_xlim(0.32, 0.88)
ax.set_ylim(0.0, 0.17)
cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.03, aspect=22)
cbar.set_label(r"$\Delta T = T - T_{\mathrm{in}}$", fontsize=8.5)
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.set_ticklabels(["1 K", "10 K", "$10^2$ K", "$10^3$ K", "$10^4$ K", "$10^5$ K"])
ax.plot([], [], color="#00e5ff", lw=1.3, label=r"$T_{\mathrm{sat}} = 373.15$ K")
ax.plot([], [], color="white", lw=1.2, ls="--", label="Solid cube")
ax.legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none", labelcolor="white", fontsize=7.4)

# Panel (b): Zoomed Solid Cube Internal Temperature T_s(z, y) in kK
ax = axes[1]
Zs, Ys = np.meshgrid(mesh_s["z_c"] * 1e3, mesh_s["y_c"] * 1e3)
cf_s = ax.contourf(Zs, Ys, T_s_mid.T / 1e3, levels=24, cmap="hot")
cs_s = ax.contour(Zs, Ys, T_s_mid.T / 1e3, levels=[40, 80, 120, 150, 170], colors="#00b4d8", linewidths=0.8, alpha=0.9)
ax.clabel(cs_s, fmt="%d", fontsize=6.8, inline=True, manual=[(406, 44), (412, 36), (414, 28), (416, 17)])
sp_z = [425.0, 425.0, 425.0]
sp_y = [25.0, 2.0, 48.0]
sp_lbl = [
    f"S0: {T_sprobes[-1, 0]/1e3:.1f} kK",
    f"S1: {T_sprobes[-1, 1]/1e3:.1f} kK",
    f"S2: {T_sprobes[-1, 2]/1e3:.1f} kK",
]
ax.scatter(sp_z, sp_y, c="#00e5ff", s=26, edgecolors="k", lw=0.7, zorder=5)
pill = dict(boxstyle="round,pad=0.15", fc="white", ec="#333333", lw=0.5, alpha=0.90)
ax.annotate(sp_lbl[1], (425, 2), xytext=(425, 8.2), ha="center", color="black", fontsize=7.2, fontweight="bold",
            bbox=pill, arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
ax.annotate(sp_lbl[0], (425, 25), xytext=(425, 28.5), ha="center", color="black", fontsize=7.2, fontweight="bold", bbox=pill)
ax.annotate(sp_lbl[2], (425, 48), xytext=(425, 41.5), ha="center", color="black", fontsize=7.2, fontweight="bold",
            bbox=pill, arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
ax.set_xlabel("Axial $z$ [mm]")
ax.set_ylabel("Vertical $y$ [mm]")
ax.set_title("(b) Solid Cube $T_{\\mathrm{solid}}$ [kK] ($x=0.10$ m)", fontweight="bold")
ax.set_xlim(400, 450)
ax.set_ylim(0, 50)
cbar_s = fig.colorbar(cf_s, ax=ax, pad=0.02, fraction=0.03, aspect=22)
cbar_s.set_label(r"$T_{\mathrm{solid}}$ [$10^3$ K]", fontsize=8.5)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_part4_temperature_fields_and_solid.{ext}"), bbox_inches="tight")
    shutil.copy2(os.path.join(FIG_DIR, f"fig_part4_temperature_fields_and_solid.{ext}"), os.path.join(RES_DIR, f"fig_part4_temperature_fields_and_solid.{ext}"))
plt.close(fig)


# -----------------------------------------------------------------------------
# 10. FIGURE 3: FACE-BY-FACE HEAT FLUX & DOWNSTREAM THERMAL WAKE (1x2)
# -----------------------------------------------------------------------------
print("[*] Generating Fig 3: Interface Heat Flux & Thermal Wake Development...")
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))

# Panel (a): Clean single-axis bar chart of interface heat flux q'' & share of Q_interface
ax = axes[0]
individual_faces = face_rows[:-1]
f_names = [r["face"].split(" ")[0] for r in individual_faces]
q_vals = [r["q_avg_MWm2"] for r in individual_faces]
pct_vals = [r["share_pct"] for r in individual_faces]

x_pos = np.arange(len(f_names))
bars = ax.bar(x_pos, q_vals, width=0.58, color="#d62728", alpha=0.86, edgecolor="k", lw=0.7, zorder=2)
q_mean_all = face_rows[-1]["q_avg_MWm2"]
ax.axhline(q_mean_all, color="#1f77b4", ls="--", lw=1.3, zorder=1, label=rf"Overall $\bar{{q}}'' = {q_mean_all:.1f}$ MW/m$^2$")
for b, pct, qv in zip(bars, pct_vals, q_vals):
    ax.text(
        b.get_x() + b.get_width() / 2, qv + 6.0,
        f"{qv:.1f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=7.2, color="black",
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88)
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(f_names)
ax.set_ylabel(r"Area-Averaged Heat Flux $\bar{q}''$ [MW/m$^2$]")
ax.set_ylim(0, 395)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
ax.set_title("(a) Interface Flux $\\bar{q}''$ & Share of $Q_{\\mathrm{int}}$", fontweight="bold")
ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.4)

# Panel (b): Vertical Fluid Wake Temperature Profiles (log10 dT vs y) across 5 downstream stations
ax = axes[1]
wake_styles = [
    ("z0475_0.5Hc",  r"$z = 0.475$ m ($0.5 H_c$)", "#d62728", "-",  1.6),
    ("z0550_2.0Hc",  r"$z = 0.550$ m ($2.0 H_c$)", "#ff7f0e", "--", 1.5),
    ("z0650_4.0Hc",  r"$z = 0.650$ m ($4.0 H_c$)", "#2ca02c", "-.", 1.5),
    ("z1000_11Hc",   r"$z = 1.000$ m ($11 H_c$)",  "#1f77b4", ":",  1.6),
    ("z1400_19Hc",   r"$z = 1.400$ m ($19 H_c$)",  "#9467bd", "-",  1.4),
]
for lbl, leg, col, ls, lw in wake_styles:
    dT_w = np.maximum(wake_cols[lbl] - T_INLET, 1e-3)
    ax.semilogx(dT_w, mesh_f["y_c"] * 1e3, color=col, ls=ls, lw=lw, label=leg)

ax.axvline(dT_mix_hist[-1], color="#1f77b4", ls="--", lw=1.1, label=rf"$\Delta T_{{\mathrm{{mix,out}}}} = {dT_mix_hist[-1]:.2f}$ K")
ax.axvline(T_BOIL - T_INLET, color="k", ls=":", lw=1.1, label=r"$T_{\mathrm{sat}} - T_{\mathrm{in}} = 75.0$ K")
ax.axhline(50.0, color="#666666", ls=":", lw=0.8)
ax.text(0.025, 52.0, "$y = H_c$", fontsize=7.4, color="#444444")
ax.set_xlabel(r"Fluid Temperature Excess $\Delta T = T - T_{\mathrm{in}}$ [K]")
ax.set_ylabel("Vertical Coordinate $y$ [mm]")
ax.set_title(r"(b) Downstream Thermal Wake ($x = 0.10$ m)", fontweight="bold")
ax.set_xlim(1e-2, 2.5e3)
ax.set_ylim(0, 140)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_part4_interface_flux_and_thermal_wake.{ext}"), bbox_inches="tight")
    shutil.copy2(os.path.join(FIG_DIR, f"fig_part4_interface_flux_and_thermal_wake.{ext}"), os.path.join(RES_DIR, f"fig_part4_interface_flux_and_thermal_wake.{ext}"))
plt.close(fig)


# -----------------------------------------------------------------------------
# 11. PRINT FINAL REPORT SUMMARY TO STDOUT
# -----------------------------------------------------------------------------
print("\n=== PART IV VERIFICATION & KEY RESULTS (ITERATION 15000) ===")
for k, v in summary_rows:
    print(f"  {k:36s}: {v}")
print("\n=== COUPLED INTERFACE BREAKDOWN BY CUBE FACE ===")
for r in face_rows:
    print(f"  {r['face']:19s}: Q={r['Q_MW']:.4f} MW ({r['share_pct']:5.2f}%), q_avg={r['q_avg_MWm2']:6.2f} MW/m2, T_avg={r['T_avg_K']:8.2f} K")
print("\n=== PROBES AT ITERATION 15000 ===")
print("  Fluid Probes T [K]:", np.round(T_fprobes[-1], 2))
print("  Fluid Probes p [Pa]:", np.round(p_fprobes[-1], 2))
print("  Fluid Probes |U| [m/s]:", np.round(np.linalg.norm(U_fprobes[-1], axis=1), 4))
print("  Fluid Probes Uz [m/s]:", np.round(U_fprobes[-1, :, 2], 4))
print("  Solid Probes T [K]:", np.round(T_sprobes[-1], 2))
print("[*] Done. All CSVs and figures generated cleanly.")
