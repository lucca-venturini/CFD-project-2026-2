# Part I: Steady Empty Duct Flow (`simpleFoam`)

## Overview
Baseline steady-state incompressible RANS simulation of hydrodynamically developing turbulent water flow in a $0.20\text{ m} \times 0.20\text{ m} \times 1.50\text{ m}$ square duct ($D_h = 0.20\text{ m}$) using `simpleFoam` (OpenFOAM v2506).

## Mesh & Physics
- **Mesh (`system/blockMeshDict`)**: Graded structured hexahedral grid of $90 \times 90 \times 150 = 1{,}215{,}000$ cells with near-wall clustering ($\Delta y_1 = 34.87\,\mu\text{m}$, $y^+_{\text{avg}} = 0.6902$).
- **Fluid (`constant/transportProperties`)**: Water at $25\,^\circ\text{C}$, Newtonian $\nu = 8.93\times 10^{-7}\text{ m}^2/\text{s}$ ($\rho = 1000\text{ kg/m}^3$, $U_b = 1.00\text{ m/s}$, $Re_{D_h} = 223{,}964$).
- **Turbulence (`constant/turbulenceProperties`)**: Menter $k$-$\omega$ SST (`kOmegaSST`), low-$Re$ wall-resolved boundary conditions (`nutUSpaldingWallFunction`, `kqRWallFunction`, `omegaWallFunction`).

## Execution
```bash
./AllRun                # Single workstation (8 cores, checkpoints supported)
reconstructPar -latestTime
python3 ../generate_part1_results.py
```
