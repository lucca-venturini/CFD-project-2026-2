# Part II: Steady Flow Over a Surface-Mounted Cube (`simpleFoam`)

## Overview
Steady-state RANS simulation of turbulent water flow past a $0.05\text{ m} \times 0.05\text{ m} \times 0.05\text{ m}$ obstacle mounted on the bottom wall ($x\in[0.075,0.125]\text{ m}$, $y\in[0,0.050]\text{ m}$, $z\in[0.400,0.450]\text{ m}$, blockage ratio $\beta = 6.25\%$, $Re_{H_c} = 55{,}991$) using `simpleFoam`.

## Mesh & Topology
- **Base grid (`system/blockMeshDict`)**: Multi-graded structured hex grid of $104 \times 96 \times 156 = 1{,}557{,}504$ cells with boundary-layer clustering on both outer walls and cube faces ($25\,\mu\text{m}$ first cell at the cube).
- **Cube extraction (`system/topoSetDict`)**: Selects cube cells and removes them via `subsetMesh fluidCells -patch cube -overwrite`, yielding `1,536,768` fluid cells and no-slip patch `cube` (`y+avg = 0.929` on cube, `0.727` on walls).
- **Fluid & Turbulence**: Water ($\nu = 8.93\times 10^{-7}\text{ m}^2/\text{s}$, $\rho = 1000\text{ kg/m}^3$), `kOmegaSST` (`endTime = 5000`).

## Execution
```bash
./AllRun                # Runs on 8 cores (resumes from latest processor checkpoint if incomplete)
reconstructPar -latestTime
python3 ../generate_part2_results.py
```
