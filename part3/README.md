# Part III: Transient Wake Flow (`pimpleFoam`)

## Overview
Unsteady URANS simulation of vortex shedding and wake dynamics past the surface-mounted cube using `pimpleFoam` over $t \in [0, 2.00]\text{ s}$ ($40\,H_c/U_b$), initialized from the converged Part II steady solution (`0.orig/`).

## Numerics & Sampling
- **Mesh & Fluid**: Same `1,536,768`-cell obstructed mesh and `kOmegaSST` / water properties ($\nu = 8.93\times 10^{-7}\text{ m}^2/\text{s}$) as Part II.
- **Time integration (`system/controlDict`)**: Adaptive time step (`adjustTimeStep yes`, `maxCo 6.0`, settled $\Delta t \approx 7.72\times 10^{-5}\text{ s}$), second-order bounded temporal blending (`CrankNicolson 0.5`).
- **Monitors (`system/controlDict`)**: `CourantNo`/`fieldMinMax`, 8 `wakeProbes` (`P0`--`P7` logging `U`, `p`, `k` every step), and `fieldAverage` over $t \in [0.10, 2.00]\text{ s}$.

## Execution
```bash
./AllRun                # 12 cores locally (./AllRun_cluster for 2-node run)
reconstructPar -newTimes
python3 ../generate_part3_results.py
```
