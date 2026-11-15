# Part V: Transient Conjugate Heat Transfer (`chtMultiRegionFoam`)

## Objective
Predict the physical time $t_{\text{boil}}$ required for any water in the computational domain to first reach the local boiling-temperature threshold $T_{\text{sat}}(p)$ when the constant volumetric heat source $\dot{q} = 2.68 \times 10^{10}\text{ W/m}^3$ ($Q = 3.350\text{ MW}$) operates continuously from $t = 0$.

## Initialization & Setup
- **Flow field**: Initialized from the converged hydrodynamic/turbulence fields (`U`, `p`, `p_rgh`, `k`, `omega`, `nut`, `alphat`, `phi`, `rho`) in `part4/15000/fluid/`.
- **Initial temperatures**: Uniform $T_0 = 298.15\text{ K}$ ($25.0\,^\circ\text{C}$) in both `fluid` and `solid` (`0.orig/fluid/T`, `0.orig/solid/T`).
- **Heat source**: Continuous `scalarSemiImplicitSource` (`h (2.68e10 0)`) in `system/solid/fvOptions`.
- **Time integration**: First-order implicit `Euler` (`ddtSchemes`), `PIMPLE` (`nOuterCorrectors 2`, `nCorrectors 2`), initial/maximum $\Delta t = 5 \times 10^{-5}\text{ s}$ with `adjustTimeStep yes` (`maxCo 5.0`), integrated to `endTime = 0.04 s` (writing 3D snapshots every $0.002\text{ s}$).

## Running
```bash
cd part5
./AllRun
```
- `AllRun` automatically resumes from the latest checkpoint in `processor0` if an incomplete run is present (`0 < t < endTime`), otherwise cleans transient data, verifies/decomposes all regions, starts `chtMultiRegionFoam` in parallel on 8 cores, launches `monitor_postprocessing.py`, and runs `reconstructPar -allRegions -newTimes` upon completion.
