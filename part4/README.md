# Part IV: Steady Conjugate Heat Transfer (`chtMultiRegionSimpleFoam`)

## Overview
Models steady-state coupled conjugate heat transfer between the water-fed duct (`fluid` region) and the heat-generating metallic cube (`solid` region) using `chtMultiRegionSimpleFoam` (OpenFOAM v2506).

## Regions & Mesh
- Base structured hex grid: `104 x 96 x 156` (`1,557,504` total cells).
- Split via `topoSet` + `splitMeshRegions -cellZones -overwrite`:
  - `solid`: `20,736` cells ($0.05\times 0.05\times 0.05\,\text{m}$, $V_s = 1.25\times 10^{-4}\,\text{m}^3$).
  - `fluid`: `1,536,768` cells ($V_f = 0.059875\,\text{m}^3$).
- Coupled interface: `fluid_to_solid` / `solid_to_fluid` ($4,032$ faces, $A = 0.0125\,\text{m}^2$).

## Thermophysical Models & Prescribed Heat Source
- `fluid` (Water, PDF Section 6 basis): `heRhoThermo` (`rhoConst`), $\rho = 1000\,\text{kg/m}^3$, $c_p = 4184\,\text{J/(kg}\cdot\text{K)}$, $\mu = 8.90\times 10^{-4}\,\text{Pa}\cdot\text{s}$, $\Pr = 6.135$ ($\kappa_f = 0.607\,\text{W/(m}\cdot\text{K)}$), $T_{\text{in}} = 298.15\,\text{K}$ ($25^\circ\text{C}$), $p_{\text{out}} = 101,325\,\text{Pa}$.
- `solid` (Dense Metallic Block): `heSolidThermo` (`constIso`, `rhoConst`), $\rho_s = 8000\,\text{kg/m}^3$, $c_{p,s} = 500\,\text{J/(kg}\cdot\text{K)}$, $\kappa_s = 50\,\text{W/(m}\cdot\text{K)}$.
- Volumetric Heat Source (`system/solid/fvOptions`): `scalarSemiImplicitSource` on `h` with `volumeMode specific`, $\dot{q} = 2.68\times 10^{10}\,\text{W/m}^3$ ($Q = \dot{q} V_s = 3.35\times 10^6\,\text{W}$, targeting $\Delta T_{\text{mix}} = 20\,\text{K}$).
- Adiabatic walls (`walls` in both `fluid` and `solid` use `zeroGradient`: $-\mathbf{n}\cdot k\nabla T = 0$).

## Execution
- Single workstation (8 cores): `./AllRun`
- 2-node Thunderbolt cluster (8+4 P-cores): `./AllRun_cluster`
- Reconstruct after run: `reconstructPar -allRegions -latestTime`
