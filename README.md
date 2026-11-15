# Water-Cooled Duct with a Heated Cube — OpenFOAM CFD Project (Parts I–V)

Flow, turbulence, unsteady wake dynamics, and conjugate heat transfer (CHT) around a $0.05\text{ m}$ heat-generating cube ($\dot{q} = 2.68\times 10^{10}\text{ W/m}^3$, $Q = 3.350\text{ MW}$) mounted on the bottom wall of a $0.20\text{ m} \times 0.20\text{ m} \times 1.50\text{ m}$ water-cooled square duct ($U_b = 1.00\text{ m/s}$, $Re_{D_h} = 223{,}964$), built and solved in OpenFOAM v2506 (`kOmegaSST`).

## Repository Layout
- `part1_emptyDuct` -> `part1/`: **Part I** — Steady empty square duct flow (`simpleFoam`, $1{,}215{,}000$ cells)
- `part2_cubeSteady` -> `part2/`: **Part II** — Steady flow with `topoSet`/`subsetMesh` cube (`simpleFoam`, $1{,}536{,}768$ cells)
- `part3_cubeTransient` -> `part3/`: **Part III** — Transient wake & Strouhal shedding (`pimpleFoam`, $t \in [0, 2.00]\text{ s}$)
- `part4_CHTSteady` -> `part4/`: **Part IV** — Steady multi-region CHT (`chtMultiRegionSimpleFoam`, $15{,}000$ iter)
- `part5_CHTTransient` -> `part5/`: **Part V** — Transient CHT & boiling-onset prediction (`chtMultiRegionFoam`, $t \in [0, 20.00]\text{ ms}$)
- `scripts/`: Master batch drivers (`runAll`, `cleanAll`, `postProcessAll`)
- `generate_part{1..5}_results.py`: Post-processing scripts producing `data/` and `figures/`
- `data/`: Exported CSV/TXT quantitative metrics for Parts I–V
- `figures/`: Publication-ready vector PDF and PNG figures used in `report.pdf`
- `report.tex` / `report.pdf`: Concise final project report

## Calculation Basis
All Parts I–V follow the PDF Section 6 calculation basis: water at $25\,^\circ\text{C}$ ($\rho = 1000\text{ kg/m}^3$, $\mu = 8.90\times 10^{-4}\text{ Pa}\cdot\text{s}$, $\nu = 8.93\times 10^{-7}\text{ m}^2/\text{s}$, $c_p = 4184\text{ J/(kg}\cdot\text{K)}$, $\mathrm{Pr} = 6.135$, $T_{\text{in}} = 298.15\text{ K}$, $p_{\text{out}} = 101{,}325\text{ Pa}$), solid cube ($\rho_s = 8000\text{ kg/m}^3$, $c_{p,s} = 500\text{ J/(kg}\cdot\text{K)}$, $\kappa_s = 50\text{ W/(m}\cdot\text{K)}$), and volumetric heat generation $\dot{q} = 2.68\times 10^{10}\text{ W/m}^3$.

## Quick Start
```bash
# Run all 5 OpenFOAM cases sequentially (or pass part numbers: ./scripts/runAll 1 2)
./scripts/runAll

# Regenerate CSV tables in data/ and figures in figures/
./scripts/postProcessAll

# Remove runtime solver outputs (processor*, reconstructed times, polyMesh, logs)
./scripts/cleanAll
```

## GitHub Notes
`.gitignore` excludes OpenFOAM runtime outputs (`processor*/`, reconstructed time directories, generated `polyMesh/`, logs, `postProcessing/`) and the uncropped 24 MB ParaView screenshot (`figures/fig7_boundary_layer_paraview.png`). Every tracked file is $\le 35\text{ MB}$ (below GitHub's $50\text{ MB}$ warning and $100\text{ MB}$ hard limit); `part3/0.orig/` (`119 MB`) and `part5/0.orig/` (`143 MB`) are kept tracked so Parts III and V can run standalone, or can be excluded by uncommenting two lines in `.gitignore`.
