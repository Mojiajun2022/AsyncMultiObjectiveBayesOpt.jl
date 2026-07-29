# Benchmark Results

The benchmarks were run with Julia 1.11.1 on 2026-07-30 using the fixed seeds
in `benchmark/run_benchmarks.jl`.

Inverted generational distance (IGD) measures the average normalized distance
from the analytical reference front to the observed Pareto front. Lower is
better and zero is exact.

| Problem | Dimensions | Budget | Pareto points | MOBO (`:auto`) IGD | Random IGD |
|---|---:|---:|---:|---:|---:|
| Schaffer N.1 | 1 | 80 | 64 | 0.008708 | 0.050024 |
| Convex bi-objective | 2 | 100 | 83 | 0.005802 | 0.062199 |
| ZDT1 | 4 | 180 | 17 | 0.023823 | 0.408050 |

The analytical ZDT1 Pareto set requires `x2=x3=x4=0`. In the observed front:

- the 180-evaluation run reduced IGD by about 94% relative to random search;
- EHVI plus ARD reduced IGD by about 56% relative to the original ParEGO run;
- a 240-evaluation run reached IGD `0.022147`;
- all auxiliary variables on that high-accuracy front were exactly zero.

These results show that the optimizer recovered both the analytical objective
front and its decision-space manifold. They do not imply that every black-box
problem will reach the same accuracy with the same budget.

Reproduce the optimization benchmarks:

```bash
julia --project=. benchmark/run_benchmarks.jl
```

Verify asynchronous MPI scheduling:

```bash
$HOME/.julia/bin/mpiexecjl -n 4 julia --project=. test/mpi_smoke.jl
```
