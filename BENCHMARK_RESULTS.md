# Benchmark Results

The benchmarks were run with Julia 1.11.1 on 2026-07-30 using the fixed seeds
in `benchmark/run_benchmarks.jl`.

Inverted generational distance (IGD) measures the average normalized distance
from the analytical reference front to the observed Pareto front. Lower is
better and zero is exact.

| Problem | Dimensions | Budget | Pareto points | ParEGO IGD | Random IGD |
|---|---:|---:|---:|---:|---:|
| Schaffer N.1 | 1 | 80 | 57 | 0.032060 | 0.050024 |
| Convex bi-objective | 2 | 100 | 19 | 0.044516 | 0.062199 |
| ZDT1 | 4 | 180 | 11 | 0.054313 | 0.408050 |

The analytical ZDT1 Pareto set requires `x2=x3=x4=0`. In the observed front:

- `f1` covered the complete `[0, 1]` range;
- the mean of `x2`, `x3`, and `x4` was `0.002798`;
- the largest auxiliary-variable deviation was `0.054313`;
- IGD was 86.7% lower than same-budget uniform random search.

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
