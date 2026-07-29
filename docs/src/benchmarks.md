# Benchmarks

The benchmark suite uses problems with analytical Pareto fronts:

- Schaffer N.1;
- a convex two-parameter, two-objective problem;
- four-dimensional ZDT1.

Run all benchmarks:

```bash
julia --project=. benchmark/run_benchmarks.jl
```

Run a subset:

```bash
julia --project=. benchmark/run_benchmarks.jl schaffer zdt1
```

## Observed accuracy

Normalized inverted generational distance (IGD) is lower when the observed
front more closely covers the analytical front.

![IGD comparison against random search](assets/igd_comparison.png)

| Problem | Budget | ParEGO IGD | Random IGD |
|---|---:|---:|---:|
| Schaffer N.1 | 80 | 0.032060 | 0.050024 |
| Convex bi-objective | 100 | 0.044516 | 0.062199 |
| ZDT1 (4D) | 180 | 0.054313 | 0.408050 |

![Convex benchmark front and Pareto decisions](assets/convex_front.png)

For ZDT1, the observed `f1` values covered `[0, 1]`, and the auxiliary decision
variables had mean value `0.002798` on the returned front. The optimizer
therefore recovered both the objective front and the decision-space manifold.

Generate every documentation figure from fresh optimization runs:

```bash
julia --project=benchmark -e '
  using Pkg
  Pkg.develop(path=pwd())
  include("benchmark/generate_plots.jl")'
```
