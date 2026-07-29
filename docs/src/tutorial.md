# Tutorial

This tutorial starts with analytical two-objective problems, moves to the
four-dimensional ZDT1 benchmark, and finishes with mixed objective directions,
three objectives, and asynchronous MPI simulations.

All programs are available in `examples/` or `benchmark/`.

## 1. Schaffer N.1

Schaffer N.1 is a one-parameter problem:

```math
f_1(x)=x^2,\qquad f_2(x)=(x-2)^2.
```

Every `x` in `[0, 2]` is Pareto-optimal.

```julia
using AsyncMultiObjectiveBayesOpt

objective(x) = [x[1]^2, (x[1] - 2.0)^2]
bounds = reshape([-5.0, 5.0], 1, 2)

result = async_mobo(
    objective,
    bounds;
    n_objectives=2,
    max_evals=80,
    n_initial=12,
    candidate_pool=5_000,
    acquisition=:auto,
    ehvi_samples=64,
    seed=2026,
)

println("Pareto points: ", length(result.pareto_indices))
println("Decision range: ", extrema(vec(result.pareto_X)))
println("Compromise point: ", result.compromise_x)
```

`acquisition=:auto` selects EHVI because this problem has two objectives.

![Schaffer analytical and observed Pareto fronts](assets/schaffer_front.png)

Run the packaged example:

```bash
julia --project=. examples/schaffer_serial.jl
```

## 2. Convex two-dimensional problem

The second problem adds a decision direction that should converge to zero:

```math
f_1(x)=x_1^2+x_2^2,\qquad
f_2(x)=(x_1-1)^2+x_2^2.
```

Its Pareto set is `x1 in [0, 1]` and `x2=0`.

```julia
using AsyncMultiObjectiveBayesOpt

function convex_objective(x)
    penalty = x[2]^2
    return [
        x[1]^2 + penalty,
        (x[1] - 1.0)^2 + penalty,
    ]
end

result = async_mobo(
    convex_objective,
    [-0.5 1.5; -1.0 1.0];
    n_objectives=2,
    max_evals=100,
    n_initial=16,
    candidate_pool=6_000,
    acquisition=:ehvi,
    seed=2028,
)
```

The left panel compares objective fronts. The right panel checks whether the
optimizer recovered the decision-space condition `x2=0`.

![Convex objective and decision fronts](assets/convex_front.png)

## 3. ZDT1

ZDT1 is a more demanding four-dimensional test:

```math
f_1=x_1,\quad
g=1+9\frac{x_2+x_3+x_4}{3},\quad
f_2=g\left(1-\sqrt{f_1/g}\right).
```

The analytical Pareto set requires `x2=x3=x4=0`, and its objective front is
`f2=1-sqrt(f1)`.

```julia
using AsyncMultiObjectiveBayesOpt

function zdt1(x)
    f1 = x[1]
    g = 1.0 + 9.0 * sum(@view x[2:end]) / (length(x) - 1)
    return [f1, g * (1.0 - sqrt(f1 / g))]
end

bounds = [
    0.0 1.0
    0.0 1.0
    0.0 1.0
    0.0 1.0
]

result = async_mobo(
    zdt1,
    bounds;
    n_objectives=2,
    max_evals=180,
    n_initial=24,
    candidate_pool=12_000,
    acquisition=:ehvi,
    ehvi_samples=64,
    optimize_length_scale=true,
    seed=2029,
)
```

The ARD length-scale refinement is important here: `x1` selects a position
along the front, while `x2:x4` control distance away from it.

![ZDT1 objective front and auxiliary decisions](assets/zdt1_front.png)

For a higher-accuracy decision manifold, increase real evaluations before
increasing Monte Carlo samples:

```julia
high_accuracy = async_mobo(
    zdt1,
    bounds;
    n_objectives=2,
    max_evals=240,
    n_initial=24,
    candidate_pool=16_000,
    acquisition=:ehvi,
    ehvi_samples=64,
    seed=2029,
)
```

The fixed-seed reference run produced:

| Configuration | IGD | Pareto points | Auxiliary decisions |
|---|---:|---:|---|
| 180 evaluations | 0.023823 | 17 | Close to the analytical manifold |
| 240 evaluations | 0.022147 | 18 | `x2=x3=x4=0` on the returned front |

Increasing `ehvi_samples` from 64 to 128 did not materially improve this
benchmark, but approximately doubled acquisition work. The limiting factor was
the number of informative objective evaluations, not Monte Carlo variance.

### Why the original ZDT1 fit was weak

The original ParEGO-only implementation compressed both objectives into one
random scalar target. That is useful for many objectives, but on ZDT1 it spent
too many proposals revisiting endpoint weights and gave relatively sparse
coverage between the endpoints. A single isotropic GP length scale also had to
represent two different roles: `x1` moves along the Pareto front, whereas
`x2:x4` move away from it. Fully random local perturbations made it difficult
to preserve a good `x1` value while independently driving an auxiliary
coordinate toward zero.

The current optimizer addresses these issues with:

- EHVI for two objectives, which directly rewards objective-space Pareto
  coverage;
- one normalized GP per objective;
- marginal-likelihood refinement of global and per-coordinate ARD length
  scales;
- coordinate-preserving candidates around observed Pareto points;
- pending-point fantasies for asynchronous workers;
- less frequent extreme ParEGO weights when ParEGO is selected explicitly.

For similar smooth two-objective problems, tune in this order:

1. Keep `acquisition=:auto` or select `:ehvi`.
2. Increase `max_evals`; this had the largest effect on ZDT1.
3. Increase `candidate_pool` when objective evaluations are expensive enough
   to justify more scheduler work.
4. Keep `optimize_length_scale=true`, especially when parameter dimensions
   have different effects.
5. Increase `ehvi_samples` only if repeated fixed-budget runs show acquisition
   noise; 64 was sufficient here.
6. Increase `noise` above `1e-8` only for genuinely noisy objectives or
   numerical instability.

Use several seeds and report the median IGD for comparative studies. The
fixed-seed values in this tutorial are reproducibility references, not a claim
that every run or black-box objective reaches the same accuracy.

Run the packaged example:

```bash
julia --project=. examples/zdt1_serial.jl
```

## 4. Mixed minimization and maximization

Objective values retain their original signs and units. Specify one direction
per objective:

```julia
using AsyncMultiObjectiveBayesOpt

function cost_and_quality(x)
    cost = x[1]^2 + 0.2 * x[2]^2
    quality = 1.0 - (x[1] - 1.0)^2 - (x[2] - 0.5)^2
    return [cost, quality]
end

result = async_mobo(
    cost_and_quality,
    [-1.0 2.0; -1.0 2.0];
    n_objectives=2,
    objective_senses=[:min, :max],
    max_evals=100,
    n_initial=16,
    candidate_pool=6_000,
    seed=2030,
)
```

Run:

```bash
julia --project=. examples/mixed_senses_serial.jl
```

## 5. Three objectives

Three or more objectives automatically use ParEGO:

```julia
using AsyncMultiObjectiveBayesOpt
using MPI

function three_objectives(x)
    a, b = x
    return [
        (a + 0.5)^2 + 0.2 * b^2,
        (a - 0.2)^2 + (b + 0.3)^2,
        0.3 * a^2 + (b - 0.6)^2,
    ]
end

result = async_mobo(
    three_objectives,
    [-2.0 2.0; -2.0 2.0];
    n_objectives=3,
    max_evals=100,
    n_initial=16,
    candidate_pool=8_000,
    acquisition=:auto,
    seed=2030,
)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Pareto points: ", length(result.pareto_indices))
    println("Compromise parameters: ", result.compromise_x)
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
```

![Three-objective Pareto archive](assets/three_objective_front.png)

Run with four workers:

```bash
$HOME/.julia/bin/mpiexecjl -n 5 \
  julia --project=. examples/three_objectives_mpi.jl
```

## 6. Expensive MPI simulation

Run the expensive simulation once and derive every objective from the same
output. Give each rank a separate working directory:

```julia
using AsyncMultiObjectiveBayesOpt
using MPI

function objective(parameters)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    workdir = joinpath("scratch", "rank_$rank")
    mkpath(workdir)

    simulation = run_simulation(parameters, workdir)
    return [
        fit_error_dataset_a(simulation),
        fit_error_dataset_b(simulation),
        fit_error_dataset_c(simulation),
    ]
end

result = async_mobo(
    objective,
    bounds;
    n_objectives=3,
    max_evals=400,
    n_initial=40,
    initial_concurrency=40,
    candidate_pool=20_000,
    gc_after_evaluation=true,
    seed=2031,
)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    save_result(result)
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
```

Launch without nested thread oversubscription:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
$HOME/.julia/bin/mpiexecjl -n 41 julia --project=. run_fit.jl
```

If the simulation throws or any objective is non-finite, the evaluation is
recorded as a failure and excluded from surrogate fitting.

## 7. Inspect and save results

Columns correspond across parameter and objective matrices:

```julia
result.pareto_X
result.pareto_Y
result.pareto_indices
result.compromise_x
result.compromise_y
result.X_history
result.Y_history
```

There is no unique best Pareto point. `compromise_x` is merely the observed
front point nearest the normalized ideal.

Save dependency-free CSV files on rank zero:

```julia
using DelimitedFiles
using MPI

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    writedlm("pareto_parameters.csv", permutedims(result.pareto_X), ',')
    writedlm("pareto_objectives.csv", permutedims(result.pareto_Y), ',')
    writedlm("all_parameters.csv", permutedims(result.X_history), ',')
    writedlm("all_objectives.csv", permutedims(result.Y_history), ',')
end
```

## 8. Reproduce tests and figures

Run all analytical-front regression tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Print benchmark IGD results:

```bash
julia --project=. benchmark/run_benchmarks.jl
```

Regenerate every documentation plot:

```bash
julia --project=benchmark -e '
  using Pkg
  Pkg.develop(path=pwd())
  include("benchmark/generate_plots.jl")'
```
