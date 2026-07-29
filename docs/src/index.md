# AsyncMultiObjectiveBayesOpt.jl

`AsyncMultiObjectiveBayesOpt.jl` is an asynchronous MPI Bayesian optimizer for
expensive multi-objective black-box functions.

It is intended for simulations, model calibration, and scientific fitting
tasks where:

- one evaluation takes seconds to hours;
- evaluations have variable runtimes;
- gradients are unavailable;
- two to several objectives must be balanced;
- evaluations can run independently across CPU cores or cluster nodes.

The optimizer combines ParEGO, exact Gaussian-process surrogates, expected
improvement, Pareto-aware candidate generation, and asynchronous MPI
master-worker scheduling.

## Key properties

- Every completed worker is immediately assigned a new point.
- Pending points are included through Kriging-believer fantasies.
- Objectives can independently use minimization or maximization.
- Returned fronts always use original objective vectors, not scalarized values.
- A one-rank execution provides a serial fallback for development and testing.
- Failed or non-finite evaluations are isolated from surrogate fitting.

The [tutorial](tutorial.md) develops a complete optimization, while
[benchmarks](benchmarks.md) compare discovered fronts with analytical fronts.
