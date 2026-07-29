# Contributing

Bug reports and focused pull requests are welcome.

## Local checks

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
$HOME/.julia/bin/mpiexecjl -n 4 julia --project=. test/mpi_smoke.jl
```

Changes to acquisition behavior should include a deterministic regression test
against an analytical Pareto front. Avoid loosening IGD thresholds without
documenting the numerical reason.

## Design scope

The package targets expensive continuous black-box objectives with hundreds to
low thousands of evaluations. New features should preserve:

- one-rank serial execution;
- asynchronous MPI scheduling;
- deterministic optimizer randomness under a fixed completion order;
- complete objective vectors for Pareto comparisons;
- isolation of failed evaluations.
