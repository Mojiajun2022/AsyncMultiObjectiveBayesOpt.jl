# MPI and clusters

## Execution model

All MPI ranks execute the same Julia script:

- the root rank maintains the surrogate and dispatches points;
- every other rank evaluates one point at a time;
- workers return objective vectors in completion order;
- the root immediately dispatches a replacement task;
- only the root returns an optimization result.

With `-n 41`, the run uses one scheduler and 40 evaluation workers.

## Local launch

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
mpiexecjl -n 9 julia --project=. examples/expensive_simulation_mpi.jl
```

## Slurm

After configuring MPI.jl to use the cluster MPI implementation:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

srun --mpi=pmix \
  julia --project=/path/to/AsyncMultiObjectiveBayesOpt.jl run_fit.jl
```

Every rank must enter `async_mobo`, `MPI.Barrier`, and `MPI.Finalize`.

## Initial concurrency

Large jobs should avoid assigning the entire initial random design at once:

```julia
result = async_mobo(
    objective,
    bounds;
    n_objectives=3,
    max_evals=400,
    n_initial=40,
    initial_concurrency=40,
)
```

Only 40 workers participate until the initial design is complete. Remaining
workers are then released into the Bayesian phase.

## Parallelism and sample efficiency

More workers reduce wall time but increase the number of pending evaluations.
Pending points do not yet provide real feedback, so very high concurrency can
reduce sample efficiency. For an expensive 5-15 dimensional problem, 32-64
workers are a reasonable first run. Increase concurrency when wall time matters
more than the number of evaluations.

Avoid nested oversubscription. If one simulation already uses multiple threads,
allocate fewer MPI ranks per node.
