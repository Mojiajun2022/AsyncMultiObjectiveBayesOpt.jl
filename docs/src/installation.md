# Installation

## From GitHub

Until the package is registered in Julia's General registry, install it
directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Mojiajun2022/AsyncMultiObjectiveBayesOpt.jl")
```

## Development checkout

```bash
git clone https://github.com/Mojiajun2022/AsyncMultiObjectiveBayesOpt.jl
cd AsyncMultiObjectiveBayesOpt.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## MPI launcher

MPI.jl can install a launcher matching its configured MPI implementation:

```bash
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
```

The launcher is normally written to `$HOME/.julia/bin/mpiexecjl`. Add that
directory to `PATH`, or invoke it by its absolute path.

Verify the multi-process scheduler:

```bash
$HOME/.julia/bin/mpiexecjl -n 4 \
  julia --project=. test/mpi_smoke.jl
```
