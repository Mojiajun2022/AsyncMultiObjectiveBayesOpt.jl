using AsyncMultiObjectiveBayesOpt
using MPI
using Printf

include(joinpath(@__DIR__, "benchmark_functions.jl"))
using .BenchmarkFunctions

selected = isempty(ARGS) ? [:schaffer, :convex, :zdt1] :
           Symbol.(lowercase.(ARGS))
unknown = setdiff(selected, collect(keys(BENCHMARKS)))
isempty(unknown) ||
    error("unknown benchmarks: $(join(string.(unknown), ", "))")

for (offset, name) in enumerate(selected)
    benchmark = BENCHMARKS[name]
    result = async_mobo(
        benchmark.objective,
        benchmark.bounds;
        n_objectives=2,
        max_evals=benchmark.budget,
        n_initial=benchmark.n_initial,
        candidate_pool=name == :zdt1 ? 12_000 : 6_000,
        seed=2026 + offset,
        verbose=false,
    )
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        igd = inverted_generational_distance(
            result.pareto_Y, benchmark.reference_front,
        )
        @printf(
            "%-20s evaluations=%3d pareto=%3d IGD=%.6f elapsed=%.2fs\n",
            benchmark.name, result.n_evaluations,
            length(result.pareto_indices), igd, result.elapsed_seconds,
        )
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

MPI.Finalize()
