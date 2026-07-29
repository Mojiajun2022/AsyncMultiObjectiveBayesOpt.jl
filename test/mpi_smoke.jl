using AsyncMultiObjectiveBayesOpt
using MPI

function objective(x)
    sleep(0.005 * MPI.Comm_rank(MPI.COMM_WORLD))
    return [x[1]^2, (x[1] - 1.0)^2]
end

budget = isempty(ARGS) ? 24 : parse(Int, ARGS[1])
result = async_mobo(
    objective, [-1.0 2.0];
    n_objectives=2, max_evals=budget, n_initial=min(8, budget),
    initial_concurrency=1, candidate_pool=800, seed=4, verbose=false,
)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @assert result.n_evaluations == budget
    @assert result.worker_count == MPI.Comm_size(MPI.COMM_WORLD) - 1
    @assert result.n_failures == 0
    @assert !isempty(result.pareto_indices)
    println("MPI_MOBO_SMOKE_OK pareto=$(length(result.pareto_indices))")
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
