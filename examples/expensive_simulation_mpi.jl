using AsyncMultiObjectiveBayesOpt
using MPI

function run_expensive_simulation(parameters, workdir)
    sleep(0.1)
    x, y = parameters
    return (signal_a=x + y, signal_b=x - y)
end

function objective(parameters)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    workdir = joinpath(@__DIR__, "scratch", "rank_$rank")
    mkpath(workdir)
    simulation = run_expensive_simulation(parameters, workdir)
    return [
        (simulation.signal_a - 0.5)^2,
        (simulation.signal_b + 0.25)^2,
        sum(abs2, parameters),
    ]
end

result = async_mobo(
    objective,
    [-2.0 2.0; -2.0 2.0];
    n_objectives=3,
    max_evals=120,
    n_initial=20,
    initial_concurrency=8,
    candidate_pool=8_000,
    gc_after_evaluation=true,
    seed=2031,
)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Pareto points: ", length(result.pareto_indices))
    println("Compromise parameters: ", result.compromise_x)
    println("Compromise objectives: ", result.compromise_y)
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
