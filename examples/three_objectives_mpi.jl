using AsyncMultiObjectiveBayesOpt
using MPI

function three_objectives(parameters)
    x, y = parameters
    return [
        (x + 0.5)^2 + 0.2 * y^2,
        (x - 0.2)^2 + (y + 0.3)^2,
        0.3 * x^2 + (y - 0.6)^2,
    ]
end

result = async_mobo(
    three_objectives,
    [-2.0 2.0; -2.0 2.0];
    n_objectives=3,
    objective_senses=[:min, :min, :min],
    max_evals=100,
    n_initial=16,
    candidate_pool=8_000,
    seed=2026,
)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Pareto points: ", length(result.pareto_indices))
    println("Compromise parameters: ", result.compromise_x)
    println("Compromise objectives: ", result.compromise_y)
end

MPI.Barrier(MPI.COMM_WORLD)
MPI.Finalize()
