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
    seed=2026,
)

println("Pareto points: ", length(result.pareto_indices))
println("Decision range: ", extrema(vec(result.pareto_X)))
println("Compromise decision: ", result.compromise_x)
println("Compromise objectives: ", result.compromise_y)
