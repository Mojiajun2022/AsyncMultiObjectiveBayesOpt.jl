using AsyncMultiObjectiveBayesOpt

# Minimize cost while maximizing quality.
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

println("Pareto objective vectors:")
display(result.pareto_Y)
