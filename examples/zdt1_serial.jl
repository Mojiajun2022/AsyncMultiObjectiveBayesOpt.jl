using AsyncMultiObjectiveBayesOpt

function zdt1(x)
    f1 = x[1]
    g = 1.0 + 9.0 * sum(@view x[2:end]) / (length(x) - 1)
    return [f1, g * (1.0 - sqrt(f1 / g))]
end

bounds = [0.0 1.0; 0.0 1.0; 0.0 1.0; 0.0 1.0]
result = async_mobo(
    zdt1,
    bounds;
    n_objectives=2,
    max_evals=180,
    n_initial=24,
    candidate_pool=12_000,
    seed=2029,
)

println("Pareto points: ", length(result.pareto_indices))
println("f1 range: ", extrema(@view result.pareto_Y[1, :]))
println("Compromise decision: ", result.compromise_x)
