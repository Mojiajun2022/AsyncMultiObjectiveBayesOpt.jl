using AsyncMultiObjectiveBayesOpt
using CairoMakie
using Random

include(joinpath(@__DIR__, "benchmark_functions.jl"))
using .BenchmarkFunctions

CairoMakie.activate!(type="png")
assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)

function optimize(name, seed, pool)
    benchmark = BENCHMARKS[name]
    result = async_mobo(
        benchmark.objective,
        benchmark.bounds;
        n_objectives=2,
        max_evals=benchmark.budget,
        n_initial=benchmark.n_initial,
        candidate_pool=pool,
        seed=seed,
        verbose=false,
    )
    return benchmark, result
end

schaffer, schaffer_result = optimize(:schaffer, 2027, 6_000)
figure = Figure(size=(820, 540))
axis = Axis(
    figure[1, 1],
    title="Schaffer N.1 Pareto Front",
    xlabel="Objective f1",
    ylabel="Objective f2",
)
lines!(
    axis,
    schaffer.reference_front[1, :],
    schaffer.reference_front[2, :];
    color=:black,
    linewidth=2,
    label="Analytical front",
)
scatter!(
    axis,
    schaffer_result.pareto_Y[1, :],
    schaffer_result.pareto_Y[2, :];
    color=:dodgerblue,
    markersize=11,
    label="ParEGO observations",
)
axislegend(axis; position=:rt)
save(joinpath(assets, "schaffer_front.png"), figure; px_per_unit=2)

convex, convex_result = optimize(:convex, 2028, 6_000)
figure = Figure(size=(1080, 480))
front_axis = Axis(
    figure[1, 1],
    title="Convex Objective Front",
    xlabel="Objective f1",
    ylabel="Objective f2",
)
lines!(
    front_axis,
    convex.reference_front[1, :],
    convex.reference_front[2, :];
    color=:black,
    linewidth=2,
    label="Analytical front",
)
scatter!(
    front_axis,
    convex_result.pareto_Y[1, :],
    convex_result.pareto_Y[2, :];
    color=:darkorange,
    markersize=10,
    label="ParEGO observations",
)
axislegend(front_axis; position=:rt)
decision_axis = Axis(
    figure[1, 2],
    title="Recovered Decision Manifold",
    xlabel="Decision x1",
    ylabel="Decision x2",
)
lines!(
    decision_axis,
    convex.reference_decisions[1, :],
    convex.reference_decisions[2, :];
    color=:black,
    linewidth=2,
    label="Analytical set",
)
scatter!(
    decision_axis,
    convex_result.pareto_X[1, :],
    convex_result.pareto_X[2, :];
    color=:seagreen,
    markersize=10,
    label="Observed set",
)
axislegend(decision_axis; position=:rt)
save(joinpath(assets, "convex_front.png"), figure; px_per_unit=2)

zdt1, zdt1_result = optimize(:zdt1, 2029, 12_000)
figure = Figure(size=(1080, 480))
front_axis = Axis(
    figure[1, 1],
    title="ZDT1 Objective Front",
    xlabel="Objective f1",
    ylabel="Objective f2",
)
lines!(
    front_axis,
    zdt1.reference_front[1, :],
    zdt1.reference_front[2, :];
    color=:black,
    linewidth=2,
    label="Analytical front",
)
scatter!(
    front_axis,
    zdt1_result.pareto_Y[1, :],
    zdt1_result.pareto_Y[2, :];
    color=:dodgerblue,
    markersize=11,
    label="ParEGO observations",
)
axislegend(front_axis; position=:rt)
decision_axis = Axis(
    figure[1, 2],
    title="ZDT1 Auxiliary Decisions",
    xlabel="Pareto objective f1",
    ylabel="Decision value",
)
colors = (:darkorange, :seagreen, :mediumpurple)
for (row, color) in zip(2:4, colors)
    scatter!(
        decision_axis,
        zdt1_result.pareto_Y[1, :],
        zdt1_result.pareto_X[row, :];
        color=color,
        markersize=9,
        label="x$row",
    )
end
hlines!(decision_axis, [0.0]; color=:black, linestyle=:dash)
axislegend(decision_axis; position=:rt)
save(joinpath(assets, "zdt1_front.png"), figure; px_per_unit=2)

function three_objectives(x)
    a, b = x
    return [
        (a + 0.5)^2 + 0.2 * b^2,
        (a - 0.2)^2 + (b + 0.3)^2,
        0.3 * a^2 + (b - 0.6)^2,
    ]
end

three_result = async_mobo(
    three_objectives,
    [-2.0 2.0; -2.0 2.0];
    n_objectives=3,
    max_evals=100,
    n_initial=16,
    candidate_pool=8_000,
    seed=2030,
    verbose=false,
)
figure = Figure(size=(820, 620))
axis = Axis3(
    figure[1, 1],
    title="Three-Objective Pareto Archive",
    xlabel="Objective f1",
    ylabel="Objective f2",
    zlabel="Objective f3",
)
scatter!(
    axis,
    three_result.Y_history[1, :],
    three_result.Y_history[2, :],
    three_result.Y_history[3, :];
    color=(:gray, 0.35),
    markersize=7,
    label="All evaluations",
)
scatter!(
    axis,
    three_result.pareto_Y[1, :],
    three_result.pareto_Y[2, :],
    three_result.pareto_Y[3, :];
    color=:crimson,
    markersize=13,
    label="Pareto archive",
)
axislegend(axis; position=:rt)
save(
    joinpath(assets, "three_objective_front.png"),
    figure;
    px_per_unit=2,
)

names = ["Schaffer", "Convex", "ZDT1"]
parego_igd = [0.032060, 0.044516, 0.054313]
random_igd = [0.050024, 0.062199, 0.408050]
figure = Figure(size=(860, 500))
axis = Axis(
    figure[1, 1],
    title="Pareto-Front Accuracy at Equal Evaluation Budget",
    xlabel="Benchmark",
    ylabel="Normalized IGD (lower is better)",
    xticks=(1:3, names),
)
barplot!(
    axis,
    (1:3) .- 0.18,
    parego_igd;
    width=0.34,
    color=:dodgerblue,
    label="ParEGO",
)
barplot!(
    axis,
    (1:3) .+ 0.18,
    random_igd;
    width=0.34,
    color=:darkorange,
    label="Uniform random",
)
axislegend(axis; position=:lt)
save(joinpath(assets, "igd_comparison.png"), figure; px_per_unit=2)

println("Generated benchmark plots in $assets")
