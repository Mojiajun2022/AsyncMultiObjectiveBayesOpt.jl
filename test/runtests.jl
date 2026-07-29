using AsyncMultiObjectiveBayesOpt
using Test

include(joinpath(@__DIR__, "..", "benchmark", "benchmark_functions.jl"))
using .BenchmarkFunctions

@testset "Pareto utilities" begin
    Y = [0.0 1.0 0.5 2.0; 2.0 0.0 0.5 2.0]
    @test nondominated_indices(Y) == [1, 2, 3]
    @test dominates([0.0, 1.0], [1.0, 2.0])
    @test !dominates([0.0, 2.0], [1.0, 1.0])
    @test nondominated_indices(Y; senses=[:max, :max]) == [4]
    @test isapprox(
        hypervolume_2d([1.0 2.0; 2.0 1.0], [3.0, 3.0]),
        3.0,
    )
    @test inverted_generational_distance(
        [0.0 1.0; 1.0 0.0],
        [0.0 1.0; 1.0 0.0],
    ) == 0.0
end

@testset "input validation and failures" begin
    @test_throws ArgumentError async_mobo(
        x -> [x[1], x[1]^2], [-1.0 1.0];
        n_objectives=1, max_evals=2, verbose=false,
    )
    result = async_mobo(
        x -> x[1] > 0.5 ? [NaN, NaN] : [x[1]^2, (x[1] - 1)^2],
        [-1.0 1.0];
        n_objectives=2, max_evals=12, n_initial=6,
        candidate_pool=300, seed=9, verbose=false,
    )
    @test result.n_evaluations == 12
    @test result.n_failures >= 0
    @test all(isfinite, result.pareto_Y)
end

@testset "Schaffer Pareto-front accuracy" begin
    benchmark = BENCHMARKS[:schaffer]
    result = async_mobo(
        benchmark.objective, benchmark.bounds;
        n_objectives=2, max_evals=benchmark.budget,
        n_initial=benchmark.n_initial, candidate_pool=5_000,
        seed=2026, verbose=false,
    )
    igd = inverted_generational_distance(
        result.pareto_Y, benchmark.reference_front,
    )
    @test igd < 0.055
    @test minimum(result.pareto_X) < 0.12
    @test maximum(result.pareto_X) > 1.88
end

@testset "Convex two-dimensional Pareto-front accuracy" begin
    benchmark = BENCHMARKS[:convex]
    result = async_mobo(
        benchmark.objective, benchmark.bounds;
        n_objectives=2, max_evals=benchmark.budget,
        n_initial=benchmark.n_initial, candidate_pool=6_000,
        seed=2027, verbose=false,
    )
    igd = inverted_generational_distance(
        result.pareto_Y, benchmark.reference_front,
    )
    @test igd < 0.065
    manifold_error = sum(abs, @view result.pareto_X[2, :]) /
                     size(result.pareto_X, 2)
    @test manifold_error < 0.08
end

@testset "ZDT1 four-dimensional Pareto-front accuracy" begin
    benchmark = BENCHMARKS[:zdt1]
    result = async_mobo(
        benchmark.objective, benchmark.bounds;
        n_objectives=2, max_evals=benchmark.budget,
        n_initial=benchmark.n_initial, candidate_pool=12_000,
        seed=2029, verbose=false,
    )
    igd = inverted_generational_distance(
        result.pareto_Y, benchmark.reference_front,
    )
    @test igd < 0.075
    @test extrema(@view result.pareto_Y[1, :]) == (0.0, 1.0)
    @test sum(@view result.pareto_X[2:end, :]) /
          length(@view result.pareto_X[2:end, :]) < 0.01
end
