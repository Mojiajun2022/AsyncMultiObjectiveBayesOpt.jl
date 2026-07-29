module BenchmarkFunctions

export MOBenchmark, BENCHMARKS

struct MOBenchmark
    name::String
    objective::Function
    bounds::Matrix{Float64}
    reference_front::Matrix{Float64}
    reference_decisions::Matrix{Float64}
    budget::Int
    n_initial::Int
end

function schaffer_front(n=501)
    x = collect(range(0.0, 2.0; length=n))
    return reshape(x, 1, :), vcat(
        reshape(x .^ 2, 1, :),
        reshape((x .- 2.0) .^ 2, 1, :),
    )
end

function convex_front(n=501)
    x = collect(range(0.0, 1.0; length=n))
    decisions = vcat(reshape(x, 1, :), zeros(1, n))
    objectives = vcat(
        reshape(x .^ 2, 1, :),
        reshape((x .- 1.0) .^ 2, 1, :),
    )
    return decisions, objectives
end

function zdt1_front(d=4, n=501)
    x = collect(range(0.0, 1.0; length=n))
    decisions = vcat(reshape(x, 1, :), zeros(d - 1, n))
    objectives = vcat(
        reshape(x, 1, :),
        reshape(1 .- sqrt.(x), 1, :),
    )
    return decisions, objectives
end

schaffer_x, schaffer_y = schaffer_front()
convex_x, convex_y = convex_front()
zdt_x, zdt_y = zdt1_front()

function schaffer_objective(x)
    return [x[1]^2, (x[1] - 2.0)^2]
end

function convex_objective(x)
    penalty = x[2]^2
    return [x[1]^2 + penalty, (x[1] - 1.0)^2 + penalty]
end

function zdt1_objective(x)
    f1 = x[1]
    g = 1.0 + 9.0 * sum(@view x[2:end]) / (length(x) - 1)
    return [f1, g * (1.0 - sqrt(f1 / g))]
end

const BENCHMARKS = Dict(
    :schaffer => MOBenchmark(
        "Schaffer N.1",
        schaffer_objective,
        reshape([-5.0, 5.0], 1, 2),
        schaffer_y,
        schaffer_x,
        80,
        12,
    ),
    :convex => MOBenchmark(
        "Convex bi-objective",
        convex_objective,
        [-0.5 1.5; -1.0 1.0],
        convex_y,
        convex_x,
        100,
        16,
    ),
    :zdt1 => MOBenchmark(
        "ZDT1 (4D)",
        zdt1_objective,
        [0.0 1.0; 0.0 1.0; 0.0 1.0; 0.0 1.0],
        zdt_y,
        zdt_x,
        180,
        24,
    ),
)

end
