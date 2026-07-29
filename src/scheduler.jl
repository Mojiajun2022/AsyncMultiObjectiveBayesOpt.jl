module Scheduler

using MPI
using Printf
using Random
using ..ParetoTools: objective_signs, nondominated_indices, compromise_index,
                     hypervolume_2d
using ..Surrogate: latin_hypercube, propose

export async_mobo

const WORK_TAG = 8301
const RESULT_TAG = 8302
const STOP_TAG = 8303

function validate_inputs(bounds, n_objectives, max_evals, n_initial,
                         candidate_pool, length_scale, noise, augmentation,
                         acquisition, ehvi_samples)
    ndims(bounds) == 2 && size(bounds, 2) == 2 ||
        throw(ArgumentError("bounds must be a d x 2 matrix"))
    size(bounds, 1) > 0 || throw(ArgumentError("bounds cannot be empty"))
    all(isfinite, bounds) || throw(ArgumentError("bounds must be finite"))
    all(bounds[:, 1] .< bounds[:, 2]) ||
        throw(ArgumentError("each lower bound must be less than its upper bound"))
    n_objectives >= 2 ||
        throw(ArgumentError("n_objectives must be at least two"))
    max_evals > 0 || throw(ArgumentError("max_evals must be positive"))
    n_initial >= 0 || throw(ArgumentError("n_initial must be non-negative"))
    candidate_pool > 0 ||
        throw(ArgumentError("candidate_pool must be positive"))
    length_scale > 0 || throw(ArgumentError("length_scale must be positive"))
    noise > 0 || throw(ArgumentError("noise must be positive"))
    augmentation >= 0 ||
        throw(ArgumentError("augmentation must be non-negative"))
    acquisition in (:auto, :ehvi, :parego) ||
        throw(ArgumentError(
            "acquisition must be :auto, :ehvi, or :parego",
        ))
    acquisition == :ehvi && n_objectives != 2 &&
        throw(ArgumentError("EHVI currently supports exactly two objectives"))
    ehvi_samples > 0 ||
        throw(ArgumentError("ehvi_samples must be positive"))
end

function prepare_initial_points(rng, bounds, count, initial_points)
    d = size(bounds, 1)
    supplied = if initial_points === nothing
        zeros(d, 0)
    elseif initial_points isa AbstractVector
        reshape(Vector{Float64}(initial_points), :, 1)
    else
        Matrix{Float64}(initial_points)
    end
    size(supplied, 1) == d ||
        throw(ArgumentError("initial_points must have one row per parameter"))
    size(supplied, 2) <= count ||
        throw(ArgumentError(
            "n_initial must cover all supplied initial points",
        ))
    all(bounds[:, 1] .<= supplied) &&
        all(supplied .<= bounds[:, 2]) ||
        throw(ArgumentError("initial_points must lie inside bounds"))
    random_count = count - size(supplied, 2)
    random_points = random_count == 0 ? zeros(d, 0) :
                    latin_hypercube(rng, bounds, random_count)
    return hcat(supplied, random_points)
end

function evaluate_safely(objective, x, n_objectives)
    started = time()
    try
        raw = objective(copy(x))
        value = collect(Float64, raw)
        length(value) == n_objectives ||
            throw(DimensionMismatch(
                "objective returned $(length(value)) values; " *
                "expected $n_objectives",
            ))
        all(isfinite, value) || fill!(value, NaN)
        return value, time() - started
    catch err
        @warn "objective evaluation failed" x exception=(
            err, catch_backtrace(),
        )
        return fill(NaN, n_objectives), time() - started
    end
end

function make_result(X, Y, durations, elapsed, worker_count, senses)
    valid = [j for j in axes(Y, 2) if all(isfinite, @view Y[:, j])]
    isempty(valid) &&
        error("all objective evaluations failed or returned non-finite values")
    front = nondominated_indices(Y; senses)
    signs = objective_signs(senses, size(Y, 1))
    compromise = compromise_index(Y, front, signs)
    hv = if size(Y, 1) == 2
        Z = signs .* Y[:, valid]
        ideal = vec(minimum(Z; dims=2))
        nadir = vec(maximum(Z; dims=2))
        scale = max.(nadir .- ideal, sqrt(eps(Float64)))
        transformed_reference = nadir .+ 0.1 .* scale
        reference = signs .* transformed_reference
        hypervolume_2d(Y[:, front], reference; senses)
    else
        nothing
    end
    return (
        pareto_X=copy(X[:, front]),
        pareto_Y=copy(Y[:, front]),
        pareto_indices=front,
        compromise_x=copy(X[:, compromise]),
        compromise_y=copy(Y[:, compromise]),
        X_history=X,
        Y_history=Y,
        evaluation_seconds=durations,
        elapsed_seconds=elapsed,
        n_evaluations=size(Y, 2),
        n_failures=size(Y, 2) - length(valid),
        worker_count=worker_count,
        objective_senses=collect(senses),
        hypervolume_2d=hv,
    )
end

function worker_loop(objective, dimension, n_objectives, comm, root,
                     gc_after_evaluation)
    point = Vector{Float64}(undef, dimension)
    message = Vector{Float64}(undef, n_objectives + 1)
    while true
        status = MPI.Probe(root, MPI.ANY_TAG, comm, MPI.Status)
        if MPI.Get_tag(status) == STOP_TAG
            MPI.Recv!(zeros(UInt8, 1), root, STOP_TAG, comm)
            return nothing
        end
        MPI.Recv!(point, root, WORK_TAG, comm)
        value, duration = evaluate_safely(
            objective, point, n_objectives,
        )
        message[1:n_objectives] .= value
        message[end] = duration
        MPI.Send(message, root, RESULT_TAG, comm)
        gc_after_evaluation && GC.gc(true)
    end
end

function serial_loop(objective, bounds, initial, max_evals, n_objectives,
                     senses, rng, options)
    d = size(bounds, 1)
    signs = objective_signs(senses, n_objectives)
    X = Matrix{Float64}(undef, d, max_evals)
    Y = Matrix{Float64}(undef, n_objectives, max_evals)
    durations = Vector{Float64}(undef, max_evals)
    started = time()
    initial_count = size(initial, 2)
    for i in 1:max_evals
        x = if i <= initial_count
            initial[:, i]
        else
            propose(
                rng, bounds, @view(X[:, 1:i-1]),
                signs .* @view(Y[:, 1:i-1]), zeros(d, 0),
                i - initial_count;
                options...,
            )
        end
        X[:, i] .= x
        Y[:, i], durations[i] = evaluate_safely(
            objective, x, n_objectives,
        )
    end
    return make_result(
        X, Y, durations, time() - started, 1, senses,
    )
end

"""
    async_mobo(objective, bounds; n_objectives, max_evals=100, kwargs...)

Run asynchronous MPI multi-objective Bayesian optimization. Every rank calls
this function. Rank zero returns a Pareto archive and worker ranks return
`nothing`. Objective values must be returned as a finite vector.

`bounds` is a `d x 2` matrix containing lower and upper parameter bounds.
`n_objectives` is required. `objective_senses` contains `:min` or `:max` for
each objective. `acquisition=:auto` selects Monte Carlo EHVI for two objectives
and ParEGO for three or more. `max_evals` includes the initial design.

Important keyword arguments include `n_initial`, `initial_points`,
`candidate_pool`, `length_scale`, `noise`, `exploration`, `augmentation`,
`optimize_length_scale`, `ehvi_samples`, `initial_concurrency`,
`gc_after_evaluation`, `seed`, `root`, `comm`, and `verbose`.

The root result contains the Pareto archive, full evaluation history, a
normalized-ideal compromise point, failure counts, timing, and two-objective
hypervolume. History columns are ordered by completion time.
"""
function async_mobo(
        objective::Function, bounds::AbstractMatrix;
        n_objectives::Int, objective_senses=fill(:min, n_objectives),
        max_evals::Int=100, n_initial::Int=0, initial_points=nothing,
        candidate_pool::Int=8192, length_scale::Real=0.2,
        noise::Real=1e-8, exploration::Real=0.005,
        augmentation::Real=0.05, optimize_length_scale::Bool=true,
        acquisition::Symbol=:auto, ehvi_samples::Int=64,
        initial_concurrency::Int=0, gc_after_evaluation::Bool=false,
        seed::Int=42, root::Int=0, comm=MPI.COMM_WORLD,
        verbose::Bool=true)
    validate_inputs(
        bounds, n_objectives, max_evals, n_initial, candidate_pool,
        length_scale, noise, augmentation, acquisition, ehvi_samples,
    )
    senses = collect(objective_senses)
    signs = objective_signs(senses, n_objectives)
    initial_concurrency >= 0 ||
        throw(ArgumentError("initial_concurrency must be non-negative"))
    MPI.Initialized() || MPI.Init()
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    0 <= root < nranks ||
        throw(ArgumentError("root is not a valid MPI rank"))
    matrix_bounds = Matrix{Float64}(bounds)
    rank != root && return worker_loop(
        objective, size(bounds, 1), n_objectives, comm, root,
        gc_after_evaluation,
    )

    workers = nranks - 1
    d = size(bounds, 1)
    automatic_initial = max(
        3 * d + 1,
        5 * n_objectives,
        2 * min(max(workers, 1), 4 * d),
    )
    initial_count = min(
        n_initial == 0 ? automatic_initial : n_initial,
        max_evals,
    )
    rng = MersenneTwister(seed)
    initial = prepare_initial_points(
        rng, matrix_bounds, initial_count, initial_points,
    )
    options = (
        candidate_pool=candidate_pool,
        length_scale=Float64(length_scale),
        noise=Float64(noise),
        exploration=Float64(exploration),
        optimize_length_scale=optimize_length_scale,
        augmentation=Float64(augmentation),
        acquisition=acquisition,
        ehvi_samples=ehvi_samples,
    )

    if nranks == 1
        verbose &&
            @info "one MPI rank detected; using sequential fallback"
        return serial_loop(
            objective, matrix_bounds, initial, max_evals,
            n_objectives, senses, rng, options,
        )
    end

    X = Matrix{Float64}(undef, d, max_evals)
    Y = Matrix{Float64}(undef, n_objectives, max_evals)
    durations = Vector{Float64}(undef, max_evals)
    assigned = Dict{Int,Vector{Float64}}()
    dispatched = 0
    completed = 0
    stopped = 0
    idle_workers = Int[]
    started = time()

    function next_point()
        dispatched < initial_count &&
            return initial[:, dispatched + 1]
        pending = isempty(assigned) ? zeros(d, 0) :
                  hcat(values(assigned)...)
        return propose(
            rng, matrix_bounds, @view(X[:, 1:completed]),
            signs .* @view(Y[:, 1:completed]), pending,
            dispatched - initial_count + 1;
            options...,
        )
    end

    function dispatch(worker)
        point = next_point()
        MPI.Send(point, worker, WORK_TAG, comm)
        assigned[worker] = copy(point)
        dispatched += 1
    end

    warmup_workers = initial_concurrency == 0 ?
        min(workers, initial_count) :
        min(workers, initial_concurrency, initial_count)
    started_workers = 0
    for worker in 0:(nranks - 1)
        worker == root && continue
        if started_workers < warmup_workers &&
                dispatched < initial_count
            dispatch(worker)
            started_workers += 1
        else
            push!(idle_workers, worker)
        end
    end

    function release_idle_workers!()
        while !isempty(idle_workers)
            worker = pop!(idle_workers)
            if dispatched < max_evals
                dispatch(worker)
            else
                MPI.Send(zeros(UInt8, 1), worker, STOP_TAG, comm)
                stopped += 1
            end
        end
    end

    message = zeros(n_objectives + 1)
    while completed < max_evals
        _, status = MPI.Recv!(
            message, MPI.ANY_SOURCE, RESULT_TAG, comm, MPI.Status,
        )
        worker = MPI.Get_source(status)
        completed += 1
        X[:, completed] .= pop!(assigned, worker)
        Y[:, completed] .= message[1:n_objectives]
        durations[completed] = message[end]

        if verbose
            front_size = length(nondominated_indices(
                @view(Y[:, 1:completed]); senses,
            ))
            values = join(
                [@sprintf("%.6g", v) for v in @view Y[:, completed]],
                ", ",
            )
            @printf(
                "[AsyncMOBO] %d/%d rank=%d y=[%s] pareto=%d time=%.2fs\n",
                completed, max_evals, worker, values, front_size,
                durations[completed],
            )
        end

        if completed < initial_count
            if dispatched < initial_count
                dispatch(worker)
            else
                push!(idle_workers, worker)
            end
        elseif completed == initial_count
            push!(idle_workers, worker)
            release_idle_workers!()
        elseif dispatched < max_evals
            dispatch(worker)
        else
            MPI.Send(zeros(UInt8, 1), worker, STOP_TAG, comm)
            stopped += 1
        end
    end
    release_idle_workers!()
    stopped == workers ||
        error("scheduler stopped $stopped of $workers workers")
    return make_result(
        X, Y, durations, time() - started, workers, senses,
    )
end

end
