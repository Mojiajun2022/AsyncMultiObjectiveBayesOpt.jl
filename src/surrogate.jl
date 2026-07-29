module Surrogate

using LinearAlgebra
using Random
using Statistics
using ..ParetoTools: nondominated_indices, normalize_objectives

export latin_hypercube, propose_parego

function latin_hypercube(rng::AbstractRNG, bounds::Matrix{Float64}, n::Int)
    d = size(bounds, 1)
    X = Matrix{Float64}(undef, d, n)
    for j in 1:d
        strata = (randperm(rng, n) .- rand(rng, n)) ./ n
        X[j, :] .= bounds[j, 1] .+
                   strata .* (bounds[j, 2] - bounds[j, 1])
    end
    return X
end

scale_to_unit(X, bounds) =
    (X .- bounds[:, 1]) ./ (bounds[:, 2] - bounds[:, 1])

function rbf_kernel(X, Z, length_scale)
    xnorm = sum(abs2, X; dims=1)
    znorm = sum(abs2, Z; dims=1)
    distance2 = max.(xnorm' .+ znorm .- 2 .* (X' * Z), 0.0)
    return exp.(-distance2 ./ (2 * length_scale^2))
end

function gp_factor(X, y, bounds, length_scale, noise)
    Xs = scale_to_unit(X, bounds)
    ymean = mean(y)
    yscale = max(std(y; corrected=false), sqrt(eps(Float64)))
    ynormal = (y .- ymean) ./ yscale
    K = rbf_kernel(Xs, Xs, length_scale)
    jitter = max(noise, eps(Float64))
    for _ in 1:8
        try
            factor = cholesky(Symmetric(K + jitter * I); check=true)
            return factor, Xs, ynormal, ymean, yscale
        catch
            jitter *= 10
        end
    end
    return nothing, Xs, ynormal, ymean, yscale
end

function select_length_scale(X, y, bounds, initial_scale, noise)
    size(X, 2) < 4 && return initial_scale
    scales = unique(clamp.(
        initial_scale .* (0.25, 0.4, 0.63, 1.0, 1.6, 2.5, 4.0),
        0.02, 2.0,
    ))
    best_scale = initial_scale
    best_likelihood = -Inf
    for scale in scales
        factor, _, ynormal, _, _ = gp_factor(
            X, y, bounds, scale, noise,
        )
        factor === nothing && continue
        alpha = factor \ ynormal
        likelihood = -0.5 * dot(ynormal, alpha) -
                     sum(log, diag(factor.L)) -
                     0.5 * length(y) * log(2pi)
        if isfinite(likelihood) && likelihood > best_likelihood
            best_likelihood = likelihood
            best_scale = scale
        end
    end
    return best_scale
end

function gp_predict(X, y, candidates, bounds, length_scale, noise)
    factor, Xs, ynormal, ymean, yscale = gp_factor(
        X, y, bounds, length_scale, noise,
    )
    factor === nothing &&
        return fill(ymean, size(candidates, 2)),
               fill(yscale, size(candidates, 2))
    Cs = scale_to_unit(candidates, bounds)
    cross_kernel = rbf_kernel(Xs, Cs, length_scale)
    alpha = factor \ ynormal
    mu = ymean .+ yscale .* vec(cross_kernel' * alpha)
    projected = factor.L \ cross_kernel
    variance = max.(1 .- vec(sum(abs2, projected; dims=1)), 1e-14)
    return mu, yscale .* sqrt.(variance)
end

const INV_SQRT_2PI = 0.3989422804014327
normal_pdf(x) = INV_SQRT_2PI * exp(-0.5 * x * x)

function normal_cdf(x)
    t = inv(1 + 0.2316419 * abs(x))
    tail = normal_pdf(x) * t * (0.319381530 + t * (-0.356563782 +
           t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return x >= 0 ? 1 - tail : tail
end

function parego_weight(rng, m, proposal_id)
    # Regularly target each extreme while using Dirichlet weights elsewhere.
    cycle = mod1(proposal_id, 2 * m)
    if cycle <= m
        weight = fill(0.001 / max(m - 1, 1), m)
        weight[cycle] = 0.999
        return weight ./ sum(weight)
    end
    weight = -log.(max.(rand(rng, m), eps(Float64)))
    return weight ./ sum(weight)
end

function scalarize(Y, weight, rho)
    values = Vector{Float64}(undef, size(Y, 2))
    for j in axes(Y, 2)
        weighted = weight .* @view(Y[:, j])
        values[j] = maximum(weighted) + rho * sum(weighted)
    end
    return values
end

function candidate_set(rng, bounds, X, Y, count)
    d = size(bounds, 1)
    candidates = latin_hypercube(rng, bounds, count)
    isempty(Y) && return candidates
    front = nondominated_indices(Y)
    isempty(front) && return candidates
    local_count = div(3 * count, 5)
    scales = (0.02, 0.05, 0.1, 0.2, 0.35)
    for j in 1:local_count
        center = @view X[:, front[mod1(j, length(front))]]
        scale = scales[mod1(j, length(scales))]
        candidates[:, j] .= clamp.(
            center .+
            scale .* (bounds[:, 2] - bounds[:, 1]) .* randn(rng, d),
            bounds[:, 1], bounds[:, 2],
        )
    end
    return candidates
end

function distance_mask(candidates, occupied, bounds; tolerance=1e-6)
    isempty(occupied) && return trues(size(candidates, 2))
    C = scale_to_unit(candidates, bounds)
    O = scale_to_unit(occupied, bounds)
    keep = trues(size(C, 2))
    for j in axes(C, 2)
        keep[j] =
            minimum(vec(sum(abs2, O .- C[:, j]; dims=1))) > tolerance^2
    end
    return keep
end

function maximin_candidate(candidates, occupied, bounds)
    isempty(occupied) && return candidates[:, 1]
    C = scale_to_unit(candidates, bounds)
    O = scale_to_unit(occupied, bounds)
    distances = [
        minimum(vec(sum(abs2, O .- C[:, j]; dims=1)))
        for j in axes(C, 2)
    ]
    return candidates[:, argmax(distances)]
end

"""
Select one asynchronous ParEGO point. All objectives in `Y` must already be
transformed to minimization. Pending points receive Kriging-believer scalar
fantasies so concurrently running evaluations remain separated.
"""
function propose_parego(
        rng::AbstractRNG, bounds::Matrix{Float64}, X::AbstractMatrix,
        Y::AbstractMatrix, pending::AbstractMatrix, proposal_id::Int;
        candidate_pool::Int, length_scale::Float64, noise::Float64,
        exploration::Float64, optimize_length_scale::Bool,
        augmentation::Float64)
    d = size(bounds, 1)
    count = max(candidate_pool, 128 * d)
    valid = [j for j in axes(Y, 2) if all(isfinite, @view Y[:, j])]
    Xvalid = Matrix(X[:, valid])
    Yvalid = Matrix(Y[:, valid])
    candidates = candidate_set(rng, bounds, Xvalid, Yvalid, count)
    occupied = hcat(X, pending)
    candidates = candidates[:, distance_mask(candidates, occupied, bounds)]
    isempty(candidates) &&
        return bounds[:, 1] .+
               rand(rng, d) .* (bounds[:, 2] - bounds[:, 1])
    length(valid) < 3 &&
        return maximin_candidate(candidates, occupied, bounds)

    normalized, _, _ = normalize_objectives(Yvalid)
    weight = parego_weight(rng, size(Y, 1), proposal_id)
    scalar_y = scalarize(normalized, weight, augmentation)
    fitted_scale = optimize_length_scale ?
        select_length_scale(
            Xvalid, scalar_y, bounds, length_scale, noise,
        ) : length_scale
    incumbent = minimum(scalar_y)
    Xfit = Xvalid
    yfit = scalar_y
    if size(pending, 2) > 0
        fantasy, _ = gp_predict(
            Xfit, yfit, pending, bounds, fitted_scale, noise,
        )
        Xfit = hcat(Xfit, pending)
        yfit = vcat(yfit, fantasy)
    end

    mu, sigma = gp_predict(
        Xfit, yfit, candidates, bounds, fitted_scale, noise,
    )
    xi = exploration *
         max(std(scalar_y; corrected=false), sqrt(eps(Float64)))
    delta = incumbent .- mu .- xi
    z = delta ./ sigma
    ei = similar(delta)
    @inbounds for i in eachindex(ei)
        ei[i] =
            delta[i] * normal_cdf(z[i]) + sigma[i] * normal_pdf(z[i])
    end
    return candidates[:, argmax(ei)]
end

end
