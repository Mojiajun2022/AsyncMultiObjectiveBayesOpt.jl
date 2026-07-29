module ParetoTools

export objective_signs, dominates, nondominated_indices, normalize_objectives,
       compromise_index, hypervolume_2d, inverted_generational_distance

function objective_signs(senses, m::Int)
    length(senses) == m ||
        throw(ArgumentError("objective_senses must contain $m entries"))
    signs = Vector{Float64}(undef, m)
    for i in 1:m
        sense = senses[i]
        sense in (:min, :max) ||
            throw(ArgumentError("objective_senses entries must be :min or :max"))
        signs[i] = sense == :min ? 1.0 : -1.0
    end
    return signs
end

"""Return true when `a` Pareto-dominates `b` under the requested senses."""
function dominates(a::AbstractVector, b::AbstractVector;
                   senses=fill(:min, length(a)))
    length(a) == length(b) ||
        throw(DimensionMismatch("objective vectors must have the same length"))
    signs = objective_signs(senses, length(a))
    za = signs .* a
    zb = signs .* b
    return all(za .<= zb) && any(za .< zb)
end

"""
    nondominated_indices(Y; senses=fill(:min, size(Y, 1)))

Return column indices of the finite, mutually nondominated observations in `Y`.
Objectives occupy rows and observations occupy columns.
"""
function nondominated_indices(Y::AbstractMatrix;
                              senses=fill(:min, size(Y, 1)))
    m, n = size(Y)
    signs = objective_signs(senses, m)
    valid = [j for j in 1:n if all(isfinite, @view Y[:, j])]
    keep = trues(length(valid))
    for a in eachindex(valid)
        keep[a] || continue
        za = signs .* @view(Y[:, valid[a]])
        for b in eachindex(valid)
            a == b && continue
            zb = signs .* @view(Y[:, valid[b]])
            if all(zb .<= za) && any(zb .< za)
                keep[a] = false
                break
            end
        end
    end
    return valid[keep]
end

function normalize_objectives(Y::AbstractMatrix)
    ideal = vec(minimum(Y; dims=2))
    nadir = vec(maximum(Y; dims=2))
    scale = max.(nadir .- ideal, sqrt(eps(Float64)))
    return (Y .- ideal) ./ scale, ideal, scale
end

function compromise_index(Y::AbstractMatrix, indices::AbstractVector{<:Integer},
                          signs::AbstractVector)
    isempty(indices) && throw(ArgumentError("Pareto index set cannot be empty"))
    Z = signs .* Y[:, indices]
    normalized, _, _ = normalize_objectives(Z)
    distances = vec(sum(abs2, normalized; dims=1))
    return indices[argmin(distances)]
end

"""
    hypervolume_2d(Y, reference; senses=[:min, :min])

Compute the exact dominated hypervolume for a two-objective point set. The
reference point must be worse than the counted points in both transformed
minimization objectives.
"""
function hypervolume_2d(Y::AbstractMatrix, reference::AbstractVector;
                        senses=[:min, :min])
    size(Y, 1) == 2 || throw(ArgumentError("hypervolume_2d requires two objectives"))
    length(reference) == 2 || throw(ArgumentError("reference must have length two"))
    signs = objective_signs(senses, 2)
    Z = signs .* Y
    ref = signs .* reference
    valid = [j for j in axes(Z, 2)
             if all(isfinite, @view Z[:, j]) && all(@view(Z[:, j]) .< ref)]
    isempty(valid) && return 0.0
    front = nondominated_indices(Z[:, valid])
    P = Z[:, valid[front]]
    order = sortperm(@view P[1, :])
    hv = 0.0
    previous_y = ref[2]
    for j in order
        x, y = P[1, j], P[2, j]
        if y < previous_y
            hv += max(ref[1] - x, 0.0) * (previous_y - y)
            previous_y = y
        end
    end
    return hv
end

"""
    inverted_generational_distance(observed, reference; normalize=true)

Average distance from each sampled reference-front point to the nearest
observed objective vector. Objectives occupy rows.
"""
function inverted_generational_distance(observed::AbstractMatrix,
                                        reference::AbstractMatrix;
                                        normalize::Bool=true)
    size(observed, 1) == size(reference, 1) ||
        throw(DimensionMismatch("observed and reference objectives differ"))
    valid = [j for j in axes(observed, 2)
             if all(isfinite, @view observed[:, j])]
    isempty(valid) && return Inf
    O = Matrix(observed[:, valid])
    R = Matrix(reference)
    if normalize
        low = vec(minimum(R; dims=2))
        high = vec(maximum(R; dims=2))
        scale = max.(high .- low, sqrt(eps(Float64)))
        O = (O .- low) ./ scale
        R = (R .- low) ./ scale
    end
    total = 0.0
    for j in axes(R, 2)
        total += sqrt(minimum(vec(sum(abs2, O .- R[:, j]; dims=1))))
    end
    return total / size(R, 2)
end

end
