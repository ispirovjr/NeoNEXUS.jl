"""
    SheetFeature <: AbstractMorphologicalFeature

Detector for planar (wall) structures.

Walls exhibit one dominant negative eigenvalue: λ1 < 0 with |λ2/λ1|, |λ3/λ1| < 1.

# Fields
- `significanceMap::Array{Float32,3}` — accumulated signature values
- `thresholdMap::Array{Float32,3}` — binary classification mask
- `kx, ky, kz::Vector{Float64}` — Fourier-space wavevectors
"""
struct SheetFeature <: AbstractMorphologicalFeature
    significanceMap::Array{Float32,3}
    thresholdMap::Array{Float32,3}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function SheetFeature(gridSize::Tuple{Int,Int,Int}, kx, ky, kz)
        sigMap = zeros(Float32, gridSize)
        thresMap = zeros(Float32, gridSize)
        new(sigMap, thresMap, collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end


"""
    LineFeature <: AbstractMorphologicalFeature

Detector for linear (filament) structures.

Filaments exhibit two dominant negative eigenvalues: λ1, λ2 < 0 with |λ3/λ1| < 1.

# Fields
- `significanceMap::Array{Float32,3}` — accumulated signature values
- `thresholdMap::Array{Float32,3}` — binary classification mask
- `kx, ky, kz::Vector{Float64}` — Fourier-space wavevectors
"""
struct LineFeature <: AbstractMorphologicalFeature
    significanceMap::Array{Float32,3}
    thresholdMap::Array{Float32,3}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function LineFeature(gridSize::Tuple{Int,Int,Int}, kx, ky, kz)
        sigMap = zeros(Float32, gridSize)
        thresMap = zeros(Float32, gridSize)
        new(sigMap, thresMap, collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end

"""
    NodeFeature <: AbstractMorphologicalFeature

Detector for nodal (spherical collapse) structures.

Nodes have all three eigenvalues negative: λ1, λ2, λ3 < 0.

# Fields
- `significanceMap::Array{Float32,3}` — accumulated signature values
- `thresholdMap::Array{Float32,3}` — binary classification mask
- `kx, ky, kz::Vector{Float64}` — Fourier-space wavevectors
"""
struct NodeFeature <: AbstractMorphologicalFeature
    significanceMap::Array{Float32,3}
    thresholdMap::Array{Float32,3}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function NodeFeature(gridSize::Tuple{Int,Int,Int}, kx, ky, kz)
        sigMap = zeros(Float32, gridSize)
        thresMap = zeros(Float32, gridSize)
        new(sigMap, thresMap, collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end

"""
    (feature::AbstractMorphologicalFeature)(field [, cache, mode])

Evaluate the morphological signature of `field`.

Computes Hessian eigenvalues (respecting `mode` ∈ {`None`, `Read`, `Write`}),
then calls [`computeSignature`](@ref). Results are aggregated into
`feature.significanceMap` via element-wise max across scales.

Returns the per-voxel signature array for this invocation.
"""
function (feature::AbstractMorphologicalFeature)(
    field::AbstractArray{<:Real,3},
    cache::Union{Nothing,HessianEigenCache}=nothing,
    mode::CacheMode=None)

    if mode == None
        localCache = computeHessianEigenvalues(field, feature.kx, feature.ky, feature.kz)
    elseif mode == Read
        @assert cache !== nothing "Cache must be provided in Read mode"
        localCache = cache
    elseif mode == Write
        @assert cache !== nothing "Cache must be provided in Write mode"
        computeHessianEigenvalues!(field, feature.kx, feature.ky, feature.kz, cache)
        localCache = cache
    end

    sigMap = computeSignature(feature, localCache)

    # voxel-wise max aggregation for multi-scale processing
    @. feature.significanceMap = max(feature.significanceMap, sigMap)

    return sigMap
end


"""
    computeSignature(feature, cache) -> Array{Float32,3}

Compute the morphological signature for `feature` from Hessian eigenvalues
stored in `cache`. Dispatches on the feature type:

- [`SheetFeature`](@ref): `S = (1 - |λ2/λ1|)(1 - |λ3/λ1|)|λ1|` when λ1 < 0.
- [`LineFeature`](@ref): `S = |λ2²/λ1|(1 - |λ3/λ1|)` when λ1, λ2 < 0.
- [`NodeFeature`](@ref): `S = |λ3²/λ1|` when λ1, λ2, λ3 < 0.
"""
function computeSignature end

function computeSignature(feature::SheetFeature, cache::HessianEigenCache)
    S = zeros(Float32, size(cache.λ1))

    @inbounds for I in CartesianIndices(S)
        λ1 = cache.λ1[I]
        λ2 = cache.λ2[I]
        λ3 = cache.λ3[I]

        # Avoid division by zero
        invλ1 = abs(λ1) > eps(Float32) ? 1f0 / λ1 : 0f0

        # Ratios of eigenvalues
        r21 = abs(λ2 * invλ1)
        r31 = abs(λ3 * invλ1)

        # Negative eigenvalue mask
        m1 = λ1 < 0 ? 1f0 : 0f0

        # Condition masks for ratios < 1
        c21 = r21 < 1 ? 1f0 : 0f0
        c31 = r31 < 1 ? 1f0 : 0f0

        # Wall signature: strength proportional to |λ1|, penalized by ratio magnitudes
        S[I] = (1f0 - r21) * (1f0 - r31) * abs(λ1) * (m1 * c21 * c31)
    end

    return S
end


function computeSignature(feature::LineFeature, cache::HessianEigenCache)
    S = zeros(Float32, size(cache.λ1))

    @inbounds for I in CartesianIndices(S)
        λ1 = cache.λ1[I]
        λ2 = cache.λ2[I]
        λ3 = cache.λ3[I]

        # Avoid division by zero
        invλ1 = abs(λ1) > eps(Float32) ? 1f0 / λ1 : 0f0

        # Ratio r31
        r31 = abs(λ3 * invλ1)

        # Negative eigenvalue masks
        m1 = λ1 < 0 ? 1f0 : 0f0
        m2 = λ2 < 0 ? 1f0 : 0f0

        # Condition mask for r31 < 1
        c31 = r31 < 1 ? 1f0 : 0f0

        # Filament signature: |λ2²/λ1| * (1 - r31) * masks
        S[I] = abs(λ2 * λ2 * invλ1) * (1f0 - r31) * (m1 * m2 * c31)
    end

    return S
end


function computeSignature(feature::NodeFeature, cache::HessianEigenCache)
    S = zeros(Float32, size(cache.λ1))

    @inbounds for I in CartesianIndices(S)
        λ1 = cache.λ1[I]
        λ2 = cache.λ2[I]
        λ3 = cache.λ3[I]

        # Avoid division by zero
        invλ1 = abs(λ1) > eps(Float32) ? 1f0 / λ1 : 0f0

        # Negative eigenvalue masks
        m1 = λ1 < 0 ? 1f0 : 0f0
        m2 = λ2 < 0 ? 1f0 : 0f0
        m3 = λ3 < 0 ? 1f0 : 0f0

        # Node signature: |λ3²/λ1| * all masks
        S[I] = abs(λ3 * λ3 * invλ1) * (m1 * m2 * m3)
    end

    return S
end