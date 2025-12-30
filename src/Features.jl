struct SheetFeature <: AbstractMorphologicalFeature # Cosmological Walls
    signifficanceMap
    responseMap
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
end

struct LineFeature <: AbstractMorphologicalFeature 
    signifficanceMap
    responseMap
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
end

struct NodeFeature <: AbstractMorphologicalFeature
    signifficanceMap
    responseMap
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
end


function (feature::AbstractFeature)(
    field::AbstractArray{<:Real,3},
    cache::HessianEigenCache = nothing,
    mode::CacheMode = CacheMode.None)

    if mode == CacheMode.None
        localCache = computeHessianEigenvalues(field, feature.kx, feature.ky, feature.kz)
    elseif mode == CacheMode.Read
        @assert cache !== nothing "Cache must be provided in Read mode"
        localCache = cache
    elseif mode == CacheMode.Write
        @assert cache !== nothing "Cache must be provided in Write mode"
        computeHessianEigenvalues!(field, feature.kx, feature.ky, feature.kz, cache)
        localCache = cache
    end

    sigMap = computeSignature(feature, localCache)

    # Store the significance map in the feature (aggregated or first scale)
    if feature.significanceMap === nothing
        feature.significanceMap = sigMap  # maybe for better type clarity we can set sig to zero from the start
    else
        # voxel-wise max aggregation for multi-scale processing
        @. feature.significanceMap = max(feature.significanceMap, sigMap)
    end

    return sigMap
end


function computeSignature(feature::SheetFeature, cache::HessianEigenCache)
    S = zeros(Float32, size(cache.λ1))

    @inbounds for I in CartesianIndices(S)
        l1, l2, l3 = cache.λ1[I], cache.λ2[I], cache.λ3[I]
        invl1 = 1.0f0 / l1
        r21, r31 = abs(l2*invl1), abs(l3*invl1)
        m1, m2, m3 = l1<0 ? 1f0 : 0f0, l2<0 ? 1f0 : 0f0, l3<0 ? 1f0 : 0f0
        c21, c31 = r21<1 ? 1f0 : 0f0, r31<1 ? 1f0 : 0f0

        # Wall (sheet) signature
        S[I] = (1 - r21)*(1 - r31)*abs(l1)*(m1*c21*c31)
    end

    return S
end