struct SheetFeature <: AbstractMorphologicalFeature # Cosmological Walls
    significanceMap::Array{Float32,3} 
    responseMap::Array{Float32,3}  
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function SheetFeature(gridSize::Tuple{Int,Int,Int}, kx, ky, kz)
        sigMap = zeros(Float32, gridSize)
        respMap = zeros(Float32, gridSize)
        new(sigMap, respMap, kx, ky, kz)
    end

end



struct LineFeature <: AbstractMorphologicalFeature  
    significanceMap
    responseMap
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
end

struct NodeFeature <: AbstractMorphologicalFeature
    significanceMap
    responseMap
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
end


function (feature::AbstractFeature)(
    field::AbstractArray{<:Real,3},
    cache::Union{Nothing,HessianEigenCache} = nothing,
    mode::CacheMode = None)



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


function computeSignature(feature::SheetFeature, cache::HessianEigenCache)
    S = zeros(Float32, size(cache.λ1))

    @inbounds for I in CartesianIndices(S)
        λ1 = cache.λ1[I]
        λ2 = cache.λ2[I]
        λ3 = cache.λ3[I]

        # Ratios of eigenvalues
        r21 = abs(λ2 / λ1)
        r31 = abs(λ3 / λ1)

        # Negative eigenvalue masks
        mask1 = λ1 < 0 ? 1f0 : 0f0
        mask2 = λ2 < 0 ? 1f0 : 0f0
        mask3 = λ3 < 0 ? 1f0 : 0f0

        # Condition masks for ratios < 1
        cond21 = r21 < 1 ? 1f0 : 0f0
        cond31 = r31 < 1 ? 1f0 : 0f0

        # Wall (sheet) signature formula
        S[I] = (1 - r21) * (1 - r31) * abs(λ1) * (mask1 * cond21 * cond31)

    end

    return S
end