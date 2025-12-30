struct SheetFeature <: AbstractMorphologicalFeature # Cosmological Walls
    parameters
end


function (feature::SheetFeature)( # signature computation
    field::AbstractArray,
    method::SignatureMethod
)::AbstractArray
    # compute Hessian
    # compute eigenvalues
    # compute sheet-like signature
end

function (feature::SheetFeature)(
    field::AbstractArray,
    method::SignatureMethod,
    cache::HessianEigenCache,
    mode::CacheMode
)::AbstractArray
    # reuse or populate cache
    # compute signature
end



struct LineFeature <: AbstractMorphologicalFeature # Fillaments
    parameters
end

function (feature::AbstractFeature)(
    field::AbstractArray,
    method::SignatureMethod
)
    error("Feature call not implemented")
end

function (feature::AbstractFeature)(
    field::AbstractArray,
    method::SignatureMethod,
    cache::HessianEigenCache,
    mode::CacheMode
)
    error("Cached feature call not implemented")
end


