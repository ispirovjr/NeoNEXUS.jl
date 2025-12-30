module NeoNEXUS

using StaticArrays, LinearAlgebra, FFTW, Statistics


include("Types.jl")
include("Hessian.jl")
include("Filters.jl")
include("Features.jl")
include("Runner.jl")

export
    # filters
    AbstractScaleFilter,

    # features
    AbstractFeature,
    AbstractMorphologicalFeature,
    SheetFeature,
    LineFeature,

    # enums
    SignatureMethod,
    CacheMode,

    # hessians
    computeHessianEigenvalues,
    computeHessianEigenvalues!,

    # orchestration
    NeoNEXUSRunner

end
