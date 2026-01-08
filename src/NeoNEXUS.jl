module NeoNEXUS

using StaticArrays, LinearAlgebra, FFTW, Statistics


include("Types.jl")
include("Hessian.jl")
include("Features.jl")
include("Filters.jl")
include("Thresholds.jl")
include("Runner.jl")

export
    # filters
    AbstractScaleFilter,
    GaussianFourierFilter,

    # features
    AbstractFeature,
    AbstractMorphologicalFeature,
    SheetFeature,
    LineFeature,
    NodeFeature,

    # enums
    SignatureMethod,
    Default,
    NexusPlus,
    CacheMode,
    Read,
    Write,

    # hessians
    computeHessianEigenvalues,
    computeHessianEigenvalues!,

    # thresholds
    flatThreshold!,
    volumeThreshold!,
    massThreshold!,

    # orchestration
    NeoNEXUSRunner

end
