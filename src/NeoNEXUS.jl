module NeoNEXUS

using StaticArrays, LinearAlgebra, FFTW, Statistics


include("Types.jl")
include("Hessian.jl")
include("Features.jl")
include("Filters.jl")
include("Thresholds.jl")
include("ConnectedComponents.jl")
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
    massCutoffThreshold!,
    thresholdedAverageDensity!,
    averageDensityThreshold!,
    calculateΔM²,
    deltaMSquaredThreshold!,
    maskSignatureMap!,


    # connected components
    ConnectedComponent,
    findConnectedComponents,
    labelConnectedComponents,
    componentAverageDensity,
    componentDensityThreshold!,

    # hessian cache (for advanced usage)
    HessianEigenCache,

    # orchestration
    MMFClassic,
    NEXUSPlus,
    run

end

