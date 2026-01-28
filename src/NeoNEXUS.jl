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
    TopHatFourierFilter,
    TopHatFourierFilter,

    # features
    AbstractFeature,
    AbstractMorphologicalFeature,
    SheetFeature,
    LineFeature,
    NodeFeature,

    # enums
    CacheMode,
    Read,
    Write,
    None,

    # hessians
    computeHessianEigenvalues,
    computeHessianEigenvalues!,

    # thresholds
    flatThreshold!,
    volumeThreshold!,
    massThreshold!,
    massCutoffThreshold!,
    thresholdedAverageDensity,
    averageDensityThreshold!,
    calculateΔM²,
    deltaMSquaredThreshold!,
    componentErosionPercentileThreshold!,
    componentErosionPlateauThreshold!,
    maskSignatureMap!,
    findComponentPercentageThreshold!,


    # connected components
    ConnectedComponent,
    findConnectedComponents,
    labelConnectedComponents,
    componentAverageDensity,
    componentDensityThreshold!,
    pruneSmallComponents!,
    pruneSmallMassComponents!,

    # hessian cache (for advanced usage)
    HessianEigenCache,

    # orchestration
    MMFClassic,
    NEXUSPlus,
    run

end

