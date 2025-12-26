module NeoNEXUS

include("Types.jl")
include("Cache.jl")
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

    # orchestration
    NeoNEXUSRunner

end
