# Abstract types
abstract type AbstractScaleFilter end
abstract type AbstractFeature end
abstract type AbstractMorphologicalFeature <: AbstractFeature end

# Enums
@enum SignatureMethod begin
    Default
    NexusPlus
end

@enum CacheMode begin
    Read
    Write
    None
end
