# Extension points for custom filters and features
abstract type AbstractScaleFilter end  # Subtype for scale-space filters (Gaussian, etc.)
abstract type AbstractFeature end
abstract type AbstractMorphologicalFeature <: AbstractFeature end  # Sheet, Line, Blob features

# Enums
@enum CacheMode begin
    Read
    Write
    None
end
