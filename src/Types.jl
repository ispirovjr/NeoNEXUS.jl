"""
    AbstractScaleFilter

Abstract supertype for scale-space filters used in Fourier-space smoothing.

Subtypes must be callable as `filter(field, scale)` and `filter(field, scale, feature)`.
"""
abstract type AbstractScaleFilter end

"""
    AbstractFeature

Abstract supertype for all feature detectors.
"""
abstract type AbstractFeature end

"""
    AbstractMorphologicalFeature <: AbstractFeature

Abstract supertype for morphological feature detectors (sheets, filaments, nodes).

Subtypes store `significanceMap` and `thresholdMap` arrays and act as functors
to compute Hessian-based morphological signatures.
"""
abstract type AbstractMorphologicalFeature <: AbstractFeature end

"""
    CacheMode

Enum controlling Hessian eigenvalue cache behaviour.

- `Read`: reuse a previously computed cache.
- `Write`: compute eigenvalues and store them in the cache.
- `None`: compute eigenvalues without caching.
"""
@enum CacheMode begin
    Read
    Write
    None
end
