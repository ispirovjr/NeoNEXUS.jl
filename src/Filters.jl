# Applies Gaussian smoothing in Fourier space at given scale
struct GaussianFourierFilter <: AbstractScaleFilter
    parameters
end

function (filter::AbstractScaleFilter)(
    densityField::AbstractArray,
    scale::Real
)
    error("Forgot to implement filters")
end
