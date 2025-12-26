struct GaussianFourierFilter <: AbstractScaleFilter
    parameters
end

function (filter::AbstractScaleFilter)(
    densityField::AbstractArray,
    scale::Real
)
    error("Forgot to implement filters")
end
