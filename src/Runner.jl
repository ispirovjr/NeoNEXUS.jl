# Orchestrates multi-scale feature detection pipeline
struct NeoNEXUSRunner # will change later for array features and filters
    filter::AbstractScaleFilter
    features::Vector{AbstractFeature}
    scales::Vector{Float64}
end

function run(
    runner::NeoNEXUSRunner,
    densityField::AbstractArray,
    method::SignatureMethod
)
    # scale loop
    # feature loop
    # max aggregation
end
