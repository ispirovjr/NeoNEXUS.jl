"""
GaussianFourierFilter struct. Carries k-vectors for Fourier space filtering.
Uses functor pattern for filtering once created (see below).
"""
struct GaussianFourierFilter <: AbstractScaleFilter
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    # Constructor that just takes box sizes
    function GaussianFourierFilter(gridSize::Tuple{Int,Int,Int})
        Nx, Ny, Nz = gridSize
        kx = FFTW.fftfreq(Nx) .* 2π
        ky = FFTW.fftfreq(Ny) .* 2π
        kz = FFTW.fftfreq(Nz) .* 2π
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end

    # Constructor that takes k-vectors directly (collects to Vector{Float64} since FFTW returns Frequency Vector)
    function GaussianFourierFilter(kx, ky, kz)
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end



"""
Standard Gaussian filtering in Fourier space:
    filtered = IFFT( FFT(field) * exp(-k²R²/2) )
where R = scale.
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    R::Real
)
    fftField = FFTW.fft(densityField)
    Nx, Ny, Nz = size(densityField)
    R² = R^2

    # Apply Gaussian kernel in Fourier space
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        k² = filter.kx[i]^2 + filter.ky[j]^2 + filter.kz[k]^2
        fftField[i, j, k] *= exp(-k² * R² / 2)
    end

    return real.(FFTW.ifft(fftField))
end

"""
For NodeFeature: use standard (linear) Gaussian filtering.
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    scale::Real,
    feature::NodeFeature
)
    return filter(densityField, scale)
end

"""
For Sheet/Line features: use log-filtering.
Process: log₁₀(density) → Gaussian filter → 10^(result)
NEXUS+ behavior
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    scale::Real,
    feature::AbstractMorphologicalFeature
)
    # Log-filtering: take log, filter, then exponentiate
    # Add small epsilon to avoid log(0)
    logField = log10.(max.(densityField, eps(Float32)))
    filteredLog = filter(logField, scale)
    return 10.0 .^ filteredLog
end


"""
TopHatFourierFilter struct. Carries k-vectors for Fourier space filtering.
Top Hat in Real Space, applied via Fourier. Corresponds to a Bessel Function.
"""
struct TopHatFourierFilter <: AbstractScaleFilter
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function TopHatFourierFilter(gridSize::Tuple{Int,Int,Int})
        Nx, Ny, Nz = gridSize
        kx = FFTW.fftfreq(Nx) .* 2π
        ky = FFTW.fftfreq(Ny) .* 2π
        kz = FFTW.fftfreq(Nz) .* 2π
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end

    function TopHatFourierFilter(kx, ky, kz)
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end

function (filter::TopHatFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    R::Real
)
    fftField = FFTW.fft(densityField)
    Nx, Ny, Nz = size(densityField)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        kVal = sqrt(filter.kx[i]^2 + filter.ky[j]^2 + filter.kz[k]^2)
        kR = kVal * R

        # Limit as kR -> 0 is 1.0 
        weight = 1.0
        if abs(kR) > 1e-4
            weight = 3.0 * (sin(kR) - kR * cos(kR)) / (kR^3)
        end

        fftField[i, j, k] *= weight
    end

    return real.(FFTW.ifft(fftField))
end

# Default implementation
function (filter::TopHatFourierFilter)(densityField::AbstractArray{<:Real,3}, scale::Real, feature::NodeFeature)
    return filter(densityField, scale)
end

# NEXUS+ behavior
function (filter::TopHatFourierFilter)(densityField::AbstractArray{<:Real,3}, scale::Real, feature::AbstractMorphologicalFeature)
    logField = log10.(max.(densityField, eps(Float32)))
    filteredLog = filter(logField, scale)
    return 10.0 .^ filteredLog
end
