"""
    GaussianFourierFilter <: AbstractScaleFilter

Gaussian smoothing filter applied in Fourier space.

Smoothing kernel: `exp(-k²R²/2)` where `R` is the smoothing scale.

# Constructors
    GaussianFourierFilter(gridSize::Tuple{Int,Int,Int})
    GaussianFourierFilter(kx, ky, kz)
"""
struct GaussianFourierFilter <: AbstractScaleFilter
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function GaussianFourierFilter(gridSize::Tuple{Int,Int,Int})
        Nx, Ny, Nz = gridSize
        kx = FFTW.fftfreq(Nx) .* 2π
        ky = FFTW.fftfreq(Ny) .* 2π
        kz = FFTW.fftfreq(Nz) .* 2π
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end

    function GaussianFourierFilter(kx, ky, kz)
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end



"""
    (filter::GaussianFourierFilter)(field, R)

Apply Gaussian smoothing at scale `R`: `IFFT(FFT(field) ⋅ exp(-k²R²/2))`.
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    R::Real
)
    fftField = FFTW.rfft(densityField)
    Nx, Ny, Nz = size(densityField)
    R² = R^2
    Nkx = size(fftField, 1)

    # Apply Gaussian kernel in Fourier space
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nkx
        k² = filter.kx[i]^2 + filter.ky[j]^2 + filter.kz[k]^2
        fftField[i, j, k] *= exp(-k² * R² / 2)
    end

    return FFTW.irfft(fftField, Nx)
end

"""
    (filter::GaussianFourierFilter)(field, scale, feature::NodeFeature)

Linear Gaussian filtering for nodes (no log transform).
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    scale::Real,
    feature::NodeFeature
)
    return filter(densityField, scale)
end

"""
    (filter::GaussianFourierFilter)(field, scale, feature::AbstractMorphologicalFeature)

Log-Gaussian filtering for sheets/filaments (NEXUS+ behaviour):
`log₁₀(field) → Gaussian filter → 10^result`.
"""
function (filter::GaussianFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    scale::Real,
    feature::AbstractMorphologicalFeature
)
    logField = log10.(max.(densityField, eps(Float32)))
    filteredLog = filter(logField, scale)
    return 10.0 .^ filteredLog
end


"""
    TopHatFourierFilter <: AbstractScaleFilter

Top-hat smoothing filter applied via Fourier space (spherical top-hat in real space).

Kernel: `3(sin(kR) - kR⋅cos(kR)) / (kR)³`.

# Constructors
    TopHatFourierFilter(gridSize::Tuple{Int,Int,Int})
    TopHatFourierFilter(kx, ky, kz)
"""
struct TopHatFourierFilter <: AbstractScaleFilter
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}

    function TopHatFourierFilter(gridSize::Tuple{Int,Int,Int})
        Nx, Ny, Nz = gridSize
        kx = FFTW.rfftfreq(Nx) .* 2π
        ky = FFTW.fftfreq(Ny) .* 2π
        kz = FFTW.fftfreq(Nz) .* 2π
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end

    function TopHatFourierFilter(kx, ky, kz)
        return new(collect(Float64, kx), collect(Float64, ky), collect(Float64, kz))
    end
end

"""
    (filter::TopHatFourierFilter)(field, R)

Apply spherical top-hat smoothing at scale `R` in Fourier space.
"""
function (filter::TopHatFourierFilter)(
    densityField::AbstractArray{<:Real,3},
    R::Real
)
    fftField = FFTW.rfft(densityField)
    Nx, Ny, Nz = size(densityField)
    Nkx = size(fftField, 1)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nkx
        kVal = sqrt(filter.kx[i]^2 + filter.ky[j]^2 + filter.kz[k]^2)
        kR = kVal * R

        # Limit as kR -> 0 is 1.0 
        weight = 1.0
        if abs(kR) > 1e-4
            weight = 3.0 * (sin(kR) - kR * cos(kR)) / (kR^3)
        end

        fftField[i, j, k] *= weight
    end

    return FFTW.irfft(fftField, Nx)
end

"""
    (filter::TopHatFourierFilter)(field, scale, feature::NodeFeature)

Linear top-hat filtering for nodes (no log transform).
"""
function (filter::TopHatFourierFilter)(densityField::AbstractArray{<:Real,3}, scale::Real, feature::NodeFeature)
    return filter(densityField, scale)
end

"""
    (filter::TopHatFourierFilter)(field, scale, feature::AbstractMorphologicalFeature)

Log–top-hat filtering for sheets/filaments (NEXUS+ behaviour).
"""
function (filter::TopHatFourierFilter)(densityField::AbstractArray{<:Real,3}, scale::Real, feature::AbstractMorphologicalFeature)
    logField = log10.(max.(densityField, eps(Float32)))
    filteredLog = filter(logField, scale)
    return 10.0 .^ filteredLog
end
