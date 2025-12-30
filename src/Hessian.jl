struct HessianCache # to avoid computing multiple times
    Hxx::Array{Float32,3}
    Hyy::Array{Float32,3}
    Hzz::Array{Float32,3}
    Hxy::Array{Float32,3}
    Hxz::Array{Float32,3}
    Hyz::Array{Float32,3}
end

function HessianCache(Nx::Int, Ny::Int, Nz::Int)
    return HessianCache(
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz)
    )
end

"""
    computeHessian(field) -> HessianCache

Compute the Hessian of a scalar field using Fourier derivatives.
Allocates and returns a new `HessianCache`.
"""
function computeHessian(
    field::AbstractArray{<:Real,3},
    kx::AbstractArray,
    ky::AbstractArray,
    kz::AbstractArray
)::HessianCache

    Nx, Ny, Nz = size(field)
    cache = HessianCache(Nx, Ny, Nz)

    computeHessian!(field, kx, ky, kz, cache)

    return cache
end


"""
    computeHessian!(field, cache)

Compute the Hessian of `field` in-place and store components in `cache`.
"""
function computeHessian!(
    field::AbstractArray{<:Real,3},
    kx::AbstractArray,
    ky::AbstractArray,
    kz::AbstractArray,
    cache::HessianCache
)
    # Forward FFT
    fftField = FFTW.fft(field)

    # Allocate a single temporary buffer for inverse FFTs
    tmp = similar(fftField)

    computeHessianComponents!(
        fftField, tmp,
        kx, ky, kz,
        cache
    )

    return nothing
end


function computeHessianComponents!(
    fftField,
    tmp,
    kx,
    ky,
    kz,
    cache::HessianCache
)
    # Hxx
    hessianComp!(tmp, fftField, kx, kx)
    cache.Hxx .= real.(FFTW.ifft(tmp))

    # Hyy
    hessianComp!(tmp, fftField, ky, ky)
    cache.Hyy .= real.(FFTW.ifft(tmp))

    # Hzz
    hessianComp!(tmp, fftField, kz, kz)
    cache.Hzz .= real.(FFTW.ifft(tmp))

    # Hxy
    hessianComp!(tmp, fftField, kx, ky)
    cache.Hxy .= real.(FFTW.ifft(tmp))

    # Hxz
    hessianComp!(tmp, fftField, kx, kz)
    cache.Hxz .= real.(FFTW.ifft(tmp))

    # Hyz
    hessianComp!(tmp, fftField, ky, kz)
    cache.Hyz .= real.(FFTW.ifft(tmp))

    return nothing
end


@inline function hessianComp!( # speedy function to not copy-paste
    tmp,
    fftField,
    kα,
    kβ
)
    @inbounds @simd for i in eachindex(fftField)
        tmp[i] = fftField[i] * (-kα[i] * kβ[i])
    end
    return nothing
end
