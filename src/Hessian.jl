using StaticArrays, LinearAlgebra


struct HessianEigenCache
    λ1::Array{Float32,3}
    λ2::Array{Float32,3}
    λ3::Array{Float32,3}
end

function HessianEigenCache(Nx::Int, Ny::Int, Nz::Int)
    return HessianEigenCache(
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz),
        Array{Float32}(undef, Nx, Ny, Nz)
    )
end


"""
    computeHessianEigenvalues(field) -> HessianEigenCache

Compute Hessian eigenvalues of a scalar field using Fourier derivatives.
Allocates and returns an eigenvalue cache.
"""
function computeHessianEigenvalues(
    field::AbstractArray{<:Real,3},
    kx, ky, kz
)::HessianEigenCache

    Nx, Ny, Nz = size(field)
    cache = HessianEigenCache(Nx, Ny, Nz)

    computeHessianEigenvalues!(field, kx, ky, kz, cache)

    return cache
end


"""
    computeHessianEigenvalues!(field,cache)

Compute Hessian eigenvalues of `field` and store them in `cache`.
"""
function computeHessianEigenvalues!(
    field::AbstractArray{<:Real,3},
    kx, ky, kz,
    cache::HessianEigenCache
)
    fftField = FFTW.fft(field)
    tmp = similar(fftField)

    # Allocate six temporary real buffers (local, not cached)
    Hxx = similar(cache.λ1)
    Hyy = similar(cache.λ1)
    Hzz = similar(cache.λ1)
    Hxy = similar(cache.λ1)
    Hxz = similar(cache.λ1)
    Hyz = similar(cache.λ1)

    computeHessianComponents!(
        fftField, tmp,
        kx, ky, kz,
        Hxx, Hyy, Hzz, Hxy, Hxz, Hyz
    )

    computeEigenvalues!(
        Hxx, Hyy, Hzz, Hxy, Hxz, Hyz,
        cache
    )

    return nothing
end

function computeHessianComponents!(
    fftField, tmp,
    kx, ky, kz,
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz
)
    hessianComp!(tmp, fftField, kx, kx); Hxx .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, ky, ky); Hyy .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, kz, kz); Hzz .= real.(FFTW.ifft(tmp))

    hessianComp!(tmp, fftField, kx, ky); Hxy .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, kx, kz); Hxz .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, ky, kz); Hyz .= real.(FFTW.ifft(tmp))

    return nothing
end


function computeEigenvalues!(
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz,
    cache::HessianEigenCache
)
    @inbounds for I in eachindex(Hxx)
        H = @SMatrix [
            Hxx[I]  Hxy[I]  Hxz[I];
            Hxy[I]  Hyy[I]  Hyz[I];
            Hxz[I]  Hyz[I]  Hzz[I]
        ]

        λ = eigen(Symmetric(H)).values
        cache.λ1[I] = λ[1]
        cache.λ2[I] = λ[2]
        cache.λ3[I] = λ[3]
    end

    return nothing
end

@inline function hessianComp!(
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
