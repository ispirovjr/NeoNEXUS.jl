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

    return cache
end

function computeHessianComponents!(
    fftField, tmp,
    kx, ky, kz,
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz
)
    # Diagonal components (∂²/∂x², ∂²/∂y², ∂²/∂z²)
    hessianComp!(tmp, fftField, kx, kx, 1, 1); Hxx .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, ky, ky, 2, 2); Hyy .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, kz, kz, 3, 3); Hzz .= real.(FFTW.ifft(tmp))

    # Off-diagonal components (∂²/∂x∂y, ∂²/∂x∂z, ∂²/∂y∂z)
    hessianComp!(tmp, fftField, kx, ky, 1, 2); Hxy .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, kx, kz, 1, 3); Hxz .= real.(FFTW.ifft(tmp))
    hessianComp!(tmp, fftField, ky, kz, 2, 3); Hyz .= real.(FFTW.ifft(tmp))

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

# Extract k-vector component based on dimension (1=x, 2=y, 3=z)
@inline selectK(kVec, i, j, k, dim) = dim == 1 ? kVec[i] : (dim == 2 ? kVec[j] : kVec[k])

"""
Compute Hessian component in Fourier space: ∂²/∂α∂β → -kα·kβ·f̂(k)
Uses 1D k-vectors and dimension indices to construct full 3D derivatives.
"""
@inline function hessianComp!(tmp, fftField, kα, kβ, dimα::Int, dimβ::Int)
    Nx, Ny, Nz = size(fftField)
    
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        kαVal = selectK(kα, i, j, k, dimα)
        kβVal = selectK(kβ, i, j, k, dimβ)
        tmp[i,j,k] = fftField[i,j,k] * (-kαVal * kβVal)
    end
    
    return nothing
end
