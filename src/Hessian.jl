"""
Stores sorted eigenvalues λ1 ≤ λ2 ≤ λ3 per voxel

Should be used in between features for a given smoothing scale to avoid
recomputing the same eigenvalues multiple times.
"""
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
    # Use real-to-Complex FFT for halving memory and compute
    fftField = FFTW.rfft(field)
    Nx = size(field, 1)
    tmp = similar(fftField)

    Hxx = similar(cache.λ1)
    Hyy = similar(cache.λ1)
    Hzz = similar(cache.λ1)
    Hxy = similar(cache.λ1)
    Hxz = similar(cache.λ1)
    Hyz = similar(cache.λ1)

    # Allocate zeroed-Nyquist wavevectors to preserve Hermitian symmetry in cross-derivatives
    Ny, Nz = size(field, 2), size(field, 3)
    kx_odd = copy(kx)
    kx_odd[end] = 0.0  # rfftfreq ends at Nyquist
    ky_odd = copy(ky)
    ky_odd[Ny÷2+1] = 0.0  # fftfreq has Nyquist at N/2 + 1
    kz_odd = copy(kz)
    kz_odd[Nz÷2+1] = 0.0

    # Compute all 6 Hessian components in Real Space
    computeHessianComponents!(fftField, tmp, kx, ky, kz, kx_odd, ky_odd, kz_odd, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, Nx)

    # Compute eigenvalues (λ1, λ2, λ3) for every voxel directly into cache
    computeEigenvalues!(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, cache)

    return cache
end

function computeHessianComponents!(
    fftField, tmp,
    kx, ky, kz,
    kx_odd, ky_odd, kz_odd,
    Hxx::AbstractArray{<:Real,3}, Hyy::AbstractArray{<:Real,3}, Hzz::AbstractArray{<:Real,3},
    Hxy::AbstractArray{<:Real,3}, Hxz::AbstractArray{<:Real,3}, Hyz::AbstractArray{<:Real,3},
    Nx::Int
)
    # Diagonal components (∂²/∂x², ∂²/∂y², ∂²/∂z²)
    hessianComp!(tmp, fftField, kx, kx, 1, 1)
    Hxx .= FFTW.irfft(tmp, Nx)
    hessianComp!(tmp, fftField, ky, ky, 2, 2)
    Hyy .= FFTW.irfft(tmp, Nx)
    hessianComp!(tmp, fftField, kz, kz, 3, 3)
    Hzz .= FFTW.irfft(tmp, Nx)

    # Off-diagonal components (∂²/∂x∂y, ∂²/∂x∂z, ∂²/∂y∂z)
    # MUST use odd k-vectors to preserve Hermitian symmetry across the Nyquist frequencies
    hessianComp!(tmp, fftField, kx_odd, ky_odd, 1, 2)
    Hxy .= FFTW.irfft(tmp, Nx)
    hessianComp!(tmp, fftField, kx_odd, kz_odd, 1, 3)
    Hxz .= FFTW.irfft(tmp, Nx)
    hessianComp!(tmp, fftField, ky_odd, kz_odd, 2, 3)
    Hyz .= FFTW.irfft(tmp, Nx)

    return nothing
end


"""
    eigvals3x3sym(a11, a22, a33, a12, a13, a23)

Analytical eigenvalues of a 3×3 symmetric matrix using Cardano's formula.
Returns sorted eigenvalues (λ1 ≤ λ2 ≤ λ3).

The characteristic polynomial of a 3×3 symmetric matrix is a depressed cubic.
This implementation uses the trigonometric solution (all roots are real for symmetric matrices).

Note: All internal computation is done in Float64 for numerical stability,
regardless of the input type.
"""
@inline function eigvals3x3sym(a11::T, a22::T, a33::T, a12::T, a13::T, a23::T) where {T<:Real}
    # Promote to Float64 internally for numerical stability near degenerate cases
    d11 = Float64(a11)
    d22 = Float64(a22)
    d33 = Float64(a33)
    d12 = Float64(a12)
    d13 = Float64(a13)
    d23 = Float64(a23)

    # Trace / 3
    c0 = (d11 + d22 + d33) / 3.0

    # Shift to zero-trace: B = A - c0*I
    b11 = d11 - c0
    b22 = d22 - c0
    b33 = d33 - c0

    # p = (1/6) * tr(B^2)
    # For symmetric B: tr(B^2) = b11^2 + b22^2 + b33^2 + 2*(d12^2 + d13^2 + d23^2)
    p = (b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * (d12 * d12 + d13 * d13 + d23 * d23)) / 6.0

    if p <= 0.0
        # All eigenvalues are equal (A = c0*I)
        c0_T = T(c0)
        return (c0_T, c0_T, c0_T)
    end

    # q = det(B) / 2
    detB = b11 * (b22 * b33 - d23 * d23) - d12 * (d12 * b33 - d23 * d13) + d13 * (d12 * d23 - b22 * d13)
    q = detB / 2.0

    # r = q / p^(3/2) — clamped to [-1, 1] for numerical safety
    sqrtp = sqrt(p)
    r = q / (p * sqrtp)
    r = clamp(r, -1.0, 1.0)

    # Trigonometric solution for depressed cubic
    θ = acos(r) / 3.0

    # Three eigenvalues (sorted: λ1 ≤ λ2 ≤ λ3)
    twosqrtp = 2.0 * sqrtp
    λ3 = c0 + twosqrtp * cos(θ)
    λ1 = c0 + twosqrtp * cos(θ + 2π / 3)
    λ2 = 3.0 * c0 - λ1 - λ3  # Use trace relation for numerical stability

    # Ensure sorting
    if λ1 > λ2
        λ1, λ2 = λ2, λ1
    end
    if λ2 > λ3
        λ2, λ3 = λ3, λ2
        if λ1 > λ2
            λ1, λ2 = λ2, λ1
        end
    end

    return (T(λ1), T(λ2), T(λ3))
end


function computeEigenvalues!(
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz,
    cache::HessianEigenCache
)
    @inbounds for I in eachindex(Hxx)
        λ1, λ2, λ3 = eigvals3x3sym(Hxx[I], Hyy[I], Hzz[I], Hxy[I], Hxz[I], Hyz[I])
        cache.λ1[I] = λ1
        cache.λ2[I] = λ2
        cache.λ3[I] = λ3
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
        tmp[i, j, k] = fftField[i, j, k] * (-kαVal * kβVal)
    end

    return nothing
end
