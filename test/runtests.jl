using Test
using Statistics
using NeoNEXUS
using FFTW

function centeredGrid(N, L=1.0)
    # Centered symmetric grid for periodicity: x[i] + x[N-i+1] = 0
    dx = L/N
    x = range(-L/2 + dx/2, L/2 - dx/2; length=N)
    return x
end


const N = 32
x = centeredGrid(N)
y = centeredGrid(N)
z = centeredGrid(N)

X = reshape(x, :, 1, 1)
Y = reshape(y, 1, :, 1)
Z = reshape(z, 1, 1, :)

# Fourier wave numbers (consistent with FFT ordering)
L = 1

kr = fftfreq(N) .* N .* 2π / L
kx = [kx for kx in kr, ky in kr, kz in kr]
ky = [ky for kx in kr, ky in kr, kz in kr] # this works, but grows in memory and we might want to change indexing in the future
kz = [kz for kx in kr, ky in kr, kz in kr]

@testset "Hessian Eigenvalue Tests" begin

    @testset "Constant Field" begin
        field = ones(Float32, N, N, N)
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    # Linear field test removed: not periodic on finite domains

    @testset "Isotropic Quadratic (Node)" begin
        field = X.^2 .+ Y.^2 .+ Z.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        # Use median to avoid boundary artifacts
        @test isapprox(median(cache.λ1), 2.0; atol=0.15)
        @test isapprox(median(cache.λ2), 2.0; atol=0.15)
        @test isapprox(median(cache.λ3), 2.0; atol=0.15)
    end

    @testset "Cylindrical Filament" begin
        field = X.^2 .+ Y.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 2.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end

    @testset "Planar Wall" begin
        field = X.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 0.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end
end
