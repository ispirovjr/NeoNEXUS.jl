using Test
using Statistics
using NeoNEXUS

function centeredGrid(N, L=1.0)
    x = range(-L/2, L/2; length=N)
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
function kgrid(N, L=1.0)
    k = vcat(0:N÷2, -N÷2+1:-1) .* (2π/L)
    reshape(k, N, 1, 1)
end

kx = kgrid(N)
ky = reshape(kgrid(N), 1, N, 1)
kz = reshape(kgrid(N), 1, 1, N)

@testset "Hessian Eigenvalue Tests" begin

    @testset "Constant Field" begin
        field = ones(Float32, N, N, N)
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Linear Field" begin
        field = X .+ Y .+ Z
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Isotropic Quadratic (Node)" begin
        field = X.^2 .+ Y.^2 .+ Z.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(mean(cache.λ1), 2.0; atol=1e-2)
        @test isapprox(mean(cache.λ2), 2.0; atol=1e-2)
        @test isapprox(mean(cache.λ3), 2.0; atol=1e-2)
    end

    @testset "Cylindrical Filament" begin
        field = X.^2 .+ Y.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(mean(cache.λ1), 0.0; atol=1e-2)
        @test isapprox(mean(cache.λ2), 2.0; atol=1e-2)
        @test isapprox(mean(cache.λ3), 2.0; atol=1e-2)
    end

    @testset "Planar Wall" begin
        field = X.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)

        @test isapprox(mean(cache.λ1), 0.0; atol=1e-2)
        @test isapprox(mean(cache.λ2), 0.0; atol=1e-2)
        @test isapprox(mean(cache.λ3), 2.0; atol=1e-2)
    end
end
