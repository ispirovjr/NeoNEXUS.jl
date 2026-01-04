@testset "Hessian Physics" begin

    # Helper function for grid generation (consistent with upstream tests)
    function centeredGrid(N, L=1.0)
        dx = L/N
        range(-L/2 + dx/2, L/2 - dx/2; length=N)
    end

    N = 32
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)  # symmetric
    Z = reshape(x, 1, 1, :)

    # Fourier wave numbers (consistent with FFT ordering)
    L = 1
    kr = fftfreq(N) .* N .* 2π / L
    kx = kr; ky = kr; kz = kr

    @testset "Constant Field" begin
        # ∇²(const) = 0
        field = ones(Float32, N, N, N)
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Linear Field (Periodic)" begin
        # ∇²(|x|) ≈ 0 everywhere (except origin)
        field = abs.(X)
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        @test isapprox(median(abs.(cache.λ1)), 0.0; atol=0.2)
        @test isapprox(median(abs.(cache.λ2)), 0.0; atol=0.2)
        @test isapprox(median(abs.(cache.λ3)), 0.0; atol=0.2)
    end

    @testset "Isotropic Quadratic (Node)" begin
        # λ = 2 for x²
        field = X.^2 .+ Y.^2 .+ Z.^2
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        @test isapprox(median(cache.λ1), 2.0; atol=0.15)
        @test isapprox(median(cache.λ2), 2.0; atol=0.15)
        @test isapprox(median(cache.λ3), 2.0; atol=0.15)
    end

    @testset "Cylindrical Filament" begin
        # x² + y² -> λ1=0, λ2=2, λ3=2 (sorted: 0, 2, 2)
        field = X.^2 .+ Y.^2 .+ Z.*0
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 2.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end

    @testset "Planar Wall" begin
        # x² -> λ1=0, λ2=0, λ3=2 (sorted: 0, 0, 2)
        field = X.^2 .+ Y.*0 .+ Z.*0
        cache = computeHessianEigenvalues(field, kx, ky, kz)
        @test isapprox(median(cache.λ1), 0.0; atol=0.1)
        @test isapprox(median(cache.λ2), 0.0; atol=0.1)
        @test isapprox(median(cache.λ3), 2.0; atol=0.1)
    end

end
