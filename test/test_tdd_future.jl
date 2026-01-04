# TDD Future Development Tests
# These tests represent expected functionality that is not yet implemented.
# They are marked as @test_broken and will serve as a guide for ongoing development.

@testset "TDD Future Features" begin

    # Shared grid setup for tests
    function centeredGrid(N, L=1.0)
        dx = L/N
        range(-L/2 + dx/2, L/2 - dx/2; length=N)
    end

    N = 16
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)
    kr = fftfreq(N) .* N .* 2π
    kx = kr; ky = kr; kz = kr

    @testset "Filament Detection (LineFeature)" begin
        # Cylindrical overdensity: X² + Y²
        field = X.^2 .+ Y.^2 .+ Z.*0
        line = LineFeature(zeros(Float32, N,N,N), zeros(Float32, N,N,N), kx, ky, kz)
        
        # This will fail current implementation (MethodError in computeSignature)
        @test_broken begin
            sigMap = line(field)
            any(sigMap .> 0)
        end
    end

    @testset "Node Detection (NodeFeature)" begin
        # Isotropic overdensity: X² + Y² + Z²
        field = X.^2 .+ Y.^2 .+ Z.^2
        node = NodeFeature(zeros(Float32, N,N,N), zeros(Float32, N,N,N), kx, ky, kz)
        
        # This will fail current implementation (MethodError in computeSignature)
        @test_broken begin
            sigMap = node(field)
            any(sigMap .> 0)
        end
    end

    @testset "Gaussian Fourier Filter" begin
        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        field = randn(Float32, N, N, N)
        
        # Current implementation throws ErrorException("Forgot to implement filters")
        @test_broken begin
            # Applying filter at scale 2.0
            filtered = filter(field, 2.0)
            size(filtered) == (N, N, N)
        end
    end

    @testset "Runner Pipeline Orchestration" begin
        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        features = AbstractFeature[SheetFeature((N,N,N), kx, ky, kz)]
        scales = [1.0, 2.0]
        runner = NeoNEXUSRunner(filter, features, scales)
        
        field = randn(Float32, N, N, N)
        
        # Current implementation of run() is empty/no-op
        @test_broken begin
            NeoNEXUS.run(runner, field, NeoNEXUS.Default)
            # Check if significance maps are actually populated
            any(features[1].significanceMap .> 0)
        end
    end

end
