# Unit tests for Thresholding functions
# Tests use real density fields with known cluster centers to validate thresholds

@testset "Threshold Functions" begin

    # Create a density field with a Gaussian cluster centered at (cx, cy, cz)
    function create_cluster_density(N, cx, cy, cz; σ=2.0, amplitude=10.0, background=1.0)
        density = fill(Float32(background), N, N, N)
        for k in 1:N, j in 1:N, i in 1:N
            r² = (i - cx)^2 + (j - cy)^2 + (k - cz)^2
            density[i, j, k] += Float32(amplitude * exp(-r² / (2 * σ^2)))
        end
        return density
    end

    # Shared setup
    N = 16
    kx = ky = kz = fftfreq(N) .* N .* 2π

    # Cluster center location
    cx, cy, cz = N ÷ 2, N ÷ 2, N ÷ 2

    @testset "flatThreshold! with cluster density" begin
        # Create density with cluster at center
        density = create_cluster_density(N, cx, cy, cz)

        # Compute node signature (detects spherical overdensities)
        node = NodeFeature((N, N, N), kx, ky, kz)
        sigMap = node(density)

        # Apply flat threshold at median of non-zero signatures
        nonzero = Base.filter(x -> x > 0, vec(sigMap))
        if !isempty(nonzero)
            thresholdVal = flatThreshold!(node, median(nonzero))

            # The cluster center should be included in the threshold
            @test node.thresholdMap[cx, cy, cz] == 1.0f0

            # Threshold map should have some voxels passing
            @test sum(node.thresholdMap) > 0
        end
    end

    @testset "volumeThreshold! with cluster density" begin
        # Create density with cluster at center
        density = create_cluster_density(N, cx, cy, cz)

        # Compute node signature
        node = NodeFeature((N, N, N), kx, ky, kz)
        sigMap = node(density)

        # Apply volume threshold - keep top 30%
        thresholdVal = volumeThreshold!(node, 0.3)

        # The cluster center should have high signature and be included
        @test node.thresholdMap[cx, cy, cz] == 1.0f0

        # Count of passing voxels should be approximately 30% of non-zero voxels
        nonzero_count = count(x -> x > 0, sigMap)
        if nonzero_count > 0
            expected_count = ceil(Int, 0.3 * nonzero_count)
            actual_count = Int(sum(node.thresholdMap))
            # Allow some tolerance due to discrete nature
            @test actual_count >= expected_count - 1
        end
    end

    @testset "volumeThreshold! edge cases" begin
        node = NodeFeature((N, N, N), kx, ky, kz)

        # All zeros → should return 0 threshold
        fill!(node.significanceMap, 0f0)
        thresholdVal = volumeThreshold!(node, 0.5)

        @test thresholdVal == 0f0
        @test sum(node.thresholdMap) == 0f0

        # Single non-zero voxel, 50% → should still keep 1 (min 1)
        node.significanceMap[cx, cy, cz] = 1.0f0
        thresholdVal = volumeThreshold!(node, 0.5)

        @test sum(node.thresholdMap) == 1.0f0
        @test node.thresholdMap[cx, cy, cz] == 1.0f0
    end

    @testset "massThreshold! with cluster density" begin
        # Create density with cluster at center
        density = create_cluster_density(N, cx, cy, cz)

        # Compute node signature
        node = NodeFeature((N, N, N), kx, ky, kz)
        sigMap = node(density)

        # Apply mass threshold - keep top 30% of mass
        thresholdVal = massThreshold!(node, density, 0.3)

        # The cluster center should be included (high signature AND high density)
        @test node.thresholdMap[cx, cy, cz] == 1.0f0

        # Mass threshold should be more selective than volume threshold
        # (fewer voxels should pass)
        @test sum(node.thresholdMap) > 0
    end

    @testset "massThreshold! edge cases" begin
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = zeros(Float32, N, N, N)

        # All zeros → should return 0
        fill!(node.significanceMap, 0f0)
        thresholdVal = massThreshold!(node, density, 0.5)

        @test thresholdVal == 0f0
        @test sum(node.thresholdMap) == 0f0

        # Non-zero sig but zero density → zero mass → return 0
        node.significanceMap[cx, cy, cz] = 1.0f0
        thresholdVal = massThreshold!(node, density, 0.5)

        @test thresholdVal == 0f0
    end

    @testset "Threshold comparison with cluster" begin
        # Create density with strong cluster
        density = create_cluster_density(N, cx, cy, cz, amplitude=20.0)

        # Compute signatures for three separate features
        node1 = NodeFeature((N, N, N), kx, ky, kz)
        node2 = NodeFeature((N, N, N), kx, ky, kz)
        node3 = NodeFeature((N, N, N), kx, ky, kz)

        node1(density)
        node2(density)
        node3(density)

        # Apply different thresholds
        flatThreshold!(node1, 0.1f0)  # Low threshold
        volumeThreshold!(node2, 0.5)   # 50% by volume
        massThreshold!(node3, density, 0.5)  # 50% by mass

        # All three should include the cluster center
        @test node1.thresholdMap[cx, cy, cz] == 1.0f0
        @test node2.thresholdMap[cx, cy, cz] == 1.0f0
        @test node3.thresholdMap[cx, cy, cz] == 1.0f0

        # Mass threshold should be more selective or equal to volume threshold
        # (mass-weighting concentrates on high-density regions)
        @test sum(node3.thresholdMap) <= sum(node2.thresholdMap)
    end

end
