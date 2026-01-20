# Unit tests for Thresholding functions
# Tests use real density fields with known cluster centers to validate thresholds
using Test, NeoNEXUS, FFTW, Statistics

@testset "Threshold Functions" begin

    # Create a density field with a Gaussian cluster centered at (cx, cy, cz)
    function createClusterDensity(N, cx, cy, cz; σ=2.0, amplitude=10.0, background=1.0)
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
        density = createClusterDensity(N, cx, cy, cz)

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
        density = createClusterDensity(N, cx, cy, cz)

        # Compute node signature
        node = NodeFeature((N, N, N), kx, ky, kz)
        sigMap = node(density)

        # Apply volume threshold - keep top 30%
        thresholdVal = volumeThreshold!(node, 0.3)

        # The cluster center should have high signature and be included
        @test node.thresholdMap[cx, cy, cz] == 1.0f0

        # Count of passing voxels should be approximately 30% of non-zero voxels
        nonzeroCount = count(x -> x > 0, sigMap)
        if nonzeroCount > 0
            expectedCount = ceil(Int, 0.3 * nonzeroCount)
            actualCount = Int(sum(node.thresholdMap))
            # Allow some tolerance due to discrete nature
            @test actualCount >= expectedCount - 1
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
        density = createClusterDensity(N, cx, cy, cz)

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
        density = createClusterDensity(N, cx, cy, cz, amplitude=20.0)

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

    @testset "massCutoffThreshold! with cluster density" begin
        # Create density with cluster at center (background=1.0, peak=11.0)
        density = createClusterDensity(N, cx, cy, cz)

        # Compute node signature
        node = NodeFeature((N, N, N), kx, ky, kz)
        sigMap = node(density)

        # Apply density cutoff at 5.0 (should include cluster center but not background)
        thresholdVal = massCutoffThreshold!(node, density, 5.0f0)

        @test thresholdVal == 5.0f0

        # The cluster center has high density (>5) and should pass
        @test node.thresholdMap[cx, cy, cz] == 1.0f0

        # Background voxels (density=1.0) should not pass even if they have signature
        # Check a corner voxel far from cluster
        @test node.thresholdMap[1, 1, 1] == 0.0f0
    end

    @testset "massCutoffThreshold! edge cases" begin
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = fill(10.0f0, N, N, N)  # All high density

        # All zeros signature → should return all zeros even if density is high
        fill!(node.significanceMap, 0f0)
        thresholdVal = massCutoffThreshold!(node, density, 5.0f0)

        @test thresholdVal == 5.0f0
        @test sum(node.thresholdMap) == 0f0

        # Non-zero sig but low density → should not pass
        node.significanceMap[cx, cy, cz] = 1.0f0
        density[cx, cy, cz] = 2.0f0  # Below cutoff of 5
        thresholdVal = massCutoffThreshold!(node, density, 5.0f0)

        @test node.thresholdMap[cx, cy, cz] == 0.0f0

        # Non-zero sig AND high density → should pass
        density[cx, cy, cz] = 10.0f0  # Above cutoff
        thresholdVal = massCutoffThreshold!(node, density, 5.0f0)

        @test node.thresholdMap[cx, cy, cz] == 1.0f0
    end

    @testset "thresholdedAverageDensity" begin
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = fill(10.0f0, N, N, N)

        # No thresholded voxels → returns 0
        fill!(node.thresholdMap, 0f0)
        @test thresholdedAverageDensity(node, density) == 0f0

        # Mark some voxels with known density
        node.thresholdMap[1, 1, 1] = 1.0f0
        density[1, 1, 1] = 100.0f0
        node.thresholdMap[2, 2, 2] = 1.0f0
        density[2, 2, 2] = 200.0f0

        # Average should be (100 + 200) / 2 = 150
        @test thresholdedAverageDensity(node, density) == 150.0f0
    end

    @testset "averageDensityThreshold! (Search Logic)" begin
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = fill(10.0f0, N, N, N)

        # Setup: Clear maps
        fill!(node.significanceMap, 0f0)
        fill!(node.thresholdMap, 0f0)

        # Scenario:
        # P1: Sig=30, Rho=500 -> Avg=500
        # P2: Sig=20, Rho=300 -> Avg(P1+P2)=400
        # P3: Sig=10, Rho=100 -> Avg(P1+P2+P3)=300

        node.significanceMap[1, 1, 1] = 10.0f0
        density[1, 1, 1] = 100.0f0

        node.significanceMap[2, 2, 2] = 20.0f0
        density[2, 2, 2] = 300.0f0

        node.significanceMap[3, 3, 3] = 30.0f0
        density[3, 3, 3] = 500.0f0

        # Test A: Target Density = 400
        # Expect to include P1 and P2 (Sig >= 20)
        # Threshold should be 20.0 (signature of P2)
        threshA = averageDensityThreshold!(node, density, 400.0f0)

        @test threshA == 20.0f0
        @test node.thresholdMap[3, 3, 3] == 1.0f0 # P1
        @test node.thresholdMap[2, 2, 2] == 1.0f0 # P2
        @test node.thresholdMap[1, 1, 1] == 0.0f0 # P3
        @test thresholdedAverageDensity(node, density) >= 400.0f0
        @test thresholdedAverageDensity(node, density) == 400.0f0

        # Test B: Target Density = 450
        # Expect to include only P1 (Sig >= 30, Avg=500)
        # If we included P2, Avg=400 < 450.
        threshB = averageDensityThreshold!(node, density, 450.0f0)

        @test threshB == 30.0f0
        @test node.thresholdMap[3, 3, 3] == 1.0f0
        @test node.thresholdMap[2, 2, 2] == 0.0f0
        @test thresholdedAverageDensity(node, density) == 500.0f0

        # Test C: Target Density = 200
        # Expect to include all (P1+P2+P3, Avg=300 >= 200)
        threshC = averageDensityThreshold!(node, density, 200.0f0)

        @test threshC == 10.0f0
        @test sum(node.thresholdMap) == 3.0f0
        @test thresholdedAverageDensity(node, density) == 300.0f0

        # Test D: Impossible Target
        # Target = 600 (Max available is 500)
        # So it returns threshold of top voxel.
        threshD = averageDensityThreshold!(node, density, 600.0f0)
        @test threshD == 30.0f0
        @test node.thresholdMap[3, 3, 3] == 1.0f0

        # Test E: Empty map
        fill!(node.significanceMap, 0f0)
        threshE = averageDensityThreshold!(node, density, 100.0f0)
        @test threshE == 0f0
        @test sum(node.thresholdMap) == 0f0
    end


    @testset "calculateΔM²" begin
        # Create a simple scenario: 
        # A uniform density field (ρ=1)
        # Signatures that increase linearly with x: S(x) = x
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = fill(1.0f0, N, N, N)

        for i in 1:N, j in 1:N, k in 1:N
            node.significanceMap[i, j, k] = Float32(i)
        end

        # S ranges from 1 to 16
        # Mass(> S) roughly proportional to (N - S + 1) * N * N
        # This is a bit complex to predict exactly due to log bins, 
        # but we can check properties.

        nBins = 10
        logS, dM2 = calculateΔM²(node, density, nBins)

        @test length(logS) == nBins
        @test length(dM2) == nBins

        # dM2 should be non-negative
        @test all(dM2 .>= 0)

        # Test with zero signatures - should return empty
        fill!(node.significanceMap, 0f0)
        logS_empty, dM2_empty = calculateΔM²(node, density)
        @test isempty(logS_empty)
        @test isempty(dM2_empty)

        # Test with constant signature - should return empty (min ≈ max)
        fill!(node.significanceMap, 5.0f0)
        logS_const, dM2_const = calculateΔM²(node, density)
        @test isempty(logS_const)
        @test isempty(dM2_const)
    end


    @testset "deltaMSquaredThreshold!" begin
        # Create abstract feature with fake signature and density
        node = NodeFeature((N, N, N), kx, ky, kz)
        density = fill(1.0f0, N, N, N)

        # Create a peak in S at value 10
        # For simplicity, we just need `calculateΔM²` to return something valid
        # Let's fill with ranom data that has range
        for i in eachindex(node.significanceMap)
            node.significanceMap[i] = Float32(i)
        end
        # Range 1 to 4096

        # Should run without error and return a positive threshold
        thresholdVal = deltaMSquaredThreshold!(node, density)
        @test thresholdVal > 0
        @test sum(node.thresholdMap) > 0 # Some voxels should pass

        # Test empty case
        fill!(node.significanceMap, 0f0)
        thresholdVal = deltaMSquaredThreshold!(node, density)
        @test thresholdVal == 0f0
    end


    @testset "componentErosionThresholds" begin
        # Create a synthetic scenario with 3 distinct components
        # C1: Max S = 100
        # C2: Max S = 50
        # C3: Max S = 10

        node = NodeFeature((N, N, N), kx, ky, kz)
        fill!(node.significanceMap, 0f0)

        # C1
        node.significanceMap[1, 1, 1] = 100.0f0
        # C2
        node.significanceMap[5, 5, 5] = 50.0f0
        # C3
        node.significanceMap[10, 10, 10] = 10.0f0

        # Test 50% percentile threshold
        # Total = 3. 
        # > 100 : 0 surviving (0%)
        # > 50  : 1 surviving (33%) -> C1
        # > 10  : 2 surviving (66%) -> C1, C2
        # > 0   : 3 surviving (100%) -> C1, C2, C3

        # Target 50%: Should fall between S=10 and S=50 (approx 20-30ish)
        # If we target 0.5 (50% survival), it should pick a threshold where 1.5 components survive?
        # Survival fraction at S=10 is 66%, at S=50 is 33%.
        # So 50% is roughly halfway in log space between 10 and 50.

        t50 = componentErosionPercentileThreshold!(node, 0.5, nBins=20)
        @test t50 > 10.0f0
        @test t50 < 100.0f0

        # Test Plateau
        # With only 3 points, derivative is noisy, but let's just ensure it runs
        tPlateau = componentErosionPlateauThreshold!(node, nBins=20)
        @test tPlateau > 0.0f0

        # Empty case
        fill!(node.significanceMap, 0f0)
        @test componentErosionPercentileThreshold!(node) == 0f0
        @test componentErosionPlateauThreshold!(node) == 0f0
    end

end
