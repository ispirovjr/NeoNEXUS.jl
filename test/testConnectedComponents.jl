# Unit tests for Connected Components

@testset "Connected Components" begin

    @testset "findConnectedComponents with synthetic clumps" begin
        N = 16
        sigMap = zeros(Float32, N, N, N)

        # Create 3 separate clumps at different locations
        # Clump 1: 3x3x3 cube at (2,2,2)
        for i in 2:4, j in 2:4, k in 2:4
            sigMap[i, j, k] = 1.0f0
        end

        # Clump 2: 2x2x2 cube at (10,10,10) 
        for i in 10:11, j in 10:11, k in 10:11
            sigMap[i, j, k] = 2.0f0
        end

        # Clump 3: Single voxel at (14,14,14)
        sigMap[14, 14, 14] = 3.0f0

        # Total non-zero voxels
        nonzeroCount = count(x -> x > 0, sigMap)
        @test nonzeroCount == 27 + 8 + 1  # 36 voxels total

        # Find connected components
        components = findConnectedComponents(sigMap)

        # Should find exactly 3 components
        @test length(components) == 3

        # Components should have fewer entries than non-zero voxels
        @test length(components) < nonzeroCount

        # Check component sizes (sorted by totalSignature descending)
        sizes = sort([length(cc) for cc in components], rev=true)
        @test 27 in sizes  # 3x3x3 clump
        @test 8 in sizes   # 2x2x2 clump
        @test 1 in sizes   # single voxel

        # Total voxels across all components equals non-zero count
        totalVoxels = sum(length(cc) for cc in components)
        @test totalVoxels == nonzeroCount
    end

    @testset "labelConnectedComponents" begin
        N = 8
        sigMap = zeros(Float32, N, N, N)

        # Two separate clumps
        sigMap[1, 1, 1] = 1.0f0
        sigMap[1, 1, 2] = 1.0f0  # Adjacent to first

        sigMap[5, 5, 5] = 2.0f0  # Separate clump

        labelMap, components = labelConnectedComponents(sigMap)

        # Should have 2 components
        @test length(components) == 2

        # Adjacent voxels should have same label
        @test labelMap[1, 1, 1] == labelMap[1, 1, 2]

        # Separate clump should have different label
        @test labelMap[5, 5, 5] != labelMap[1, 1, 1]

        # Background should be 0
        @test labelMap[4, 4, 4] == 0
    end

    @testset "ConnectedComponent struct" begin
        voxels = [CartesianIndex(1, 1, 1), CartesianIndex(1, 1, 2)]
        cc = ConnectedComponent(1, voxels, 5.0f0)

        @test cc.id == 1
        @test length(cc) == 2
        @test cc.totalSignature == 5.0f0
        @test NeoNEXUS.volume(cc) == 2
    end

    @testset "Edge cases" begin
        N = 8

        # Empty signature map
        emptyMap = zeros(Float32, N, N, N)
        components = findConnectedComponents(emptyMap)
        @test length(components) == 0

        # Single voxel
        singleMap = zeros(Float32, N, N, N)
        singleMap[4, 4, 4] = 1.0f0
        components = findConnectedComponents(singleMap)
        @test length(components) == 1
        @test length(components[1]) == 1

        # All non-zero (one giant component)
        fullMap = ones(Float32, N, N, N)
        components = findConnectedComponents(fullMap)
        @test length(components) == 1
        @test length(components[1]) == N^3
    end

    @testset "componentDensityThreshold!" begin
        N = 16
        kx = ky = kz = FFTW.fftfreq(N) .* N .* 2π

        # Create a feature with manual signature
        node = NodeFeature((N, N, N), kx, ky, kz)
        fill!(node.significanceMap, 0f0)

        # Create density field with two regions of different densities
        density = ones(Float32, N, N, N)

        # Component 1: High density clump at (2,2,2) - 2x2x2
        for i in 2:3, j in 2:3, k in 2:3
            node.significanceMap[i, j, k] = 1.0f0
            density[i, j, k] = 100.0f0  # High density
        end

        # Component 2: Low density clump at (10,10,10) - 2x2x2
        for i in 10:11, j in 10:11, k in 10:11
            node.significanceMap[i, j, k] = 1.0f0
            density[i, j, k] = 10.0f0  # Low density
        end

        # Test with low density cutoff - both should qualify
        result = componentDensityThreshold!(node, density, 5.0f0)
        @test result[1] == 2  # Both qualify (avg density > 5)
        @test result[2] == 2  # Total 2 components
        @test sum(node.thresholdMap) == 16  # All 16 voxels marked (8+8)

        # Test with medium density cutoff - only high density should qualify
        result = componentDensityThreshold!(node, density, 50.0f0)
        @test result[1] == 1  # Only high-density qualifies
        @test result[2] == 2  # Total 2 components
        @test sum(node.thresholdMap) == 8  # Only 8 voxels marked

        # Verify the high-density voxels are marked
        @test node.thresholdMap[2, 2, 2] == 1.0f0
        @test node.thresholdMap[10, 10, 10] == 0.0f0
    end

end
