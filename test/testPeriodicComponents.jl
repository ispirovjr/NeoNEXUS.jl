using Test, NeoNEXUS

@testset "Periodic connected components" begin
    @testset "Opposite faces on each axis, non-cubic grid" begin
        dims = (5, 7, 9)
        for axis in 1:3
            field = zeros(Float32, dims)
            firstVoxel = CartesianIndex(3, 4, 5)
            low = CartesianIndex(ntuple(i -> i == axis ? 1 : firstVoxel[i], 3))
            high = CartesianIndex(ntuple(i -> i == axis ? dims[i] : firstVoxel[i], 3))
            field[low] = 1f0
            field[high] = 3f0

            @test length(findConnectedComponents(field)) == 2
            @test length(findConnectedComponents(field; periodicBounds=false)) == 2
            periodic = findConnectedComponents(field; periodicBounds=true)
            @test length(periodic) == 1
            @test Set(only(periodic).voxels) == Set([low, high])
            @test only(periodic).maxSignature == 3f0
            labels, components = labelConnectedComponents(field; periodicBounds=true)
            @test length(components) == 1
            @test labels[low] == labels[high] != 0
        end
    end

    @testset "Corner wrapping and face connectivity" begin
        field = zeros(Float32, 5, 7, 9)
        for x in (1, 5), y in (1, 7), z in (1, 9)
            field[x, y, z] = 1f0
        end
        @test length(findConnectedComponents(field; periodicBounds=false)) == 8
        periodic = findConnectedComponents(field; periodicBounds=true)
        @test length(periodic) == 1
        @test length(only(periodic)) == 8

        fill!(field, 0f0)
        field[1, 1, 1] = field[5, 7, 9] = 1f0
        @test length(findConnectedComponents(field; periodicBounds=true)) == 2
        field[5, 7, 9] = -1f0
        @test length(findConnectedComponents(field; periodicBounds=true)) == 1
    end

    @testset "Empty and singleton axes" begin
        for dims in ((0, 3, 4), (3, 0, 4), (3, 4, 0))
            @test isempty(findConnectedComponents(zeros(Float32, dims); periodicBounds=true))
        end
        for dims in ((1, 1, 1), (1, 2, 1), (1, 4, 7))
            periodic = findConnectedComponents(ones(Float32, dims); periodicBounds=true)
            @test length(periodic) == 1
            @test length(only(periodic)) == prod(dims)
            @test length(unique(only(periodic).voxels)) == prod(dims)
        end
    end

    @testset "Pruning uses the merged volume and mass" begin
        field = zeros(Float32, 8, 8, 8)
        field[1, 3, 4] = field[8, 3, 4] = 1f0
        bounded = copy(field)
        periodic = copy(field)
        @test pruneSmallComponents!(bounded, 2; periodicBounds=false) == (0, 2, 2)
        @test pruneSmallComponents!(periodic, 2; periodicBounds=true) == (1, 0, 0)
        @test periodic == field
        density = fill(2f0, size(field))
        @test pruneSmallMassComponents!(copy(field), density, 3; periodicBounds=false) == (0, 2, 2, 4.0)
        @test pruneSmallMassComponents!(copy(field), density, 3; periodicBounds=true) == (1, 0, 0, 0.0)
    end

    @testset "Node density and percentage thresholds" begin
        node = NodeFeature((8, 8, 8), zeros(8), zeros(8), zeros(8))
        node.significanceMap[1, 3, 4] = 3f0
        node.significanceMap[8, 3, 4] = 1f0
        node.significanceMap[4, 4, 4] = 2f0
        density = zeros(Float32, 8, 8, 8)
        density[1, 3, 4] = 600f0

        @test componentDensityThreshold!(node, density, 370; periodicBounds=false) == (1, 3)
        @test componentDensityThreshold!(node, density, 370; periodicBounds=true) == (0, 2)
        @test findComponentPercentageThreshold!(node, density, 370, 0.5;
            nBins=3, excludeEmpty=false, periodicBounds=false) == 1f0
        @test findComponentPercentageThreshold!(node, density, 370, 0.5;
            nBins=3, excludeEmpty=false, periodicBounds=true) == 2f0
        @test node.thresholdMap[8, 3, 4] == 0f0
        @test node.thresholdMap[1, 3, 4] == 1f0

        _, boundedSurvival = NeoNEXUS.calculateComponentSurvival(node, 3; periodicBounds=false)
        _, periodicSurvival = NeoNEXUS.calculateComponentSurvival(node, 3; periodicBounds=true)
        @test first(boundedSurvival) == first(periodicSurvival) == 1.0
        @test last(boundedSurvival) ≈ 1 / 3
        @test last(periodicSurvival) ≈ 1 / 2
        @test pruneSmallMassComponents!(node, density, 1; periodicBounds=true)[1:2] == (1, 1)
    end

    @testset "NEXUSPlus forwards the option in both execution modes" begin
        density = fill(0.001f0, 8, 8, 8)
        density[1, 3, 4] = 600f0
        for multithread in (false, true), periodicBounds in (false, true)
            # Empty scale list isolates thresholding from signature extraction.
            runner = NEXUSPlus(8, Float64[])
            runner.node.significanceMap[1, 3, 4] = 3f0
            runner.node.significanceMap[8, 3, 4] = 1f0
            runner.node.significanceMap[4, 4, 4] = 2f0
            result = runner(density; multithread=multithread, periodicBounds=periodicBounds)
            @test periodicBounds ? 1f0 < result.nodeThres < 2f0 : result.nodeThres == 1f0
        end
    end
end
