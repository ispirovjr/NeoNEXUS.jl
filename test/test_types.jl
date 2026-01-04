# Tests for type hierarchy and feature/filter instantiation
@testset "Types" begin

    #==========================================================================
    Type Hierarchy:
    
        AbstractFeature
            └── AbstractMorphologicalFeature
                    ├── SheetFeature (walls)
                    └── LineFeature (filaments)
        
        AbstractScaleFilter
            └── GaussianFourierFilter
    ==========================================================================#

    @testset "Type Hierarchy" begin
        @test AbstractMorphologicalFeature <: AbstractFeature
        @test SheetFeature <: AbstractMorphologicalFeature
        @test LineFeature <: AbstractMorphologicalFeature
    end

    @testset "Feature Instantiation" begin
        # Features now require grid size and wavenumbers
        gridSize = (4, 4, 4)
        indices = ones(Float64, 4) # Must match grid dimension (Nx=4)
        kx = indices; ky = indices; kz = indices

        sheet = SheetFeature(gridSize, kx, ky, kz)
        
        # LineFeature uses default constructor currently (requires maps)
        # Creating dummy maps for testing
        sigMap = zeros(Float32, gridSize)
        respMap = zeros(Float32, gridSize)
        line = LineFeature(sigMap, respMap, kx, ky, kz)
        
        @test sheet isa AbstractFeature
        @test line isa AbstractFeature
        
        # Parameters reference removed in upstream API
        # @test sheet.parameters.threshold == 0.1
    end

    @testset "Signature Methods" begin
        @test Default isa SignatureMethod
        @test NexusPlus isa SignatureMethod
        @test Default != NexusPlus
    end

    @testset "Cache Modes" begin
        @test Read isa CacheMode
        @test Write isa CacheMode
    end

end
