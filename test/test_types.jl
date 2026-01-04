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
        sheet = SheetFeature((threshold=0.1, smoothing=2.0))
        line = LineFeature((threshold=0.2,))
        
        @test sheet isa AbstractFeature
        @test line isa AbstractFeature
        @test sheet.parameters.threshold == 0.1
        @test line.parameters.threshold == 0.2
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
