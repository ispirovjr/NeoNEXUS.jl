# Tests for pipeline orchestration and callable interfaces
@testset "Pipeline" begin

    @testset "Runner Configuration" begin
        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        features = AbstractFeature[
            SheetFeature((threshold=0.1,)),
            LineFeature((threshold=0.2,))
        ]
        scales = [1.0, 2.0, 4.0]
        
        runner = NeoNEXUSRunner(filter, features, scales)
        
        @test runner.filter isa NeoNEXUS.AbstractScaleFilter
        @test length(runner.features) == 2
        @test runner.scales == [1.0, 2.0, 4.0]
    end

    #==========================================================================
    Callable Interface:
    Features and filters are callable structs (functors).
    Unimplemented callables throw ErrorException.
    ==========================================================================#

    @testset "Feature Callable Fallback" begin
        line = LineFeature(())
        field = randn(Float32, 4, 4, 4)
        @test_throws ErrorException line(field, Default)
    end

    @testset "Filter Callable Fallback" begin
        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        field = randn(Float32, 4, 4, 4)
        @test_throws ErrorException filter(field, 1.0)
    end

end
