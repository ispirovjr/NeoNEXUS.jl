# Tests for pipeline orchestration and callable interfaces
@testset "Pipeline" begin

    @testset "Runner Configuration" begin
        # Dummy grid info for feature construction
        grid = (4, 4, 4)
        k = [1.0]
        
        # Construct actual feature instances
        sheet = SheetFeature(grid, k, k, k)
        # LineFeature needs manual maps
        sig = zeros(Float32, grid)
        resp = zeros(Float32, grid)
        line = LineFeature(sig, resp, k, k, k)

        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        features = AbstractFeature[sheet, line]
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
        # Dummy inputs
        grid = (4, 4, 4)
        k = ones(Float64, 4) # 1D vector matching dimension size
        line = LineFeature(zeros(Float32, grid), zeros(Float32, grid), k, k, k)
        field = randn(Float32, 4, 4, 4)
        # New API: feature(field, cache, mode). `Default` method arg removed.
        # Calling unimplemented feature (LineFeature) fails at computeSignature -> MethodError
        @test_throws MethodError line(field)
    end

    @testset "Filter Callable Fallback" begin
        filter = NeoNEXUS.GaussianFourierFilter((sigma=1.0,))
        field = randn(Float32, 4, 4, 4)
        @test_throws ErrorException filter(field, 1.0)
    end

end
